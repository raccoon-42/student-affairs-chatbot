import sys
import time
import uuid
from functools import lru_cache
from threading import Lock, Thread

import requests

from fastapi import Cookie, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import auth, storage
from app.conversation import Conversation, QueryRewriter, STAGE_MARKERS
from app.ratelimit import RateLimiter
from app.guardrails import ScopeGate, ABUSE_MESSAGE, REFUSALS
from app.llm import OpenRouterLLM
from app.retrieval import default_retriever
from config import settings

app = FastAPI(
    title="Student Affairs Chatbot API",
    description="API for querying academic calendar and regulations information",
    version="1.0.0",
)

SESSION_TTL_SECONDS = 30 * 60


# --- localized error messages -------------------------------------------
# Every user-facing HTTP error is a key here; LocalizedError carries the key
# and _localized_error renders it in the caller's language. Language is taken
# from ?lang=, then the X-Lang header (the web UI sets it on every call),
# then Accept-Language — defaulting to Turkish.
MESSAGES = {
    "model_unsupported": {"tr": "Bu model desteklenmiyor",
                          "en": "This model is not supported"},
    "query_too_long": {"tr": "Mesaj çok uzun.",
                       "en": "The message is too long."},
    "service_unavailable": {
        "tr": "Şu anda yanıt veremiyorum — hizmette geçici bir sorun var. "
              "Lütfen daha sonra tekrar dene.",
        "en": "I can't answer right now — the service is having a temporary "
              "issue. Please try again later."},
    "google_auth_failed": {"tr": "Google kimliği doğrulanamadı",
                           "en": "Google sign-in could not be verified"},
    "no_session": {"tr": "Oturum bulunamadı", "en": "No active session"},
    "invalid_education": {"tr": "Geçersiz eğitim türü",
                          "en": "Invalid education type"},
    "not_your_conversation": {"tr": "Bu konuşma size ait değil",
                              "en": "This conversation isn't yours"},
    "invalid_conversation": {"tr": "Geçersiz konuşma içeriği",
                             "en": "Invalid conversation content"},
    "admin_forbidden": {"tr": "Bu sayfaya erişim yetkin yok",
                        "en": "You don't have access to this page"},
    "abuse_blocked": {
        "tr": "Uygunsuz dil nedeniyle mesaj gönderimin geçici olarak engellendi.",
        "en": "Messaging is temporarily blocked due to inappropriate language."},
    "spam_brake": {
        "tr": "Art arda çok fazla konu dışı ya da uygunsuz mesaj gönderdin. "
              "Lütfen bir süre sonra tekrar dene.",
        "en": "You've sent too many off-topic or inappropriate messages in a "
              "row. Please try again later."},
    "limit_user": {
        "tr": "Mesaj limitine ulaştın. Lütfen bir süre sonra tekrar dene.",
        "en": "You've reached the message limit. Please try again later."},
    "limit_anon": {
        "tr": "Mesaj limitine ulaştın. Google ile giriş yaparak daha yüksek "
              "bir limitle devam edebilirsin.",
        "en": "You've reached the message limit. Sign in with Google to "
              "continue with a higher limit."},
    "stt_not_configured": {"tr": "Ses tanıma yapılandırılmamış.",
                           "en": "Speech recognition isn't configured."},
    "stt_limit": {
        "tr": "Ses tanıma limitine ulaştın. Lütfen bir süre sonra tekrar dene.",
        "en": "You've reached the speech-recognition limit. Please try "
              "again later."},
    "no_audio": {"tr": "Ses verisi yok.", "en": "No audio data."},
    "audio_too_long": {"tr": "Kayıt çok uzun.", "en": "The recording is too long."},
    "stt_failed": {"tr": "Ses tanıma başarısız oldu.",
                   "en": "Speech recognition failed."},
    "invalid_image": {"tr": "Geçersiz görsel.", "en": "Invalid image."},
    "image_too_large": {"tr": "Görsel çok büyük.", "en": "The image is too large."},
    "empty_message": {"tr": "Boş mesaj.", "en": "Empty message."},
}

# suffix appended to a partial streamed answer when the upstream drops
BROKE_OFF = {
    "tr": "\n\n*(Bağlantı hatası — yanıtın devamı alınamadı.)*",
    "en": "\n\n*(Connection error — the rest of the answer could not be retrieved.)*",
}


def _lang(request: Request) -> str:
    lang = request.query_params.get("lang") or request.headers.get("x-lang")
    if lang in ("tr", "en"):
        return lang
    accept = (request.headers.get("accept-language") or "").lower()
    return "en" if accept.startswith("en") else "tr"


class LocalizedError(HTTPException):
    """An HTTP error whose detail is a MESSAGES key; _localized_error renders
    it in the caller's language, so routes never hardcode a language."""
    def __init__(self, status_code: int, key: str, headers: dict | None = None):
        super().__init__(status_code=status_code, detail=key, headers=headers)
        self.key = key


@app.exception_handler(LocalizedError)
async def _localized_error(request: Request, exc: LocalizedError):
    return JSONResponse(status_code=exc.status_code,
                        content={"detail": MESSAGES[exc.key][_lang(request)]},
                        headers=exc.headers)


class ChatResponse(BaseModel):
    query: str
    response: str
    model: str
    session_id: str


@lru_cache
def _backend(name: str):
    """LLM, retriever and model names are shared by every session;
    only the conversation history is per-session."""
    retriever = default_retriever()
    llm = OpenRouterLLM()
    return llm, retriever, settings.OPENROUTER_MODEL, settings.GUARD_MODEL


_sessions: dict = {}  # (backend, session_id) -> (Conversation, last_used)
_sessions_lock = Lock()


def get_conversation(session_id: str, backend: str = "openrouter") -> Conversation:
    """One conversation per session id, evicted after SESSION_TTL_SECONDS
    of inactivity. On a cache miss the history of a persisted conversation
    is rehydrated from SQLite, so restarts don't lose context."""
    now = time.time()
    with _sessions_lock:
        for key in [k for k, (_, last_used) in _sessions.items()
                    if now - last_used > SESSION_TTL_SECONDS]:
            del _sessions[key]

        key = (backend, session_id)
        if key not in _sessions:
            llm, retriever, model, small_model = _backend(backend)
            conversation = Conversation(llm, retriever, model,
                                        gate=ScopeGate(llm, small_model),
                                        rewriter=QueryRewriter(llm, small_model))
            stored = storage.conversation_messages(session_id)
            if stored:
                conversation.load_history(stored)
            _sessions[key] = (conversation, now)
        conversation, _ = _sessions[key]
        _sessions[key] = (conversation, now)
        return conversation


_anon_limiter = RateLimiter(settings.CHAT_LIMIT_ANON)
_user_limiter = RateLimiter(settings.CHAT_LIMIT_USER)
# spam brake: repeated gate refusals (gibberish, off-topic floods, abuse)
_refusal_limiter = RateLimiter(settings.CHAT_REFUSAL_LIMIT, window_seconds=600)
# abuse block: a single abusive message locks the sender out for the window
_abuse_limiter = RateLimiter(limit=1, window_seconds=settings.ABUSE_BLOCK_MINUTES * 60)
# voice input has its own quota so a transcription doesn't eat a chat slot
_stt_limiter = RateLimiter(settings.STT_LIMIT)


def _limit_key(request: Request, user) -> str:
    return user["email"] if user else (request.client.host if request.client else "unknown")


def _enforce_rate_limit(request: Request, user) -> None:
    """Per-email for signed-in users, per-IP for anonymous. The anonymous
    429 message doubles as the sign-in nudge shown by the UI."""
    if _limit_key(request, user) in settings.RATELIMIT_EXEMPT:
        return  # judge suite / local eval runs
    if (_limit_key(request, user) not in settings.ABUSE_EXEMPT
            and _abuse_limiter.is_blocked(_limit_key(request, user))):
        raise LocalizedError(429, "abuse_blocked", headers={"X-Block-Reason": "abuse"})
    if _refusal_limiter.is_blocked(_limit_key(request, user)):
        raise LocalizedError(429, "spam_brake")
    if user:
        if not _user_limiter.allow(user["email"]):
            raise LocalizedError(429, "limit_user")
    else:
        if not _anon_limiter.allow(request.client.host if request.client else "unknown"):
            raise LocalizedError(429, "limit_anon")


def _record_turn_usage(key: str, entries: list) -> None:
    """One usage_log row per LLM call of a finished turn (gate, rewrite,
    image, chat), written off-thread. Anonymous traffic counts too, and
    gate rows exist even for refused turns."""
    if not entries:
        return

    def write():
        for entry in entries:
            storage.record_usage(key, entry["model"], entry["kind"],
                                 entry["prompt_tokens"], entry["completion_tokens"],
                                 entry.get("cost"))
    Thread(target=write, daemon=True).start()


def _register_refusal(request: Request, user, response_text: str) -> None:
    """Called after the answer is known — canned refusals count toward
    the spam brake; abuse additionally triggers the immediate block."""
    if response_text in REFUSALS.values():
        _refusal_limiter.record(_limit_key(request, user))
    if response_text == ABUSE_MESSAGE and _limit_key(request, user) not in settings.ABUSE_EXEMPT:
        _abuse_limiter.record(_limit_key(request, user))


def _authorize_conversation(session_id: str, user) -> None:
    """A persisted conversation may only be continued or read by its
    owner. Anonymous conversation ids never reach the database, so they
    can't collide with anyone's history."""
    owner = storage.conversation_owner(session_id)
    if owner and (user is None or user["email"] != owner):
        raise LocalizedError(403, "not_your_conversation")


class GoogleCredential(BaseModel):
    credential: str


@app.get("/auth/config")
async def auth_config(request: Request, auth_token: str = Cookie(None)):
    # the client id is public by design; null tells the UI to hide sign-in.
    # abuse_message lets the UI recognize the abuse refusal without
    # duplicating the string; abuse_exempt lets a dev skip the client lock.
    user = auth.sessions.get(auth_token)
    return {"client_id": settings.GOOGLE_CLIENT_ID, "abuse_message": ABUSE_MESSAGE,
            "off_topic_message": REFUSALS["off_topic"],
            "abuse_block_seconds": settings.ABUSE_BLOCK_MINUTES * 60,
            "abuse_exempt": _limit_key(request, user) in settings.ABUSE_EXEMPT,
            "stt": bool(settings.GROQ_API_KEY),
            "context_window": settings.LLM_CONTEXT_WINDOW}


@app.post("/auth/google")
async def auth_google(body: GoogleCredential, response: Response):
    try:
        user = auth.verify_google_token(body.credential)
    except ValueError:
        raise LocalizedError(401, "google_auth_failed")
    token = auth.sessions.create(user)
    response.set_cookie("auth_token", token, httponly=True, samesite="lax",
                        secure=settings.COOKIE_SECURE,
                        max_age=settings.AUTH_SESSION_TTL_DAYS * 24 * 3600)
    # read back from storage: a returning user already has education_type
    return auth.sessions.get(token)


@app.get("/auth/me")
async def auth_me(auth_token: str = Cookie(None)):
    return auth.sessions.get(auth_token)


class Profile(BaseModel):
    education_type: str


@app.post("/auth/profile")
async def auth_profile(body: Profile, auth_token: str = Cookie(None)):
    user = auth.sessions.get(auth_token)
    if user is None:
        raise LocalizedError(401, "no_session")
    if body.education_type not in auth.EDUCATION_TYPES:
        raise LocalizedError(422, "invalid_education")
    storage.set_education_type(user["email"], body.education_type)
    return auth.sessions.get(auth_token)


@app.post("/auth/logout")
async def auth_logout(response: Response, auth_token: str = Cookie(None)):
    auth.sessions.drop(auth_token)
    response.delete_cookie("auth_token")
    return {"ok": True}


@app.get("/conversations")
async def conversations(auth_token: str = Cookie(None)):
    user = auth.sessions.get(auth_token)
    if user is None:
        raise LocalizedError(401, "no_session")
    return storage.list_conversations(user["email"])


class ImportedMessage(BaseModel):
    role: str
    text: str


class ConversationImport(BaseModel):
    id: str
    messages: list[ImportedMessage]
    title: str | None = None  # the model-written title the browser already has


@app.post("/conversations/import")
async def conversation_import(body: ConversationImport, auth_token: str = Cookie(None)):
    """Adopt the anonymous conversation a browser kept locally, so it
    continues seamlessly after sign-in."""
    user = auth.sessions.get(auth_token)
    if user is None:
        raise LocalizedError(401, "no_session")
    owner = storage.conversation_owner(body.id)
    if owner == user["email"]:
        return {"ok": True}  # already adopted (e.g. repeated sign-in)
    if owner:
        raise LocalizedError(403, "not_your_conversation")
    if not (0 < len(body.messages) <= 200) or any(len(m.text) > 20000 for m in body.messages):
        raise LocalizedError(422, "invalid_conversation")
    storage.import_conversation(body.id, user["email"], [
        {"role": "assistant" if m.role == "bot" else "user", "content": m.text}
        for m in body.messages
    ], title=body.title[:80] if body.title else None)
    return {"ok": True}


@app.get("/conversations/{conversation_id}")
async def conversation_detail(conversation_id: str, auth_token: str = Cookie(None)):
    user = auth.sessions.get(auth_token)
    _authorize_conversation(conversation_id, user)
    return {"id": conversation_id, "messages": storage.conversation_messages(conversation_id)}


@app.delete("/conversations/{conversation_id}")
async def conversation_delete(conversation_id: str, auth_token: str = Cookie(None)):
    user = auth.sessions.get(auth_token)
    if user is None:
        raise LocalizedError(401, "no_session")
    _authorize_conversation(conversation_id, user)
    storage.delete_conversation(conversation_id, user["email"])
    with _sessions_lock:
        _sessions.pop(("openrouter", conversation_id), None)
    return {"ok": True}


@app.get("/chat/sources")
async def chat_sources(session_id: str, auth_token: str = Cookie(None)):
    """What the current conversation's latest answer was grounded on.
    Reads the cached conversation only — never creates one."""
    user = auth.sessions.get(auth_token)
    _authorize_conversation(session_id, user)
    with _sessions_lock:
        cached = _sessions.get(("openrouter", session_id))
    if cached and cached[0].last_sources:
        # usage rides along: the UI fetches this after every answer, so the
        # token counter needs no extra request
        return {"sources": cached[0].last_sources, "usage": cached[0].last_usage}
    # the cache is per-process: a restart (e.g. --reload in dev) between
    # streaming and this call would lose the sources — fall back to the
    # persisted transcript's latest assistant message
    for message in reversed(storage.conversation_messages(session_id)):
        if message["role"] == "assistant" and message["sources"]:
            return {"sources": message["sources"], "usage": None}
    return {"sources": [], "usage": None}


@app.get("/chat/title")
async def chat_title(request: Request, session_id: str, auth_token: str = Cookie(None)):
    """A short LLM-written title for the conversation's first exchange —
    the UI swaps its first-question placeholder for it. Reads the cached
    conversation only; the chat that filled it was already rate-limited,
    and the UI asks once per conversation."""
    user = auth.sessions.get(auth_token)
    _authorize_conversation(session_id, user)
    with _sessions_lock:
        cached = _sessions.get(("openrouter", session_id))
    if not cached:
        return {"title": None}
    _, _, _, small_model = _backend("openrouter")
    try:
        title, usage = cached[0].suggest_title(small_model)
    except Exception as error:  # a failed title is not worth an error state
        print(f"[title] generation failed: {error}", file=sys.stderr)
        return {"title": None}
    if usage:
        storage.record_usage(_limit_key(request, user), small_model, "title",
                             usage["prompt_tokens"], usage["completion_tokens"],
                             usage.get("cost"))
    if not title or len(title) > 80:  # a rambling model is not a title
        return {"title": None}
    storage.set_conversation_title(session_id, title)  # no-op for anonymous
    return {"title": title}


@app.get("/chat/debug")
async def chat_debug(session_id: str, auth_token: str = Cookie(None)):
    """CLI-style log lines for the current conversation's latest turn —
    shown in the UI when developer mode is on. Cached conversation only."""
    user = auth.sessions.get(auth_token)
    _authorize_conversation(session_id, user)
    with _sessions_lock:
        cached = _sessions.get(("openrouter", session_id))
    return {"debug": cached[0].last_debug if cached else [],
            "reference": cached[0].last_reference if cached else [],
            "usage": cached[0].last_usage if cached else None}


@app.get("/usage/me")
async def usage_me(request: Request, days: float = 30, auth_token: str = Cookie(None)):
    """The caller's own LLM spend, broken down by pipeline step — feeds
    the settings Usage tab. Anonymous callers see their IP's usage."""
    user = auth.sessions.get(auth_token)
    return {"days": days, "usage": storage.usage_by_kind(_limit_key(request, user), days)}


def _require_admin(auth_token) -> None:
    user = auth.sessions.get(auth_token)
    if user is None or user["email"] not in settings.ADMIN_EMAILS:
        raise LocalizedError(403, "admin_forbidden")


@app.get("/admin/usage")
async def admin_usage(days: float = 7, kind: str = None, auth_token: str = Cookie(None)):
    """Per-person token totals (email or IP) over the last `days`, biggest
    spender first — for spotting outliers. `kind` narrows to one call
    type (chat/gate/rewrite/image/title). ADMIN_EMAILS only; the pretty
    face is /admin.html."""
    _require_admin(auth_token)
    return {"days": days, "kind": kind, "usage": storage.usage_by_key(days, kind)}


@app.get("/admin/usage/detail")
async def admin_usage_detail(key: str, days: float = 7, kind: str = None,
                             auth_token: str = Cookie(None)):
    """One person's spend broken down by pipeline step and model — the
    expandable row in /admin.html."""
    _require_admin(auth_token)
    return {"key": key, "days": days, "usage": storage.usage_by_kind(key, days, kind)}


def _validate_model(model_name: str) -> None:
    """model_name is client-supplied — only allowlisted models may run,
    so the public API can't burn the OpenRouter key on arbitrary models."""
    if model_name and model_name not in settings.ALLOWED_CHAT_MODELS:
        raise LocalizedError(400, "model_unsupported")


@app.get("/chat", response_model=ChatResponse)
async def chat(request: Request, query: str, session_id: str = None, model_name: str = None,
               auth_token: str = Cookie(None)):
    if len(query) > MAX_QUERY_CHARS:
        raise LocalizedError(413, "query_too_long")
    session_id = session_id or uuid.uuid4().hex
    user = auth.sessions.get(auth_token)
    _enforce_rate_limit(request, user)
    _authorize_conversation(session_id, user)
    _validate_model(model_name)
    model = model_name or settings.OPENROUTER_MODEL
    conversation = get_conversation(session_id)
    try:
        response = conversation.respond(
            query, model, education_type=user["education_type"] if user else None)
    except Exception as error:  # upstream LLM stall/failure — not a server bug
        print(f"[error] chat failed: {error}", file=sys.stderr)  # detail stays server-side
        raise LocalizedError(503, "service_unavailable")
    _register_refusal(request, user, response)
    _record_turn_usage(_limit_key(request, user), list(conversation.turn_usage))
    if user and response not in REFUSALS.values():  # refusals aren't part of the transcript
        storage.record_exchange(session_id, user["email"], query, response,
                                sources=conversation.last_sources)
    return ChatResponse(query=query, response=response, model=model, session_id=session_id)


MAX_IMAGE_CHARS = 8_000_000  # ~6 MB of image as a base64 data URL
MAX_QUERY_CHARS = 4_000  # matches the composer's maxlength; a query
# beyond this is either abuse or a paste that belongs in an image


class ChatStreamBody(BaseModel):
    query: str
    session_id: str | None = None
    model_name: str | None = None
    image: str | None = None  # data URL, current turn only
    lang: str | None = None  # UI language, for error messages


def _stream_chat(request, query, session_id, model_name, auth_token, image=None, lang="tr"):
    lang = lang if lang in ("tr", "en") else "tr"
    if len(query) > MAX_QUERY_CHARS:
        raise LocalizedError(413, "query_too_long")
    session_id = session_id or uuid.uuid4().hex
    user = auth.sessions.get(auth_token)
    _enforce_rate_limit(request, user)
    _authorize_conversation(session_id, user)
    _validate_model(model_name)
    conversation = get_conversation(session_id)
    stream = conversation.respond_stream(
        query, model_name or settings.OPENROUTER_MODEL,
        education_type=user["education_type"] if user else None, image=image)

    def recorded():
        tokens = []
        try:
            for token in stream:
                # stage markers go to the browser (cursor animation) but
                # must stay out of the persisted/refusal-checked answer
                if token not in STAGE_MARKERS:
                    tokens.append(token)
                yield token
        except Exception as error:
            # an upstream failure (e.g. out of OpenRouter credits) must not
            # abort the HTTP stream — the browser would show a raw
            # NetworkError instead of a readable message
            print(f"[error] llm stream failed: {error}", file=sys.stderr)
            # partial answer already on screen: say it broke instead of
            # stopping silently mid-sentence
            yield BROKE_OFF[lang] if tokens else MESSAGES["service_unavailable"][lang]
            return
        answer = "".join(tokens)
        _register_refusal(request, user, answer)
        _record_turn_usage(_limit_key(request, user), list(conversation.turn_usage))
        # persist only after the full answer arrived; refusals aren't
        # part of the transcript. The write happens off-thread: the HTTP
        # stream stays open until this generator returns, so a slow SQLite
        # write would show as a hang after the last visible token.
        if user and answer not in REFUSALS.values():
            # image-only messages have no text; the image itself isn't stored
            Thread(target=storage.record_exchange,
                   args=(session_id, user["email"], query or "(görsel)", answer),
                   kwargs={"sources": conversation.last_sources,
                           "usage": conversation.last_usage},
                   daemon=True).start()

    return StreamingResponse(
        recorded(),
        media_type="text/plain; charset=utf-8",
        headers={"X-Session-Id": session_id},
    )


@app.get("/chat/stream")
async def chat_stream(request: Request, query: str, session_id: str = None,
                      model_name: str = None, auth_token: str = Cookie(None)):
    return _stream_chat(request, query, session_id, model_name, auth_token)


@app.post("/chat/stream")
async def chat_stream_post(request: Request, body: ChatStreamBody,
                           auth_token: str = Cookie(None)):
    """Same as the GET route, but a JSON body can carry an image
    (data URL) that the model sees for this turn."""
    if body.image is not None:
        if not body.image.startswith("data:image/"):
            raise LocalizedError(400, "invalid_image")
        if len(body.image) > MAX_IMAGE_CHARS:
            raise LocalizedError(413, "image_too_large")
    query = body.query.strip()
    if not query and body.image is None:
        raise LocalizedError(400, "empty_message")
    # an image-only message is fine: the conversation extracts the search
    # query from the image itself (see QueryRewriter.image_query)
    return _stream_chat(request, query, body.session_id, body.model_name,
                        auth_token, image=body.image, lang=body.lang or "tr")


MAX_AUDIO_BYTES = 10 * 1024 * 1024  # a minute of browser opus is ~0.5 MB


@app.post("/transcribe")
async def transcribe(request: Request, auth_token: str = Cookie(None)):
    """Voice input: the UI posts the recorded audio blob as the raw request
    body (no multipart, so no extra dependency) and gets back the text to
    put in the composer. Groq hosts the Whisper model. No language param:
    forcing one makes Whisper translate into it — the UI language says
    nothing about which language the student speaks, so auto-detect."""
    if not settings.GROQ_API_KEY:
        raise LocalizedError(503, "stt_not_configured")

    user = auth.sessions.get(auth_token)
    key = _limit_key(request, user)
    if key not in settings.ABUSE_EXEMPT and _abuse_limiter.is_blocked(key):
        raise LocalizedError(429, "abuse_blocked", headers={"X-Block-Reason": "abuse"})
    if not _stt_limiter.allow(key):
        raise LocalizedError(429, "stt_limit")

    audio = await request.body()
    if not audio:
        raise LocalizedError(400, "no_audio")
    if len(audio) > MAX_AUDIO_BYTES:
        raise LocalizedError(413, "audio_too_long")

    content_type = request.headers.get("content-type") or "audio/webm"
    extension = "mp4" if "mp4" in content_type else "ogg" if "ogg" in content_type else "webm"

    groq = requests.post(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {settings.GROQ_API_KEY}"},
        files={"file": (f"audio.{extension}", audio, content_type)},
        data={"model": settings.GROQ_WHISPER_MODEL},
        timeout=60,
    )
    if groq.status_code != 200:
        raise LocalizedError(502, "stt_failed")
    return {"text": groq.json().get("text", "").strip()}


# the web UI — mounted last so the API routes above win the match first
app.mount("/", StaticFiles(directory=settings.ROOT / "web", html=True), name="web")
