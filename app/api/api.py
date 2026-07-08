import time
import uuid
from functools import lru_cache
from threading import Lock

from fastapi import Cookie, FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import auth, storage
from app.conversation import Conversation, QueryRewriter
from app.ratelimit import RateLimiter
from app.guardrails import ScopeGate, ABUSE_MESSAGE, REFUSALS
from app.llm import OpenRouterLLM, OllamaLLM
from app.retrieval import default_retriever
from config import settings

app = FastAPI(
    title="Student Affairs Chatbot API",
    description="API for querying academic calendar and regulations information",
    version="1.0.0",
)

SESSION_TTL_SECONDS = 30 * 60


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
    if name == "ollama":
        # local backend gates and rewrites locally too, so it stays fully offline
        llm = OllamaLLM()
        return llm, retriever, settings.OLLAMA_MODEL, settings.OLLAMA_MODEL
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


def _limit_key(request: Request, user) -> str:
    return user["email"] if user else (request.client.host if request.client else "unknown")


def _enforce_rate_limit(request: Request, user) -> None:
    """Per-email for signed-in users, per-IP for anonymous. The anonymous
    429 message doubles as the sign-in nudge shown by the UI."""
    if (_limit_key(request, user) not in settings.ABUSE_EXEMPT
            and _abuse_limiter.is_blocked(_limit_key(request, user))):
        raise HTTPException(status_code=429,
                            detail="Uygunsuz dil nedeniyle mesaj gönderimin geçici "
                                   "olarak engellendi.",
                            headers={"X-Block-Reason": "abuse"})
    if _refusal_limiter.is_blocked(_limit_key(request, user)):
        raise HTTPException(status_code=429,
                            detail="Art arda çok fazla konu dışı ya da uygunsuz mesaj "
                                   "gönderdin. Lütfen bir süre sonra tekrar dene.")
    if user:
        if not _user_limiter.allow(user["email"]):
            raise HTTPException(status_code=429,
                                detail="Mesaj limitine ulaştın. Lütfen bir süre sonra tekrar dene.")
    else:
        if not _anon_limiter.allow(request.client.host if request.client else "unknown"):
            raise HTTPException(status_code=429,
                                detail="Mesaj limitine ulaştın. Google ile giriş yaparak "
                                       "daha yüksek bir limitle devam edebilirsin.")


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
        raise HTTPException(status_code=403, detail="Bu konuşma size ait değil")


class GoogleCredential(BaseModel):
    credential: str


@app.get("/auth/config")
async def auth_config(request: Request, auth_token: str = Cookie(None)):
    # the client id is public by design; null tells the UI to hide sign-in.
    # abuse_message lets the UI recognize the abuse refusal without
    # duplicating the string; abuse_exempt lets a dev skip the client lock.
    user = auth.sessions.get(auth_token)
    return {"client_id": settings.GOOGLE_CLIENT_ID, "abuse_message": ABUSE_MESSAGE,
            "abuse_block_seconds": settings.ABUSE_BLOCK_MINUTES * 60,
            "abuse_exempt": _limit_key(request, user) in settings.ABUSE_EXEMPT}


@app.post("/auth/google")
async def auth_google(body: GoogleCredential, response: Response):
    try:
        user = auth.verify_google_token(body.credential)
    except ValueError:
        raise HTTPException(status_code=401, detail="Google kimliği doğrulanamadı")
    token = auth.sessions.create(user)
    response.set_cookie("auth_token", token, httponly=True, samesite="lax",
                        max_age=30 * 24 * 3600)
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
        raise HTTPException(status_code=401, detail="Oturum bulunamadı")
    if body.education_type not in auth.EDUCATION_TYPES:
        raise HTTPException(status_code=422, detail="Geçersiz eğitim türü")
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
        raise HTTPException(status_code=401, detail="Oturum bulunamadı")
    return storage.list_conversations(user["email"])


class ImportedMessage(BaseModel):
    role: str
    text: str


class ConversationImport(BaseModel):
    id: str
    messages: list[ImportedMessage]


@app.post("/conversations/import")
async def conversation_import(body: ConversationImport, auth_token: str = Cookie(None)):
    """Adopt the anonymous conversation a browser kept locally, so it
    continues seamlessly after sign-in."""
    user = auth.sessions.get(auth_token)
    if user is None:
        raise HTTPException(status_code=401, detail="Oturum bulunamadı")
    owner = storage.conversation_owner(body.id)
    if owner == user["email"]:
        return {"ok": True}  # already adopted (e.g. repeated sign-in)
    if owner:
        raise HTTPException(status_code=403, detail="Bu konuşma size ait değil")
    if not (0 < len(body.messages) <= 200) or any(len(m.text) > 20000 for m in body.messages):
        raise HTTPException(status_code=422, detail="Geçersiz konuşma içeriği")
    storage.import_conversation(body.id, user["email"], [
        {"role": "assistant" if m.role == "bot" else "user", "content": m.text}
        for m in body.messages
    ])
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
        raise HTTPException(status_code=401, detail="Oturum bulunamadı")
    _authorize_conversation(conversation_id, user)
    storage.delete_conversation(conversation_id, user["email"])
    with _sessions_lock:
        _sessions.pop(("openrouter", conversation_id), None)
    return {"ok": True}


@app.get("/chat", response_model=ChatResponse)
async def chat(request: Request, query: str, session_id: str = None, model_name: str = None,
               auth_token: str = Cookie(None)):
    session_id = session_id or uuid.uuid4().hex
    user = auth.sessions.get(auth_token)
    _enforce_rate_limit(request, user)
    _authorize_conversation(session_id, user)
    model = model_name or settings.OPENROUTER_MODEL
    response = get_conversation(session_id).respond(
        query, model, education_type=user["education_type"] if user else None)
    _register_refusal(request, user, response)
    if user and response not in REFUSALS.values():  # refusals aren't part of the transcript
        storage.record_exchange(session_id, user["email"], query, response)
    return ChatResponse(query=query, response=response, model=model, session_id=session_id)


@app.get("/chat/stream")
async def chat_stream(request: Request, query: str, session_id: str = None,
                      model_name: str = None, auth_token: str = Cookie(None)):
    session_id = session_id or uuid.uuid4().hex
    user = auth.sessions.get(auth_token)
    _enforce_rate_limit(request, user)
    _authorize_conversation(session_id, user)
    conversation = get_conversation(session_id)
    stream = conversation.respond_stream(
        query, model_name or settings.OPENROUTER_MODEL,
        education_type=user["education_type"] if user else None)

    def recorded():
        tokens = []
        for token in stream:
            tokens.append(token)
            yield token
        answer = "".join(tokens)
        _register_refusal(request, user, answer)
        # persist only after the full answer arrived; refusals aren't
        # part of the transcript
        if user and answer not in REFUSALS.values():
            storage.record_exchange(session_id, user["email"], query, answer)

    return StreamingResponse(
        recorded(),
        media_type="text/plain; charset=utf-8",
        headers={"X-Session-Id": session_id},
    )


@app.get("/chat_local", response_model=ChatResponse)
async def chat_local(query: str, session_id: str = None, model_name: str = None):
    session_id = session_id or uuid.uuid4().hex
    model = model_name or settings.OLLAMA_MODEL
    response = get_conversation(session_id, backend="ollama").respond(query, model)
    return ChatResponse(query=query, response=response, model=model, session_id=session_id)


# the web UI — mounted last so the API routes above win the match first
app.mount("/", StaticFiles(directory=settings.ROOT / "web", html=True), name="web")
