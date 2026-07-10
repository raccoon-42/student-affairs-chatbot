"""One Conversation module for every backend.

Prompt assembly, retrieval context, and history trimming live here once.
Which LLM answers is decided by the adapter passed in (OpenRouter, Ollama,
or a fake in tests) — history is owned by the instance, not the module.
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from app.guardrails import REFUSALS
from config import settings

# In-band stage markers for streaming consumers. Control characters so
# they can never collide with answer text; the web UI switches the cursor
# animation on them, every other consumer must filter them out.
# \x02: the query rewrite is done, the gate is checking (retrieval runs
#       alongside, but the gate verdict is what resolves first).
# \x03: the gate passed, only retrieval is still running.
# \x01: retrieval is done, the model is about to write.
STAGE_GATING = "\x02"
STAGE_SEARCHING = "\x03"
STAGE_WRITING = "\x01"
STAGE_MARKERS = (STAGE_GATING, STAGE_SEARCHING, STAGE_WRITING)

CONTEXT_TEMPLATE = """
<conversation>
    <current_datetime>
    {now}
    </current_datetime>

    <student_question>
    {query}
    </student_question>

    <student_profile>
    {profile}
    </student_profile>

    <available_reference_data>
    # AKADEMİK TAKVİM BİLGİLERİ:
    {calendar_context}

    # YÖNETMELİK BİLGİLERİ:
    {regulations_context}

    # SIKÇA SORULAN SORULAR:
    {faq_context}

    # FORM VE DİLEKÇELER (başlık + kullanım amacı; bağlantı için kaynak işareti yeterli):
    {forms_context}

    # KAMPÜS YAŞAMI (SPOR TESİSLERİ, ÖĞRENCİ TOPLULUKLARI, YEMEKHANE):
    {sks_context}

    # BÖLÜM VE PROGRAMLAR:
    {programs_context}

    # ÖĞRETİM ÜYELERİ VE PERSONEL:
    {people_context}

    # DERS KATALOĞU (ders içerikleri ve önkoşullar; güncel dönem programı/şubeler burada yoktur):
    {courses_context}

    # ÖĞRENCİ İŞLERİ SÜREÇ REHBERLERİ (ders seçimi, yatay geçiş, çift ana dal/yan dal, başvuru, askerlik):
    {guides_context}
    </available_reference_data>
</conversation>

GÖREV:
1. SADECE <student_question> etiketleri arasındaki soruyu yanıtla.
2. <available_reference_data> içindeki bilgileri SADECE öğrencinin sorusuna yanıt vermek için kullan.
3. Eğer öğrencinin sorusu belirsizse veya eksik bilgi varsa, açıklama iste.
4. Referans verilerini doğrudan paylaşma, sadece soruya yanıt vermek için kullan.
5. Bugünün tarihi <current_datetime> etiketinde verilmiştir. Referans verilerdeki bu tarihten
   önce kalan tarihleri gelecekmiş gibi sunma; geçmişse bunu açıkça belirt ve varsa bir
   sonraki/güncel tarihi öne çıkar. Referans verilerde güncel bilgi yoksa bunu söyle.
"""

MONTHS_TR = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
             "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]
WEEKDAYS_TR = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]


def _now_tr():
    from datetime import datetime
    now = datetime.now()
    return (f"{now.day} {MONTHS_TR[now.month - 1]} {now.year} "
            f"{WEEKDAYS_TR[now.weekday()]}, saat {now:%H:%M}")

REWRITE_PROMPT = """Görevin bir sohbetteki son öğrenci mesajını, sohbet geçmişi olmadan da anlaşılacak bağımsız bir soruya dönüştürmek.
- Soruyu CEVAPLAMA, sadece yeniden yaz.
- Mesaj zaten bağımsız anlaşılıyorsa olduğu gibi döndür.
- Sadece yeniden yazılmış soruyu döndür, açıklama ekleme.

Sohbet geçmişi:
{history}

Son mesaj: {query}"""

IMAGE_QUERY_PROMPT = """Bir üniversite öğrencisi aşağıdaki görseli gönderdi.
{typed}Görseldeki metni ve niyeti kullanarak öğrencinin sormak istediği soruyu tek bir bağımsız Türkçe cümle olarak yaz.
- Soruyu CEVAPLAMA, sadece yaz.
- Görselde soru yoksa, görselin neyle ilgili olduğunu kısa bir arama sorgusu olarak yaz.
- Sadece soruyu/sorguyu döndür, açıklama ekleme."""


class QueryRewriter:
    """Turns follow-up messages into standalone queries, so retrieval and
    the scope gate see the full intent ("lisans için kaç kredi lazım?")
    instead of the bare reply ("lisans icin soruyorum")."""

    def __init__(self, llm, model):
        self._llm = llm
        self._model = model

    def rewrite(self, query: str, history: list) -> str:
        if not history:
            return query
        lines = "\n".join(
            f"{'Öğrenci' if m['role'] == 'user' else 'Bot'}: {m['content']}" for m in history
        )
        try:
            rewritten = self._llm.chat(self._model, [{
                "role": "user",
                "content": REWRITE_PROMPT.format(history=lines, query=query),
            }]).strip()
            return rewritten or query
        except Exception as error:
            # retrieval on the raw query beats no answer at all
            print(f"[rewrite] failed, using raw query: {error}", file=sys.stderr)
            return query

    def image_query(self, query: str, image: str) -> str:
        """Turn an attached image (plus whatever the student typed) into a
        standalone search query, so the gate and retrieval work on what the
        image actually asks — the vector search can't see pixels."""
        typed = f"Öğrencinin yazdığı mesaj: {query}\n" if query else ""
        try:
            extracted = self._llm.chat(self._model, [{
                "role": "user",
                "content": [
                    {"type": "text", "text": IMAGE_QUERY_PROMPT.format(typed=typed)},
                    {"type": "image_url", "image_url": {"url": image}},
                ],
            }]).strip()
            return extracted or query
        except Exception as error:
            print(f"[image] query extraction failed, using text only: {error}", file=sys.stderr)
            return query


EDUCATION_LABELS = {
    "aday": "İYTE'ye gelmeyi düşünen aday öğrenci (henüz kayıtlı değil)",
    "lisans": "lisans öğrencisi",
    "yukseklisans": "yüksek lisans öğrencisi",
    "doktora": "doktora öğrencisi",
}

# which FAQ corpus copy fits the profile: the lisans and lisansüstü FAQ
# PDFs repeat 34 questions verbatim, so retrieval filters to one audience.
# Prospective students get the lisans copy; no profile means no filter
# (the retriever dedupes by question text instead).
FAQ_AUDIENCES = {
    "aday": "lisans",
    "lisans": "lisans",
    "yukseklisans": "lisansustu",
    "doktora": "lisansustu",
}


class Conversation:
    def __init__(self, llm, retriever, model, max_exchanges=5, gate=None, rewriter=None):
        self._llm = llm
        self._retriever = retriever
        self._model = model
        self._max_exchanges = max_exchanges
        self._gate = gate
        self._rewriter = rewriter
        self._messages = []
        self.last_sources = []  # what the latest answer was grounded on
        self.last_debug = []  # CLI-style log lines for the latest turn

    def _log(self, line):
        print(line, file=sys.stderr)
        self.last_debug.append(line)

    def respond(self, query: str, model: str = None, education_type: str = None) -> str:
        verdict = self._prepare(query, education_type)
        if verdict != "ok":
            return REFUSALS[verdict]

        start = time.perf_counter()
        answer = self._llm.chat(model or self._model, self._messages)
        self._log(f"[timing] llm {time.perf_counter() - start:.2f}s")
        self._add_assistant_message(answer)
        return answer

    def respond_stream(self, query: str, model: str = None, education_type: str = None,
                       image: str = None):
        """Same as respond(), but yields the answer token by token."""
        query, search_query = self._resolve_search_query(query, image)
        yield STAGE_GATING
        stages = self._gate_and_build(query, search_query, education_type, image)
        while True:
            try:
                yield next(stages)  # STAGE_SEARCHING once the gate passes
            except StopIteration as done:
                verdict = done.value
                break
        if verdict != "ok":
            yield REFUSALS[verdict]
            return

        yield STAGE_WRITING
        start = time.perf_counter()
        first_token_at = None
        tokens = []
        for token in self._llm.chat_stream(model or self._model, self._messages):
            if first_token_at is None:
                first_token_at = time.perf_counter() - start
            tokens.append(token)
            yield token
        self._log(f"[timing] llm first token {first_token_at:.2f}s, "
                  f"total {time.perf_counter() - start:.2f}s")
        self._add_assistant_message("".join(tokens))

    def _prepare(self, query, education_type=None, image=None) -> str:
        """Rewrite, then gate + retrieve; see the two halves below."""
        query, search_query = self._resolve_search_query(query, image)
        stages = self._gate_and_build(query, search_query, education_type, image)
        while True:  # non-streaming path: drain the stage events, keep the verdict
            try:
                next(stages)
            except StopIteration as done:
                return done.value

    def _resolve_search_query(self, query, image=None):
        """First pipeline stage: turn the raw message into a search query.

        Follow-ups get rewritten into standalone questions for the gate
        and retrieval; the model itself still sees the original message.
        Returns (query, search_query) — query changes only for image-only
        messages, where the extracted question stands in as the student's
        text for the prompt and the history."""
        self.last_debug = []
        search_query = query
        if self._rewriter is not None and image is not None:
            extract_start = time.perf_counter()
            search_query = self._rewriter.image_query(query, image)
            self._log(f"[timing] image query {time.perf_counter() - extract_start:.2f}s")
            if search_query != query:
                self._log(f"[image] search query: {search_query!r}")
            if not query:
                query = search_query
        elif self._rewriter is not None and len(self._messages) > 1:
            rewrite_start = time.perf_counter()
            search_query = self._rewriter.rewrite(query, self._messages[1:])
            self._log(f"[timing] rewrite {time.perf_counter() - rewrite_start:.2f}s")
            if search_query != query:
                self._log(f"[rewrite] {query!r} -> {search_query!r}")
        return query, search_query

    def _gate_and_build(self, query, search_query, education_type=None, image=None):
        """Second pipeline stage: gate and retrieval run concurrently —
        retrieval is speculative, its result is discarded if the gate
        blocks. A generator: yields STAGE_SEARCHING once the gate verdict
        is in (only retrieval still running), and returns "ok" when the
        query goes through, otherwise the gate's refusal verdict.

        An image (data URL) rides along for this turn only: the gate,
        rewriter and retrieval see just the text, and history keeps the
        bare question, so the image never bloats later requests."""
        start = time.perf_counter()
        audience = FAQ_AUDIENCES.get(education_type)
        if self._gate is None:
            yield STAGE_SEARCHING
            results = self._retriever.retrieve_all(search_query, audience=audience)
        else:
            with ThreadPoolExecutor(max_workers=2) as pool:
                verdict = pool.submit(self._gate.verdict, search_query)
                speculative = pool.submit(self._retriever.retrieve_all, search_query,
                                          audience)
                answer = verdict.result()
                self._log(f"[timing] gate {time.perf_counter() - start:.2f}s "
                          f"(verdict: {answer})")
                if answer != "ok":
                    return answer
                yield STAGE_SEARCHING
                results = speculative.result()
        timings = getattr(self._retriever, "last_timings", None)
        if timings:
            self._log(f"[timing] retrieval {timings['embed'] + timings['search']:.2f}s "
                      f"(embed {timings['embed']:.2f}s | search {timings['search']:.2f}s, "
                      f"alongside the gate)")
        self._log(f"[timing] gate + retrieval combined {time.perf_counter() - start:.2f}s")
        corpora = ("calendar", "regulations", "faq", "forms", "sks", "programs", "people",
                   "courses", "guides")
        self._log("[retrieval] " + ", ".join(
            f"{corpus}: {len(results.get(corpus, []))}" for corpus in corpora))

        # every chunk gets a [n] marker; the model cites them inline and
        # last_sources[n-1] is what marker [n] points to
        numbered = {corpus: [] for corpus in corpora}
        self.last_sources = []
        for corpus in corpora:
            for result in results.get(corpus, []):
                self.last_sources.append(self._source_entry(corpus, result))
                numbered[corpus].append(f"[{len(self.last_sources)}] {result['text']}")

        context = CONTEXT_TEMPLATE.format(
            query=query,
            now=_now_tr(),
            profile=EDUCATION_LABELS.get(education_type, "bilinmiyor"),
            calendar_context="\n".join(numbered["calendar"]),
            regulations_context="\n".join(numbered["regulations"]),
            faq_context="\n\n".join(numbered["faq"]),
            forms_context="\n\n".join(numbered["forms"]),
            sks_context="\n\n".join(numbered["sks"]),
            programs_context="\n\n".join(numbered["programs"]),
            people_context="\n\n".join(numbered["people"]),
            courses_context="\n\n".join(numbered["courses"]),
            guides_context="\n\n".join(numbered["guides"]),
        )

        if not self._messages:
            self._messages.append({"role": "system", "content": settings.load_system_prompt()})
        text = f"Öğrenci: {query}\n\n{context}"
        content = text if image is None else [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": image}},
        ]
        self._messages.append({"role": "user", "content": content})
        self._pending_query = query
        return "ok"

    @staticmethod
    def _source_entry(corpus, result):
        """One 'view sources' entry. Labels are the stored chunk texts, so
        they're readable as-is; FAQ entries carry their own page URL,
        calendar/regulations fall back to the configured corpus page."""
        corpus_names = {"calendar": "Akademik takvim", "regulations": "Yönetmelik",
                        "faq": "SSS", "forms": "Form", "sks": "SKS", "programs": "Bölüm",
                        "people": "Kişi", "courses": "Ders", "guides": "Rehber"}
        corpus_urls = {"calendar": settings.CALENDAR_SOURCE_URL,
                       "regulations": settings.REGULATIONS_SOURCE_URL, "faq": "",
                       "forms": settings.FORMS_SOURCE_URL,
                       "sks": settings.SKS_SOURCE_URL,
                       "programs": settings.PROGRAMS_SOURCE_URL,
                       "people": "", "courses": "", "guides": ""}
        metadata = result.get("metadata") or {}
        # chips labeled by hostname all look alike — the document's own
        # title (indexed with each mevzuat chunk) tells sources apart
        source = {"type": corpus_names[corpus],
                  "title": metadata.get("document_title") or corpus_names[corpus],
                  "label": result["text"][:120]}
        url = metadata.get("source_url") or corpus_urls[corpus]
        if url:
            source["url"] = url
        return source

    def _add_assistant_message(self, answer):
        # the reference data was for this turn only — history keeps the bare
        # question, fresh context is retrieved every turn anyway
        self._messages[-1] = {"role": "user", "content": self._pending_query}
        self._messages.append({"role": "assistant", "content": answer})

        # system prompt + last N user/assistant pairs
        limit = 1 + 2 * self._max_exchanges
        if len(self._messages) > limit:
            self._messages = [self._messages[0]] + self._messages[-(limit - 1):]

    def reset(self):
        self._messages = []
        return "Konuşma sıfırlandı."

    def load_history(self, stored_messages):
        """Rebuild in-memory context from persisted turns (after a server
        restart or cache eviction), keeping the usual trimming."""
        self._messages = [{"role": "system", "content": settings.load_system_prompt()}]
        for message in stored_messages:
            self._messages.append({"role": message["role"], "content": message["content"]})
        limit = 1 + 2 * self._max_exchanges
        if len(self._messages) > limit:
            self._messages = [self._messages[0]] + self._messages[-(limit - 1):]


if __name__ == "__main__":
    import argparse

    from app.guardrails import ScopeGate
    from app.llm import OpenRouterLLM, OllamaLLM
    from app.retrieval import default_retriever

    parser = argparse.ArgumentParser(description="Chat with the bot from the terminal")
    parser.add_argument("--backend", choices=["openrouter", "ollama"], default="openrouter")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    if args.backend == "openrouter":
        llm, model, gate_model = OpenRouterLLM(), args.model or settings.OPENROUTER_MODEL, settings.GUARD_MODEL
    else:
        llm, model, gate_model = OllamaLLM(), args.model or settings.OLLAMA_MODEL, args.model or settings.OLLAMA_MODEL
    conversation = Conversation(llm, default_retriever(), model,
                                gate=ScopeGate(llm, gate_model),
                                rewriter=QueryRewriter(llm, gate_model))

    print("İyteBot'a hoş geldiniz. Çıkmak için 'q', sıfırlamak için 'sıfırla'.")
    while True:
        user_input = input("\n>> ")
        if user_input.lower() in ["çıkış", "exit", "quit", "q"]:
            break
        if user_input.lower() in ["sıfırla", "reset"]:
            print(conversation.reset())
            continue
        print()
        for token in conversation.respond_stream(user_input):
            if token not in STAGE_MARKERS:
                print(token, end="", flush=True)
        print()
