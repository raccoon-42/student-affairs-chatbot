"""One Conversation module for every backend.

Prompt assembly, retrieval context, and history trimming live here once.
Which LLM answers is decided by the adapter passed in (OpenRouter, or a
fake in tests) — history is owned by the instance, not the module.
"""
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait

from app.citations import add_missing_markers, check_citations, correct_markers
from app.links import check_links
from app.guardrails import REFUSALS
from app.llm import chat_with_usage
from app.retrieval import normalize_query
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

    <current_term>
    {term}
    </current_term>

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

    # KAMPÜS YAŞAMI — Sağlık, Kültür ve Spor (SKS) Daire Başkanlığı (SPOR TESİSLERİ, ÖĞRENCİ TOPLULUKLARI, YEMEKHANE):
    {sks_context}

    # BÖLÜM VE PROGRAMLAR:
    {programs_context}

    # ÖĞRETİM ÜYELERİ VE PERSONEL:
    {people_context}

    # DERS KATALOĞU (ders içerikleri ve önkoşullar; güncel dönem programı/şubeler burada yoktur):
    Ders kataloğu olan bölümler: {course_departments}. Diğer bölümlerin ders kataloğu elimde yok.
    {courses_context}

    # SÜREÇ REHBERLERİ (ders seçimi, yatay geçiş, çift ana dal/yan dal, başvuru, askerlik, Erasmus ve değişim programları):
    {guides_context}
    </available_reference_data>
</conversation>

GÖREV:
1. SADECE <student_question> etiketleri arasındaki soruyu yanıtla.
2. <available_reference_data> içindeki bilgileri SADECE öğrencinin sorusuna yanıt vermek için kullan.
3. Soru birden çok şekilde okunabiliyorsa en olası okumayı seç ve YANITLA; varsayımını tek
   cümleyle belirt (örn. "Bilgisayar Mühendisliği yüksek lisans dersleri için yanıtlıyorum;
   başka bölümse söyle."). Kısaltma ve yazım hatalarını en yakın anlamıyla oku ("bilgiayar" =
   bilgisayar). Netleştirme sorusunu yalnızca makul hiçbir okuma yoksa sor.
4. Referans verilerini doğrudan paylaşma, sadece soruya yanıt vermek için kullan (istisna: 6).
5. Bugünün tarihi <current_datetime> etiketinde verilmiştir. Takvim maddelerinin sonundaki
   ve rehberlerdeki tarihlerin yanındaki (DURUM: ...) etiketi bugüne göre HESAPLANMIŞTIR — geçti/gelecek kararını yalnızca bu
   etikete dayandır, kendi tarih karşılaştırmanı yapma. "geçti" etiketli tarihi gelecekmiş
   gibi sunma; "gelecekte" veya "devam ediyor" etiketli tarihe "geçti" deme. Konuşma
   geçmişinde bunlarla çelişen bir ifaden varsa etikete güven, eski ifadeni tekrarlama.
   DURUM etiketini yanıtına kopyalama; en yakın güncel tarihi öne çıkar.
   Takvim sorusu dönem belirtmiyorsa açıklama isteme: önce devam eden veya en yakın
   gelecekteki dönemin tarihini ver, diğer dönemleri istenirse ekle.
6. "X dersleri neler", "hangi dersler var" sorusunda DERS KATALOĞU'nda o seviyenin tam listesi
   varsa HER dersi kod ve adıyla madde imli listele; özetleme, alan başlıklarına gruplama.
   Yönetmelikteki ders yükü/ders sayısı ayrı bir sorudur, yalnızca o sorulursa ver. Hangi
   bölümlerin ders kataloğu olduğunu YALNIZCA "Ders kataloğu olan bölümler" satırından al;
   PROGRAMLAR'da bir doktora/lisans programının görünmesi, o bölümün ders kataloğu olduğu
   anlamına gelmez. Bölüm belirtilmemişse: o satırda tek bölüm varsa onun listesini ver ve
   hangi bölümün listesi olduğunu söyle; birden fazlaysa yalnızca o bölümleri seçenek olarak sor.
7. Sorulan şey referansta yoksa bir birime yönlendirebilirsin. URL, e-posta ya da telefon ise
   YALNIZCA referanstaki bir maddede aynen geçiyorsa yazılır ve o cümleye maddenin işareti
   konur (örn. "... [Erasmus+ Giden Öğrenci](https://...) sayfasına bak [5]"); referansta
   geçmiyorsa adres yazma, yalnızca birimin adını söyle.
8. Birden çok madde ya da adım içeren cevaplarda her maddeyi kalın kısa bir başlıkla başlat
   (örn. "- **Gerekli belgeler:** ..."). Öğrencinin yapacağı işi somut söyle: hangi sayfaya
   gideceği, hangi bağlantıya tıklayacağı, hangi dosyayı indireceği, neyi seçeceği. Bir
   sayfaya ya da dosyaya yönlendirirken bağlantıyı adıyla ver: [Sayfanın adı](URL) [n]
   (URL yalnızca kural 7'deki gibi). Öğrencinin yazacağı ya da seçeceği tam değerleri
   `böyle` yaz. Tek bilgilik kısa cevapları kısa tut, başlık ekleme.
9. Referanstaki her madde kendi numarasıyla başlar ve biter ([3] ... [3]). Bir bilgiyi hangi
   maddeden aldıysan, o bilginin geçtiği cümlenin ya da liste satırının sonuna O MADDENİN
   numarasını koy (örn. "… bilginin geçtiği cümle [3]."); birden fazla maddeyi art
   arda yaz: [1][4]. Bilgi [36] numaralı maddedense işaret [36]'dır: numaraları makale
   dipnotu gibi 1'den yeniden sıralama, kendi numaranı verme, cevabın sonuna kaynak listesi
   ya da yalnızca işaretlerden oluşan bir satır ekleme. Referansta olmayan bir numara
   kullanma; kaynağı olmayan cümleye işaret koyma.
"""

MONTHS_TR = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
             "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]
WEEKDAYS_TR = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]


def _now_tr():
    # the ISO form is for date comparisons (rule 5): models misjudge
    # "13 Temmuz 2026 geçti mi?" in prose but compare ISO dates reliably
    from datetime import datetime
    now = datetime.now()
    return (f"{now.day} {MONTHS_TR[now.month - 1]} {now.year} "
            f"{WEEKDAYS_TR[now.weekday()]}, saat {now:%H:%M} (ISO: {now:%Y-%m-%d})")

REWRITE_PROMPT = """Görevin bir sohbetteki son öğrenci mesajını, sohbet geçmişi olmadan da anlaşılacak bağımsız bir soruya dönüştürmek.
- Soruyu CEVAPLAMA, sadece yeniden yaz.
- Açılımını bilmediğin kısaltmaları (bölüm kodları gibi: "id", "eee") yazıldığı gibi soruda bırak; tahmin etme, silme. Bu konuda açıklama yazma, yine yalnızca soruyu döndür.
- Yazım hatalarını ve öğrenci kısaltmalarını düzelt (örn. "bilgiayarda" → bilgisayar mühendisliğinde, "yl" → yüksek lisans); soruya yeni bilgi ekleme, anlamını değiştirme.
- Akademik takvim sorularını takvimin terimleriyle yaz: "okul kapanıyor" / "dönem bitiyor" → yarıyıl derslerinin ve yarıyıl sonu sınavlarının bitişi, "okul açılıyor" → yarıyıl derslerinin başlangıcı, "dönem" → yarıyıl, "final" → yarıyıl sonu sınavları.
- Mesaj zaten bağımsız ve düzgünse olduğu gibi döndür.
- Sadece yeniden yazılmış soruyu döndür, açıklama ekleme.

Sohbet geçmişi:
{history}

Son mesaj: {query}"""

TITLE_PROMPT = """Aşağıdaki sohbet için en fazla dört kelimelik kısa bir başlık yaz.
- Öğrencinin yazdığı dilde yaz.
- Sadece başlığı döndür; tırnak, nokta veya açıklama ekleme.

Öğrenci: {query}
Bot: {answer}"""

IMAGE_QUERY_PROMPT = """Bir üniversite öğrencisi aşağıdaki görseli gönderdi.
{typed}Görseldeki metni ve niyeti kullanarak öğrencinin sormak istediği soruyu tek bir bağımsız Türkçe cümle olarak yaz.
- Soruyu CEVAPLAMA, sadece yaz.
- Görselde soru yoksa, görselin neyle ilgili olduğunu kısa bir arama sorgusu olarak yaz.
- Sadece soruyu/sorguyu döndür, açıklama ekleme."""


# the university the student named must survive the rewrite: lowercase
# "iyte" came back as "İTÜ", "İYTE'de" once as "İZTE'de"
_IYTE = re.compile(r"\b[iıIİ]yte\b|izmir\s+yüksek\s+teknoloji", re.IGNORECASE)


def _words(text):
    """The words alone: Turkish lowercase ("İ" -> "i", "I" -> "ı"), no
    punctuation."""
    return re.findall(r"\w+", text.replace("İ", "i").replace("I", "ı").lower())


class QueryRewriter:
    """Turns messages into clean standalone queries, so retrieval and the
    scope gate see the full intent ("lisans için kaç kredi lazım?") instead
    of a bare reply ("lisans icin soruyorum") or shorthand with typos."""

    def __init__(self, llm, model, image_model=None):
        self._llm = llm
        self._model = model
        self.image_model = image_model or model  # reading images is the harder job
        self.last_usage = None  # tokens/cost of the latest rewrite/image call

    def rewrite(self, query: str, history: list) -> str:
        # first messages too: typos and shorthand ("yl dersleri ne bilgiayarda")
        # misled the main model as much as bare follow-ups did
        lines = "\n".join(
            f"{'Öğrenci' if m['role'] == 'user' else 'Bot'}: {m['content']}" for m in history
        ) or "(yok)"
        self.last_usage = None
        try:
            rewritten, self.last_usage = chat_with_usage(self._llm, self._model, [{
                "role": "user",
                "content": REWRITE_PROMPT.format(history=lines, query=query),
            }])
            rewritten = rewritten.strip()
            # on an unknown abbreviation the model sometimes answers with its
            # reasoning ("bu kısaltmayı tahmin etmek riskli...") instead of
            # one question; none of 330 real rewrites had a line break
            if "\n" in rewritten:
                print(f"[rewrite] not a single question, using raw query: {rewritten[:80]!r}",
                      file=sys.stderr)
                return query
            if _IYTE.search(query) and not _IYTE.search(rewritten):
                print(f"[rewrite] dropped İYTE, using raw query: {rewritten[:80]!r}", file=sys.stderr)
                return query
            # capitals and punctuation alone don't make a new question (53 of
            # 197 changed eval rewrites); the prompt can't stop Haiku doing it
            if _words(rewritten) == _words(query):
                return query
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
        self.last_usage = None
        try:
            extracted, self.last_usage = chat_with_usage(self._llm, self.image_model, [{
                "role": "user",
                "content": [
                    {"type": "text", "text": IMAGE_QUERY_PROMPT.format(typed=typed)},
                    {"type": "image_url", "image_url": {"url": image}},
                ],
            }])
            return extracted.strip() or query
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
# PDFs repeat 34 questions verbatim, so retrieval filters to one audience
# (plus hazırlık, which every profile gets: see Retriever.retrieve_faq).
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
        self.last_reference = []  # numbered chunk texts sent to the model last turn
        self.last_answer = None  # the latest answer after the citation check
        self._chunk_texts = []  # what marker [n] points to: _chunk_texts[n-1]
        self.last_debug = []  # CLI-style log lines for the latest turn
        self.last_usage = None  # {"prompt_tokens", "completion_tokens"} of the latest main call
        self.turn_usage = []  # every LLM call of the latest turn: {kind, model, tokens, cost}
        self._early = None  # gate + retrieval started on the student's words, see _start

    def _log(self, line):
        print(line, file=sys.stderr)
        self.last_debug.append(line)

    def _log_row(self, label, value, detail=""):
        """Aligned three-column log line: stage, timing/count, detail —
        reads as a table in the dev card (monospace) and in stderr."""
        self._log(f"{label:<10}{value:>8}  {detail}".rstrip())

    def _track(self, kind, model, usage):
        """One LLM call of the current turn, for per-kind cost accounting."""
        if usage:
            self.turn_usage.append({"kind": kind, "model": model, **usage})

    def respond(self, query: str, model: str = None, education_type: str = None) -> str:
        self.last_answer = None
        verdict = self._prepare(query, education_type)
        if verdict != "ok":
            return REFUSALS[verdict]

        start = time.perf_counter()
        answer, self.last_usage = chat_with_usage(
            self._llm, model or self._model, self._messages)
        self._track("chat", model or self._model, self.last_usage)
        self._log_row("llm", f"{time.perf_counter() - start:.2f}s")
        answer = self._check_citations(answer)
        self._add_assistant_message(answer)
        return answer

    def respond_stream(self, query: str, model: str = None, education_type: str = None,
                       image: str = None):
        """Same as respond(), but yields the answer token by token. The
        citation check runs once the answer is complete, so the tokens
        already sent can differ from last_answer; the UI re-renders with it."""
        self.last_answer = None
        query, search_query = self._resolve_search_query(query, image, education_type)
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
        stream = self._llm.chat_stream(model or self._model, self._messages)
        while True:  # manual next(): the generator's return value is the usage
            try:
                token = next(stream)
            except StopIteration as done:
                self.last_usage = done.value
                self._track("chat", model or self._model, self.last_usage)
                break
            if first_token_at is None:
                first_token_at = time.perf_counter() - start
            tokens.append(token)
            yield token
        self._log_row("llm", f"{time.perf_counter() - start:.2f}s",
                      f"first token {first_token_at:.2f}s")
        if self.last_usage:
            cost = self.last_usage.get("cost")
            cached = self.last_usage.get("cached_tokens") or 0
            written = self.last_usage.get("cache_write_tokens") or 0
            self._log_row("usage", "",
                          f"prompt {self.last_usage['prompt_tokens']} + "
                          f"completion {self.last_usage['completion_tokens']} token"
                          + (f" · cache read {cached}" if cached else "")
                          + (f" · cache write {written}" if written else "")
                          + (f" · ${cost:.4f}" if cost is not None else ""))
        self._add_assistant_message(self._check_citations("".join(tokens)))

    def _prepare(self, query, education_type=None, image=None) -> str:
        """Rewrite, then gate + retrieve; see the two halves below."""
        query, search_query = self._resolve_search_query(query, image, education_type)
        stages = self._gate_and_build(query, search_query, education_type, image)
        while True:  # non-streaming path: drain the stage events, keep the verdict
            try:
                next(stages)
            except StopIteration as done:
                return done.value

    def _resolve_search_query(self, query, image=None, education_type=None):
        """First pipeline stage: turn the raw message into a search query.

        Every message gets rewritten into a clean standalone question for
        the gate and retrieval; the model sees the original message with
        the rewrite next to it.
        Returns (query, search_query) — query changes only for image-only
        messages, where the extracted question stands in as the student's
        text for the prompt and the history.

        The gate and retrieval start on the student's words while the
        rewrite runs: most rewrites give them back (unchanged or only
        recapitalized), and then nothing waits for the rewrite twice."""
        self._early = None
        self.last_debug = []
        self.last_usage = None  # refusals never reach the main call
        self.turn_usage = []
        search_query = query
        if self._rewriter is not None and image is not None:
            extract_start = time.perf_counter()
            search_query = self._rewriter.image_query(query, image)
            self._track("image", self._rewriter.image_model, self._rewriter.last_usage)
            self._log_row("image", f"{time.perf_counter() - extract_start:.2f}s",
                          f"search query: {search_query!r}" if search_query != query else "")
            if not query:
                query = search_query
        elif self._rewriter is not None:
            self._early = self._start(normalize_query(query), query, education_type)
            rewrite_start = time.perf_counter()
            # the rewriter guesses codes it doesn't know ("mbg" became
            # "Makine Bilgisayar Mühendisliği"), so it gets them named
            search_query = self._rewriter.rewrite(normalize_query(query), self._messages[1:])
            self._track("rewrite", self._rewriter._model, self._rewriter.last_usage)
            self._log_row("rewrite", f"{time.perf_counter() - rewrite_start:.2f}s",
                          f"{query!r}\n{'':<20}-> {search_query!r}"
                          if search_query != query else "unchanged")
        # expanded here, not only inside retrieval, so the gate and the
        # model's "(anlaşılan soru: ...)" see "yüksek lisans" too
        return query, normalize_query(search_query)

    def _gate_and_build(self, query, search_query, education_type=None, image=None):
        """Second pipeline stage: gate and retrieval run concurrently —
        retrieval is speculative, its result is discarded if the gate
        blocks. A generator: yields STAGE_SEARCHING once the gate verdict
        is in (only retrieval still running), and returns "ok" when the
        query goes through, otherwise the gate's refusal verdict.

        An image (data URL) rides along for this turn only: the gate,
        rewriter and retrieval see just the text, and history keeps the
        bare question, so the image never bloats later requests."""
        early, self._early = self._early, None
        if early is not None and early["text"] != search_query:
            # the rewrite changed the question: what ran on the student's
            # words is set aside, after it finishes (its gate call is
            # counted, and two calls never share the gate's last_usage)
            wait([future for future in (early["verdict"], early["results"]) if future])
            if early["verdict"] is not None and early["verdict"].exception() is None:
                self._track("gate", self._gate._model, self._gate.last_usage)
            early = None
        started = early or self._start(search_query, query, education_type)
        start = started["start"]
        if started["verdict"] is not None:
            answer = started["verdict"].result()
            self._track("gate", self._gate._model, self._gate.last_usage)
            self._log_row("gate", f"{time.perf_counter() - start:.2f}s",
                          f"verdict {answer}")
            if answer != "ok":
                return answer
        yield STAGE_SEARCHING
        results = started["results"].result()
        timings = getattr(self._retriever, "last_timings", None)
        if timings:
            self._log_row("retrieval", f"{timings['embed'] + timings['search']:.2f}s",
                          f"embed {timings['embed']:.2f}s + search {timings['search']:.2f}s, "
                          f"alongside the gate")
        self._log_row("pipeline", f"{time.perf_counter() - start:.2f}s",
                      "gate + retrieval, started with the rewrite" if early
                      else "gate + retrieval combined")
        routing = getattr(self._retriever, "last_routing", None)
        if routing and routing["scores"]:
            # every turn, so "nothing left out" reads as a decision, not a gap
            scores = routing["scores"]
            best, lowest = max(scores, key=scores.get), min(scores, key=scores.get)
            detail = f"best {best} {scores[best]:.2f}"
            if settings.ROUTE_GAP is not None:
                detail += f" · cutoff {scores[best] - settings.ROUTE_GAP:.2f}"
            detail += (" · left out " + " · ".join(f"{name} {scores[name]:.2f}"
                                                  for name in routing["dropped"])
                       if routing["dropped"] else
                       f" · nothing left out (lowest {lowest} {scores[lowest]:.2f})")
            self._log_row("routing", f"-{len(routing['dropped'])}", detail)
        corpora = ("calendar", "regulations", "faq", "forms", "sks", "programs", "people",
                   "courses", "guides")
        counts = [f"{corpus} {len(results.get(corpus, []))}" for corpus in corpora]
        self._log_row("chunks", str(sum(len(results.get(c, [])) for c in corpora)),
                      " · ".join(counts[:5]) + f"\n{'':<20}" + " · ".join(counts[5:]))

        # every chunk gets a [n] marker; the model cites them inline and
        # last_sources[n-1] is what marker [n] points to
        numbered = {corpus: [] for corpus in corpora}
        self.last_sources = []
        self._chunk_texts = []
        for corpus in corpora:
            for result in results.get(corpus, []):
                self.last_sources.append(self._source_entry(corpus, result))
                self._chunk_texts.append(result["text"])
                # the marker ends the chunk too: a fact on a chunk's last line
                # (an office number) sat right before the NEXT chunk's marker,
                # and the model cited that one (4 of 10 office answers)
                n = len(self.last_sources)
                numbered[corpus].append(f"[{n}] {result['text']} [{n}]")
        self.last_reference = [entry for corpus in corpora for entry in numbered[corpus]]

        # a follow-up ("akts kac") only makes sense with the history; without
        # the resolved question next to it, the model asked what was meant
        question = query if search_query == query else f"{query}\n    (anlaşılan soru: {search_query})"
        context = CONTEXT_TEMPLATE.format(
            query=question,
            now=_now_tr(),
            term=results.get("term") or "bilinmiyor",
            profile=EDUCATION_LABELS.get(education_type, "bilinmiyor"),
            calendar_context="\n".join(numbered["calendar"]),
            regulations_context="\n".join(numbered["regulations"]),
            faq_context="\n\n".join(numbered["faq"]),
            forms_context="\n\n".join(numbered["forms"]),
            sks_context="\n\n".join(numbered["sks"]),
            programs_context="\n\n".join(numbered["programs"]),
            people_context="\n\n".join(numbered["people"]),
            courses_context="\n\n".join(numbered["courses"]),
            course_departments=", ".join(results.get("course_departments") or []) or "yok",
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

    def _start(self, search_query, query, education_type):
        """Gate and retrieval for search_query, running in the background.
        Retrieval is speculative: its result is discarded if the gate
        blocks."""
        pool = ThreadPoolExecutor(max_workers=2)
        started = {
            "text": search_query, "start": time.perf_counter(),
            "verdict": pool.submit(self._gate.verdict, search_query) if self._gate else None,
            # the student's own words search too: the rewrite fixes typos and
            # follow-ups but sometimes misreads ("mbg" expanded wrong)
            "results": pool.submit(self._retriever.retrieve_all, search_query,
                                   FAQ_AUDIENCES.get(education_type), also=[query]),
        }
        pool.shutdown(wait=False)  # a refusal doesn't wait for retrieval
        return started

    @staticmethod
    def _source_entry(corpus, result):
        """One 'view sources' entry. Labels are the stored chunk texts, so
        they're readable as-is; every chunk carries its own source_url from
        the scrape manifest (synthetic list/area chunks have none)."""
        corpus_names = {"calendar": "Akademik takvim", "regulations": "Yönetmelik",
                        "faq": "SSS", "forms": "Form", "sks": "SKS", "programs": "Bölüm",
                        "people": "Kişi", "courses": "Ders", "guides": "Rehber"}
        metadata = result.get("metadata") or {}
        # chips labeled by hostname all look alike — the document's own
        # title (indexed with each mevzuat chunk) tells sources apart
        source = {"type": corpus_names[corpus],
                  "title": metadata.get("document_title") or corpus_names[corpus],
                  "label": result["text"][:120]}
        if metadata.get("source_url"):
            source["url"] = metadata["source_url"]
        return source

    def _check_citations(self, answer):
        """The answer after the deterministic checks: citation markers its
        chunks don't back, and links no chunk contains."""
        answer, removed = check_citations(answer, self._chunk_texts)
        if removed:
            self._log_row("citations", f"-{len(removed)}",
                          "removed " + " ".join(f"[{n}]" for n in sorted(set(removed)))
                          + ": no word shared with the cited chunk, or a marker-only last line")
        answer, moved = correct_markers(answer, self._chunk_texts)
        if moved:
            self._log_row("markers", f"~{len(moved)}",
                          "moved " + " ".join(f"[{a}]→[{b}]" for a, b in moved)
                          + ": the number is in the other chunk only")
        answer, added = add_missing_markers(answer, self._chunk_texts)
        if added:
            self._log_row("markers", f"+{len(added)}",
                          "added " + " ".join(f"[{n}]" for n in added)
                          + ": a bare fact that a cited chunk holds")
        answer, links = check_links(answer, self._chunk_texts)
        if links:
            self._log_row("links", f"-{len(links)}",
                          "removed " + " ".join(links) + ": in no chunk of this turn")
        self.last_answer = answer
        return answer

    def _add_assistant_message(self, answer):
        # the reference data was for this turn only — history keeps the bare
        # question, fresh context is retrieved every turn anyway
        self._messages[-1] = {"role": "user", "content": self._pending_query}
        self._messages.append({"role": "assistant", "content": answer})

        # system prompt + last N user/assistant pairs
        limit = 1 + 2 * self._max_exchanges
        if len(self._messages) > limit:
            self._messages = [self._messages[0]] + self._messages[-(limit - 1):]

    def suggest_title(self, model):
        """(title, usage): a short sidebar title for the first exchange,
        written in the student's language; (None, None) when there is no
        exchange to summarize. The answer is clipped — four title words
        don't need all of it."""
        exchange = [m for m in self._messages if m["role"] != "system"]
        if len(exchange) < 2:
            return None, None
        title, usage = chat_with_usage(self._llm, model, [
            {"role": "user", "content": TITLE_PROMPT.format(
                query=exchange[0]["content"], answer=exchange[1]["content"][:500])}])
        return title.strip().strip('"\'') or None, usage

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
    from app.llm import OpenRouterLLM
    from app.retrieval import default_retriever

    parser = argparse.ArgumentParser(description="Chat with the bot from the terminal")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    llm, model, gate_model = OpenRouterLLM(), args.model or settings.OPENROUTER_MODEL, settings.GUARD_MODEL
    conversation = Conversation(llm, default_retriever(), model,
                                gate=ScopeGate(llm, gate_model),
                                rewriter=QueryRewriter(llm, gate_model, settings.IMAGE_MODEL))

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
