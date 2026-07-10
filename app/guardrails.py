"""Input gate: off-topic questions are refused before they reach
retrieval or the main model.

Uses the same LLM seam as everything else — any adapter with
chat(model, messages) -> str works, so tests use a fake.
"""

GATE_PROMPT_TEMPLATE = """Bir üniversite öğrenci işleri chatbot'una gelen mesajları süzüyorsun.
Mesaj hakaret, küfür veya kaba/saldırgan bir dil içeriyorsa SADECE "kaba" yaz.
Mesaj üniversiteyle ilgiliyse (akademik takvim, kayıt, ders seçimi, sınavlar, yönetmelikler, dersler, bölümler ve programlar, öğretim üyeleri, harç, mezuniyet, öğrenci işleri: yatay geçiş, çift ana dal/yan dal, başvurular, öğrenci askerlik/tecil işlemleri, belgeler, kampüs yaşamı: yemekhane, spor tesisleri, öğrenci toplulukları, burslar) SADECE "evet" yaz.
Üniversiteyle ilgisi yoksa SADECE "hayır" yaz.
Emin değilsen "evet" yaz.

Mesaj: {query}"""

REFUSAL_MESSAGE = (
    "Üzgünüm, yalnızca üniversiteyle ilgili konularda "
    "(akademik takvim, yönetmelikler, öğrenci işleri) yardımcı olabilirim."
)
ABUSE_MESSAGE = "Lütfen saygılı bir dil kullan."

# verdict -> canned reply; anything the model answers with is NOT in here
REFUSALS = {"off_topic": REFUSAL_MESSAGE, "abusive": ABUSE_MESSAGE}


class ScopeGate:
    def __init__(self, llm, model):
        self._llm = llm
        self._model = model

    def verdict(self, query: str) -> str:
        """'ok', 'off_topic' or 'abusive'. Only explicit "hayır"/"kaba"
        block — anything unclear falls open, matching the prompt's
        "emin değilsen evet"."""
        reply = self._llm.chat(
            self._model,
            [{"role": "user", "content": GATE_PROMPT_TEMPLATE.format(query=query)}],
        )
        words = reply.strip().lower().split()
        first = words[0] if words else ""
        if first.startswith("kaba"):
            return "abusive"
        if first.startswith("hayır"):
            return "off_topic"
        return "ok"

    def allows(self, query: str) -> bool:
        return self.verdict(query) == "ok"
