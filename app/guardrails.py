"""Input gate: off-topic questions are refused before they reach
retrieval or the main model.

Uses the same LLM seam as everything else — any adapter with
chat(model, messages) -> str works, so tests use a fake.
"""

GATE_PROMPT_TEMPLATE = """Bir üniversite öğrenci işleri chatbot'una gelen soruları süzüyorsun.
Soru üniversiteyle ilgiliyse (akademik takvim, kayıt, sınavlar, yönetmelikler, dersler, harç, mezuniyet, öğrenci işleri) SADECE "evet" yaz.
Üniversiteyle ilgisi yoksa SADECE "hayır" yaz.
Emin değilsen "evet" yaz.

Soru: {query}"""

REFUSAL_MESSAGE = (
    "Üzgünüm, yalnızca üniversiteyle ilgili konularda "
    "(akademik takvim, yönetmelikler, öğrenci işleri) yardımcı olabilirim."
)


class ScopeGate:
    def __init__(self, llm, model):
        self._llm = llm
        self._model = model

    def allows(self, query: str) -> bool:
        reply = self._llm.chat(
            self._model,
            [{"role": "user", "content": GATE_PROMPT_TEMPLATE.format(query=query)}],
        )
        words = reply.strip().lower().split()
        # Only an explicit "hayır" blocks — anything unclear falls open,
        # matching the prompt's "emin değilsen evet"
        return not (words and words[0].startswith("hayır"))
