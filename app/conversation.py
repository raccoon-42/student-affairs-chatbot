"""One Conversation module for every backend.

Prompt assembly, retrieval context, and history trimming live here once.
Which LLM answers is decided by the adapter passed in (OpenRouter, Ollama,
or a fake in tests) — history is owned by the instance, not the module.
"""
from app.guardrails import REFUSAL_MESSAGE
from config import settings

CONTEXT_TEMPLATE = """
<conversation>
    <student_question>
    {query}
    </student_question>

    <available_reference_data>
    # AKADEMİK TAKVİM BİLGİLERİ:
    {calendar_context}

    # YÖNETMELİK BİLGİLERİ:
    {regulations_context}
    </available_reference_data>
</conversation>

GÖREV:
1. SADECE <student_question> etiketleri arasındaki soruyu yanıtla.
2. <available_reference_data> içindeki bilgileri SADECE öğrencinin sorusuna yanıt vermek için kullan.
3. Eğer öğrencinin sorusu belirsizse veya eksik bilgi varsa, açıklama iste.
4. Referans verilerini doğrudan paylaşma, sadece soruya yanıt vermek için kullan.
"""


class Conversation:
    def __init__(self, llm, retriever, model, max_exchanges=5, gate=None):
        self._llm = llm
        self._retriever = retriever
        self._model = model
        self._max_exchanges = max_exchanges
        self._gate = gate
        self._messages = []

    def respond(self, query: str, model: str = None) -> str:
        if self._gate and not self._gate.allows(query):
            return REFUSAL_MESSAGE

        self._add_user_message(query)
        answer = self._llm.chat(model or self._model, self._messages)
        self._add_assistant_message(answer)
        return answer

    def respond_stream(self, query: str, model: str = None):
        """Same as respond(), but yields the answer token by token."""
        if self._gate and not self._gate.allows(query):
            yield REFUSAL_MESSAGE
            return

        self._add_user_message(query)
        tokens = []
        for token in self._llm.chat_stream(model or self._model, self._messages):
            tokens.append(token)
            yield token
        self._add_assistant_message("".join(tokens))

    def _add_user_message(self, query):
        calendar_results = self._retriever.retrieve_calendar(query)
        regulations_results = self._retriever.retrieve_regulations(query)

        context = CONTEXT_TEMPLATE.format(
            query=query,
            calendar_context="\n".join(r["text"] for r in calendar_results),
            regulations_context="\n".join(r["text"] for r in regulations_results),
        )

        if not self._messages:
            self._messages.append({"role": "system", "content": settings.load_system_prompt()})
        self._messages.append({"role": "user", "content": f"Öğrenci: {query}\n\n{context}"})

    def _add_assistant_message(self, answer):
        self._messages.append({"role": "assistant", "content": answer})

        # system prompt + last N user/assistant pairs
        limit = 1 + 2 * self._max_exchanges
        if len(self._messages) > limit:
            self._messages = [self._messages[0]] + self._messages[-(limit - 1):]

    def reset(self):
        self._messages = []
        return "Konuşma sıfırlandı."


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
    conversation = Conversation(llm, default_retriever(), model, gate=ScopeGate(llm, gate_model))

    print("Bilgi Bot'a hoş geldiniz. Çıkmak için 'q', sıfırlamak için 'sıfırla'.")
    while True:
        user_input = input("\n>> ")
        if user_input.lower() in ["çıkış", "exit", "quit", "q"]:
            break
        if user_input.lower() in ["sıfırla", "reset"]:
            print(conversation.reset())
            continue
        print()
        for token in conversation.respond_stream(user_input):
            print(token, end="", flush=True)
        print()
