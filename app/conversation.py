"""One Conversation module for every backend.

Prompt assembly, retrieval context, and history trimming live here once.
Which LLM answers is decided by the adapter passed in (OpenRouter, Ollama,
or a fake in tests) — history is owned by the instance, not the module.
"""
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
    def __init__(self, llm, retriever, model, max_exchanges=5):
        self._llm = llm
        self._retriever = retriever
        self._model = model
        self._max_exchanges = max_exchanges
        self._messages = []

    def respond(self, query: str, model: str = None) -> str:
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

        answer = self._llm.chat(model or self._model, self._messages)
        self._messages.append({"role": "assistant", "content": answer})

        # system prompt + last N user/assistant pairs
        limit = 1 + 2 * self._max_exchanges
        if len(self._messages) > limit:
            self._messages = [self._messages[0]] + self._messages[-(limit - 1):]

        return answer

    def reset(self):
        self._messages = []
        return "Konuşma sıfırlandı."


if __name__ == "__main__":
    import argparse

    from app.llm import OpenRouterLLM, OllamaLLM
    from app.retrieval import default_retriever

    parser = argparse.ArgumentParser(description="Chat with the bot from the terminal")
    parser.add_argument("--backend", choices=["openrouter", "ollama"], default="openrouter")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    if args.backend == "openrouter":
        conversation = Conversation(OpenRouterLLM(), default_retriever(),
                                    args.model or settings.OPENROUTER_MODEL)
    else:
        conversation = Conversation(OllamaLLM(), default_retriever(),
                                    args.model or settings.OLLAMA_MODEL)

    print("Bilgi Bot'a hoş geldiniz. Çıkmak için 'q', sıfırlamak için 'sıfırla'.")
    while True:
        user_input = input("\n>> ")
        if user_input.lower() in ["çıkış", "exit", "quit", "q"]:
            break
        if user_input.lower() in ["sıfırla", "reset"]:
            print(conversation.reset())
            continue
        print(f"\n{conversation.respond(user_input)}")
