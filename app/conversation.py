"""One Conversation module for every backend.

Prompt assembly, retrieval context, and history trimming live here once.
Which LLM answers is decided by the adapter passed in (OpenRouter, Ollama,
or a fake in tests) — history is owned by the instance, not the module.
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor

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

    # SIKÇA SORULAN SORULAR:
    {faq_context}
    </available_reference_data>
</conversation>

GÖREV:
1. SADECE <student_question> etiketleri arasındaki soruyu yanıtla.
2. <available_reference_data> içindeki bilgileri SADECE öğrencinin sorusuna yanıt vermek için kullan.
3. Eğer öğrencinin sorusu belirsizse veya eksik bilgi varsa, açıklama iste.
4. Referans verilerini doğrudan paylaşma, sadece soruya yanıt vermek için kullan.
"""

REWRITE_PROMPT = """Görevin bir sohbetteki son öğrenci mesajını, sohbet geçmişi olmadan da anlaşılacak bağımsız bir soruya dönüştürmek.
- Soruyu CEVAPLAMA, sadece yeniden yaz.
- Mesaj zaten bağımsız anlaşılıyorsa olduğu gibi döndür.
- Sadece yeniden yazılmış soruyu döndür, açıklama ekleme.

Sohbet geçmişi:
{history}

Son mesaj: {query}"""


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


class Conversation:
    def __init__(self, llm, retriever, model, max_exchanges=5, gate=None, rewriter=None):
        self._llm = llm
        self._retriever = retriever
        self._model = model
        self._max_exchanges = max_exchanges
        self._gate = gate
        self._rewriter = rewriter
        self._messages = []

    def respond(self, query: str, model: str = None) -> str:
        if not self._prepare(query):
            return REFUSAL_MESSAGE

        start = time.perf_counter()
        answer = self._llm.chat(model or self._model, self._messages)
        print(f"[timing] llm {time.perf_counter() - start:.2f}s", file=sys.stderr)
        self._add_assistant_message(answer)
        return answer

    def respond_stream(self, query: str, model: str = None):
        """Same as respond(), but yields the answer token by token."""
        if not self._prepare(query):
            yield REFUSAL_MESSAGE
            return

        start = time.perf_counter()
        first_token_at = None
        tokens = []
        for token in self._llm.chat_stream(model or self._model, self._messages):
            if first_token_at is None:
                first_token_at = time.perf_counter() - start
            tokens.append(token)
            yield token
        print(f"[timing] llm first token {first_token_at:.2f}s, "
              f"total {time.perf_counter() - start:.2f}s", file=sys.stderr)
        self._add_assistant_message("".join(tokens))

    def _prepare(self, query) -> bool:
        """Gate and retrieval run concurrently: retrieval is speculative,
        its result is discarded if the gate blocks. Returns False when
        the query is refused."""
        start = time.perf_counter()
        # follow-ups get rewritten into standalone questions for the gate
        # and retrieval; the model itself still sees the original message
        search_query = query
        if self._rewriter is not None:
            search_query = self._rewriter.rewrite(query, self._messages[1:])
            if search_query != query:
                print(f"[rewrite] {query!r} -> {search_query!r}", file=sys.stderr)

        if self._gate is None:
            results = self._retriever.retrieve_all(search_query)
        else:
            with ThreadPoolExecutor(max_workers=2) as pool:
                allowed = pool.submit(self._gate.allows, search_query)
                speculative = pool.submit(self._retriever.retrieve_all, search_query)
                if not allowed.result():
                    return False
                results = speculative.result()
        print(f"[timing] gate + retrieval {time.perf_counter() - start:.2f}s", file=sys.stderr)

        context = CONTEXT_TEMPLATE.format(
            query=query,
            calendar_context="\n".join(r["text"] for r in results["calendar"]),
            regulations_context="\n".join(r["text"] for r in results["regulations"]),
            faq_context="\n\n".join(r["text"] for r in results["faq"]),
        )

        if not self._messages:
            self._messages.append({"role": "system", "content": settings.load_system_prompt()})
        self._messages.append({"role": "user", "content": f"Öğrenci: {query}\n\n{context}"})
        self._pending_query = query
        return True

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
