# ADR-0002: One Conversation module, LLMs behind one seam

**Decision.** One `Conversation` module owns prompt assembly and history.
Every LLM consumer (conversation, judge) depends on the interface
`chat(model, messages) -> str` (`app/llm.py`); OpenRouter and Ollama are
adapters behind it. The judge uses OpenRouter by default, Ollama stays
available.

**Why.** `chatbot.py` and `chatbot_local.py` were whole-file duplicates that
had already drifted (history trim kept 11 vs 2 messages), with module-global
conversation state shared across all API users. Two providers = a real seam;
a fake adapter makes conversation and judge testable offline.

**Consequence.** History belongs to a `Conversation` instance. The API still
holds one instance per backend per process — per-user sessions are a TODO.
