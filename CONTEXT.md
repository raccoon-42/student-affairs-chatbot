# Domain glossary

- **Academic Calendar** — dated university events (exams, registration, holidays), parsed from the calendar PDF into one chunk per event.
- **Regulation** — a numbered article (MADDE) from the university regulations PDF, one chunk per article.
- **FAQ** — one question/answer pair scraped from the university's FAQ sources (2 PDFs + 1 HTML page), tagged with `audience` (hazirlik/lisans/lisansustu) and `category`. Only the question is embedded; the answer travels in metadata.
- **Chunk** — one indexed unit: `{text, metadata}`. Calendar chunks carry `event_type`, `academic_period`, dates; regulation chunks carry section/article info; FAQ chunks carry the answer and audience.
- **Retriever** (`app/retrieval.py`) — the one interface for finding chunks: `retrieve_calendar(query)` / `retrieve_regulations(query)`. Hides embedding, vector search, filters, and hybrid scoring.
- **Hybrid score** — `0.7 * semantic similarity + 0.3 * BM25`, computed in one place inside the Retriever.
- **Conversation** (`app/conversation.py`) — assembles the prompt (system prompt + retrieved context + history) and asks an LLM adapter for the answer.
- **LLM adapter** (`app/llm.py`) — anything with `chat(model, messages) -> str`. Two real ones: OpenRouter and Ollama.
- **Judge** (`tests/evaluators/llm_judge.py`) — an LLM that scores chatbot answers 0/1 against expected answers during evaluation.
- **Scope gate** (`app/guardrails.py`) — a cheap LLM call that decides whether a question is university-related before it reaches retrieval or the main model; off-topic questions get a canned refusal.
