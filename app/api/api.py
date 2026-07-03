from functools import lru_cache

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.conversation import Conversation
from app.guardrails import ScopeGate
from app.llm import OpenRouterLLM, OllamaLLM
from app.retrieval import default_retriever
from config import settings

app = FastAPI(
    title="Student Affairs Chatbot API",
    description="API for querying academic calendar and regulations information",
    version="1.0.0",
)


class ChatResponse(BaseModel):
    query: str
    response: str
    model: str


@lru_cache
def get_conversation(backend: str) -> Conversation:
    # One shared conversation per backend per process (same behavior as
    # before, now explicit). Per-user sessions are a TODO.
    retriever = default_retriever()
    if backend == "ollama":
        # local backend gates locally too, so it stays fully offline
        llm = OllamaLLM()
        return Conversation(llm, retriever, settings.OLLAMA_MODEL,
                            gate=ScopeGate(llm, settings.OLLAMA_MODEL))
    llm = OpenRouterLLM()
    return Conversation(llm, retriever, settings.OPENROUTER_MODEL,
                        gate=ScopeGate(llm, settings.GUARD_MODEL))


@app.get("/")
async def root():
    return {"message": "Welcome to Student Affairs Chatbot API"}


@app.get("/chat", response_model=ChatResponse)
async def chat(query: str, model_name: str = None):
    model = model_name or settings.OPENROUTER_MODEL
    response = get_conversation("openrouter").respond(query, model)
    return ChatResponse(query=query, response=response, model=model)


@app.get("/chat/stream")
async def chat_stream(query: str, model_name: str = None):
    conversation = get_conversation("openrouter")
    return StreamingResponse(
        conversation.respond_stream(query, model_name or settings.OPENROUTER_MODEL),
        media_type="text/plain; charset=utf-8",
    )


@app.get("/chat_local", response_model=ChatResponse)
async def chat_local(query: str, model_name: str = None):
    model = model_name or settings.OLLAMA_MODEL
    response = get_conversation("ollama").respond(query, model)
    return ChatResponse(query=query, response=response, model=model)
