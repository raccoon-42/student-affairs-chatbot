import time
import uuid
from functools import lru_cache
from threading import Lock

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.conversation import Conversation, QueryRewriter
from app.guardrails import ScopeGate
from app.llm import OpenRouterLLM, OllamaLLM
from app.retrieval import default_retriever
from config import settings

app = FastAPI(
    title="Student Affairs Chatbot API",
    description="API for querying academic calendar and regulations information",
    version="1.0.0",
)

SESSION_TTL_SECONDS = 30 * 60


class ChatResponse(BaseModel):
    query: str
    response: str
    model: str
    session_id: str


@lru_cache
def _backend(name: str):
    """LLM, retriever and model names are shared by every session;
    only the conversation history is per-session."""
    retriever = default_retriever()
    if name == "ollama":
        # local backend gates and rewrites locally too, so it stays fully offline
        llm = OllamaLLM()
        return llm, retriever, settings.OLLAMA_MODEL, settings.OLLAMA_MODEL
    llm = OpenRouterLLM()
    return llm, retriever, settings.OPENROUTER_MODEL, settings.GUARD_MODEL


_sessions: dict = {}  # (backend, session_id) -> (Conversation, last_used)
_sessions_lock = Lock()


def get_conversation(session_id: str, backend: str = "openrouter") -> Conversation:
    """One conversation per session id, evicted after SESSION_TTL_SECONDS
    of inactivity."""
    now = time.time()
    with _sessions_lock:
        for key in [k for k, (_, last_used) in _sessions.items()
                    if now - last_used > SESSION_TTL_SECONDS]:
            del _sessions[key]

        key = (backend, session_id)
        if key not in _sessions:
            llm, retriever, model, small_model = _backend(backend)
            conversation = Conversation(llm, retriever, model,
                                        gate=ScopeGate(llm, small_model),
                                        rewriter=QueryRewriter(llm, small_model))
            _sessions[key] = (conversation, now)
        conversation, _ = _sessions[key]
        _sessions[key] = (conversation, now)
        return conversation


@app.get("/")
async def root():
    return {"message": "Welcome to Student Affairs Chatbot API"}


@app.get("/chat", response_model=ChatResponse)
async def chat(query: str, session_id: str = None, model_name: str = None):
    session_id = session_id or uuid.uuid4().hex
    model = model_name or settings.OPENROUTER_MODEL
    response = get_conversation(session_id).respond(query, model)
    return ChatResponse(query=query, response=response, model=model, session_id=session_id)


@app.get("/chat/stream")
async def chat_stream(query: str, session_id: str = None, model_name: str = None):
    session_id = session_id or uuid.uuid4().hex
    conversation = get_conversation(session_id)
    return StreamingResponse(
        conversation.respond_stream(query, model_name or settings.OPENROUTER_MODEL),
        media_type="text/plain; charset=utf-8",
        headers={"X-Session-Id": session_id},
    )


@app.get("/chat_local", response_model=ChatResponse)
async def chat_local(query: str, session_id: str = None, model_name: str = None):
    session_id = session_id or uuid.uuid4().hex
    model = model_name or settings.OLLAMA_MODEL
    response = get_conversation(session_id, backend="ollama").respond(query, model)
    return ChatResponse(query=query, response=response, model=model, session_id=session_id)
