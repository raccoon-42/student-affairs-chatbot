from fastapi import FastAPI
from pydantic import BaseModel
from app.chatbot import chat_with_bot
from app.chatbot_local import chat_with_bot as chat_with_bot_local

app = FastAPI(
    title="Student Affairs Chatbot API",
    description="API for querying academic calendar and regulations information",
    version="1.0.0"
)

class ChatResponse(BaseModel):
    query: str
    response: str
    model: str
    
@app.get("/")
async def root():
    return {"message": "Welcome to Student Affairs Chatbot API"}


@app.get("/chat", response_model=ChatResponse)
async def chat(query: str, model_name: str):
    # Call the actual chatbot implementation
    response = chat_with_bot(query, model_name)
    return ChatResponse(
        query=query,
        response=response,
        model=model_name
    )

@app.get("/chat_local", response_model=ChatResponse)
async def chat(query: str, model_name: str):
    # Call the actual chatbot implementation
    response = chat_with_bot_local(query, model_name)
    return ChatResponse(
        query=query,
        response=response,
        model=model_name
    )