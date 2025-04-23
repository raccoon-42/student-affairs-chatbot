from fastapi import FastAPI
from pydantic import BaseModel
from app.chatbot import chat_with_bot
from app.chatbot_local import chat_with_bot as chat_with_bot_local

app = FastAPI(
    title="Academic Calendar Chatbot API",
    description="API for querying academic calendar information",
    version="1.0.0"
)

class ChatResponse(BaseModel):
    query: str
    response: str

@app.get("/")
async def root():
    return {"message": "Welcome to the Academic Calendar Chatbot API"}


@app.get("/chat", response_model=ChatResponse)
async def chat(query: str):
    # Call the actual chatbot implementation
    response = chat_with_bot(query)
    return ChatResponse(
        query=query,
        response=response
    )

@app.get("/chat_local", response_model=ChatResponse)
async def chat(query: str):
    # Call the actual chatbot implementation
    response = chat_with_bot_local(query)
    return ChatResponse(
        query=query,
        response=response
    )