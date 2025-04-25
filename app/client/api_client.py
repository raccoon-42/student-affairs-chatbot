import requests
from typing import Optional

# Wrapper for OgrenciİsleriChatbot API

class ChatbotClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def get_response_openrouter(self, query: str, model_name: str) -> str:
        response = requests.get(
            f"{self.base_url}/chat",
            params={"query": query, "model_name": model_name}
        )
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")

    def get_response_local(self, query: str, model_name: str) -> str:
        response = requests.get(
            f"{self.base_url}/chat_local",
            params={"query": query, "model_name": model_name}
        )
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")