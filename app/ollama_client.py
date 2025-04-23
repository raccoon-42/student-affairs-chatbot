import requests
import json

class OllamaClient:
    def __init__(self, local_url="http://localhost:11434"):
        self.local_url = local_url

    def chat_completion(self, model, messages):
        url = f"{self.local_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        
        response = requests.post(url, data=json.dumps(payload))
       
        if response.status_code == 200:
            return response.json()  
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")
        
    def generate_response(self, model, prompt):
        url = f"{self.local_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False 
        }
        response = requests.post(url, data=json.dumps(payload))
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")
    


if __name__ == "__main__":
    client = OllamaClient()
    
    messages = [
        {"role": "user", "content": "Hello, how are you?"}
    ]

    print(client.chat_completion("llama3.1", messages))