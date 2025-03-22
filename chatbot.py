from config import OPENAI_API_KEY
from query_handler import query_qdrant
from openai import OpenAI

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENAI_API_KEY
)

def chat_with_bot(user_query):
    """fetches relevant context"""
    
    context = "\n".join(query_qdrant(user_query, collection_name='academic_calendar_2025'))
    completion=client.chat.completions.create(
        model="google/gemini-2.0-flash-001",
        messages=[
            {"role": "system", "content": "Sen arkadaş canlısı bir öğrenci işleri botusun, uslu dur."},
            {"role": "user", "content": f"Kullanıcı şunu sordu: {user_query}\n\nBağlam:\n{context}"}
        ]
    )
    return completion.choices[0].message.content
    
# Example usage:
if __name__ == "__main__":
    while(1):
        user_input = input(">> ")
        print("🤖", chat_with_bot(user_input))