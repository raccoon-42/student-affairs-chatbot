from query_handler import query_qdrant_regulations, query_qdrant_academic_calendar
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENAI_API_KEY
)

def load_system_prompt():
    with open("../config/prompts/system_prompt.txt", "r", encoding="utf-8") as f:
        return f.read()

# Keep track of conversation to maintain context
messages = []
is_first_message = True

def chat_with_bot(user_query):
    """fetches relevant context and generates response"""
    global messages
    global is_first_message
    
    print("Querying academic calendar...")
    # Query both collections and combine results
    calendar_results = query_qdrant_academic_calendar(user_query)
    calendar_context = "\n".join(result["text"] for result in calendar_results)
    
    print("Querying regulations...")
    regulations_results = query_qdrant_regulations(user_query)
    regulations_context = "\n".join(result["text"] for result in regulations_results)
  
    # Combine contexts with clear separation
    context = f"=== Academic Calendar Information ===\n{calendar_context}\n\n=== Regulations Information ===\n{regulations_context}"
    
    # Load the system prompt (only once)
    if not messages:
        system_prompt = load_system_prompt()
        messages.append({"role": "system", "content": system_prompt})
    
    # Add new user message with context
    if is_first_message:
        user_message = f"Öğrenci ilk sorusunu sordu: {user_query}\n\n{context}"
        is_first_message = False
    else:
        user_message = f"Öğrenci: {user_query}\n\n{context}"
    
    messages.append({"role": "user", "content": user_message})
    
    # Generate the response
    print("Yanıt oluşturuluyor...")
    completion = client.chat.completions.create(
        model="google/gemini-2.0-flash-001",
        messages=messages
    )
    
    # Print API response for debugging
    print(f"API response structure: {type(completion)}")
    
    # Check if completion has the expected structure
    if not hasattr(completion, 'choices') or not completion.choices:
        print("API ERROR: 'choices' field missing or empty in response")
        print(f"Full API response: {completion}")
        return "Üzgünüm, teknik bir sorun oluştu. Lütfen tekrar deneyin."
        
    # Store assistant's response to maintain conversation flow
    assistant_message = completion.choices[0].message.content
    messages.append({"role": "assistant", "content": assistant_message})
    
    # Keep conversation history manageable (last 5 exchanges)
    if len(messages) > 11:  # system + 5 pairs of messages
        messages = [messages[0]] + messages[-10:]
    
    return assistant_message
 

def reset_conversation():
    """Reset the conversation history"""
    global messages
    global is_first_message
    messages = []
    is_first_message = True
    return "Konuşma sıfırlandı."

if __name__ == "__main__":
    print("🎓 Bilgi Bot'a Hoş Geldiniz! Akademik takvim, dersler ve okul hakkında sorularınızı sorabilirsiniz.")
    print("Konuşmayı sıfırlamak için 'sıfırla' yazabilirsiniz.")
    
    while True:
        user_input = input("\n>> ")
        if user_input.lower() in ['çıkış', 'exit', 'quit', 'q']:
            print("İyi günler! Başka bir sorunuz olursa tekrar bekleriz.")
            break
        elif user_input.lower() in ['sıfırla', 'reset']:
            print(reset_conversation())
            continue
            
        response = chat_with_bot(user_input)
        print(f"\n{response}")