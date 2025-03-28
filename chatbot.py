from config import OPENAI_API_KEY
from query_handler import query_qdrant
from openai import OpenAI
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENAI_API_KEY
)

def load_system_prompt():
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        return f.read()

# Keep track of conversation to maintain context
messages = []
is_first_message = True

def chat_with_bot(user_query):
    """fetches relevant context and generates response"""
    global messages
    global is_first_message
    
    try:
        # Get relevant information from the vector store
        context = "\n".join(query_qdrant(user_query, collection_name='academic_calendar_2025'))
        
        # Load the system prompt (only once)
        if not messages:
            system_prompt = load_system_prompt()
            messages.append({"role": "system", "content": system_prompt})
        
        # Add new user message with context
        if is_first_message:
            user_message = f"Öğrenci ilk sorusunu sordu: {user_query}\n\nAkademik takvim bilgileri:\n{context}"
            is_first_message = False
        else:
            user_message = f"Öğrenci: {user_query}\n\nAkademik takvim bilgileri:\n{context}"
        
        messages.append({"role": "user", "content": user_message})
        
        # Generate the response
        print("Yanıt oluşturuluyor...")
        completion = client.chat.completions.create(
            model="google/gemini-2.0-flash-001",  # Try a different model
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
        
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {str(e)}")
        return f"Üzgünüm, bir hata oluştu: {str(e)}. Lütfen tekrar deneyin veya sistem yöneticisine başvurun."

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