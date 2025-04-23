from app.query_handler import query_qdrant_regulations, query_qdrant_academic_calendar
from openai import OpenAI
import os
from dotenv import load_dotenv
from app.ollama_client import OllamaClient  
load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

client = OllamaClient()
LLM_MODEL = "gemma3:4b"

def load_system_prompt():
    base_dir = os.path.dirname(__file__)  # path to chatbot_local.py
    prompt_path = os.path.join(base_dir, "../config/prompts/system_prompt.txt")
    with open(os.path.abspath(prompt_path), "r", encoding="utf-8") as f:
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
    context = f"""
    <conversation>
        <student_question>
        {user_query}
        </student_question>

        <available_reference_data>
        # AKADEMİK TAKVİM BİLGİLERİ:
        {calendar_context}

        # YÖNETMELİK BİLGİLERİ:
        {regulations_context}
        </available_reference_data>
    </conversation>

    GÖREV: 
    1. SADECE <student_question> etiketleri arasındaki soruyu yanıtla.
    2. <available_reference_data> içindeki bilgileri SADECE öğrencinin sorusuna yanıt vermek için kullan.
    3. Eğer öğrencinin sorusu belirsizse veya eksik bilgi varsa, açıklama iste.
    4. Referans verilerini doğrudan paylaşma, sadece soruya yanıt vermek için kullan.
    """

    if not messages:
        system_prompt = load_system_prompt()
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": context})
    
    # Generate the response
    print("Yanıt oluşturuluyor...")
    completion = client.chat_completion(LLM_MODEL, messages)
   
    # Print API response for debugging
    print(f"API response structure: {type(completion)}")
    
    # Check if completion has the expected structure
    #if not hasattr(messages, 'messages') or not completion.message:
     #   print("API ERROR: 'choices' field missing or empty in response")
      #  print(f"Full API response: {completion}")
       # return "Üzgünüm, teknik bir sorun oluştu. Lütfen tekrar deneyin."
        
    # Store assistant's response to maintain conversation flow
    assistant_message = completion['message']['content']
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