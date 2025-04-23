from typing import Dict, Any
from app.client.ollama_client import OllamaClient

class LLMJudge:
    def __init__(self, model_name: str):
        self.__model_name = model_name
        self.ollama_client = OllamaClient()
    
    def evaluate_response(self, query: str, response: str, expected_response: str) -> Dict[str, Any]:
        """
        Evaluate a response against an expected response.
        
        Args:
            query (str): The original query
            response (str): The actual response from the LLM
            expected_response (str): The expected response
            
        Returns:
            Dict[str, Any]: Evaluation results including score and reasoning
        """
        
        JUDGE_PROMPT = f"""Sen bir akademik takvim asistanı chatbot'unun cevaplarını değerlendiren bir uzman değerlendiricisin. 

        Değerlendirme Kuralları:
        1. Selamlamaları ve ek bilgileri göz ardı et
        2. Sadece sorulan soruya verilen cevabın doğruluğuna odaklan
        3. ÖNEMLİ: Eğer soru belirsiz ise (örn. hangi dönem olduğu belli değilse) ve chatbot bu belirsizliği gidermek için soru soruyorsa, bu yanıtı DOĞRU kabul et
        4. Chatbot hem belirsizliği gidermek için soru sorup hem de genel bilgi veriyorsa, bu özellikle takdir edilmeli

        Örnek Değerlendirmeler:

        Örnek 1:
        Soru: "okul ne zaman kapanıyor?"
        Beklenen Cevap: "hangi dönem için sorduğunu sorar ya da sınavların ne zaman bittiğini söyler."
        Verilen Cevap: "Hangi dönemi kastettiğinizi öğrenebilir miyim? Güz ve bahar dönemlerinin kapanış tarihleri farklıdır."
        Puan: 1
        Mantık yürütme: Belirsizliği gidermek için soru sorulmuş ve ek bilgi de verilmiş.

        Örnek 2:
        Soru: "dersten çekilmek için son tarih nedir?"
        Beklenen Cevap: "hangi dönem çekilmek istediği sorusunu sorar ya da 25 nisan 2025 diye söyler."
        Verilen Cevap: "Hangi dönem için çekilmek istediğinizi belirtir misiniz? Bahar dönemi için son tarih 25 Nisan 2025'tir."
        Puan: 1
        Mantık yürütme: Hem dönem belirsizliğini gidermek için soru sorulmuş hem de yararlı bilgi verilmiş.

        Şimdi şu yanıtı değerlendir:
        Sorulan soru: {query}
        Beklenen cevap: {expected_response}
        Verilen cevap: {response}

        Aşağıdaki formatta cevap ver:
        Puan: (0 veya 1)
        Mantık yürütme: (Puanlamanın nedeni)
        """
        
        evaluation_result = self.ollama_client.generate_response(self.__model_name, JUDGE_PROMPT)
        
        # Extract the actual text response from the Ollama API result
        if isinstance(evaluation_result, dict) and 'response' in evaluation_result:
            evaluation_text = evaluation_result['response']
        else:
            evaluation_text = str(evaluation_result)
            
        # Parse the evaluation response to extract score and reasoning
        import re
        
        # Try to extract score using regex
        score_match = re.search(r'(?<=Puan:\s).*', evaluation_text)
        if score_match:
            score = int(score_match.group(0))
        else:
            score = ""
            
        reasoning_match = re.search(r'(?<=Mantık yürütme:\s).*', evaluation_text)
        if reasoning_match:
            reasoning = reasoning_match.group(0)
        else:
            reasoning = ""
    
        return {
            "score": score,
            "reasoning": reasoning,
            "query": query,
            "response": response,
            "expected_response": expected_response
        } 
