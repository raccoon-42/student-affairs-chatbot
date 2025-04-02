from typing import Dict, Any
from .ollama_client import OllamaClient
import requests

class LLMJudge:
    def __init__(self, model_name: str):
        self.__model_name = model_name
        self.ollama_client = OllamaClient()
    
    def get_response(self, query: str) -> str:
        url = "http://localhost:8000/chat"
        response = requests.get(url, 
                        params = {
                            "query": query,
                        }
                    )
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")
        
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
        evaluation_prompt = f"""
        Sorulan soru: {query}
        Beklenen cevap: {expected_response}
        Verilen cevap: {response}
        
        Lütfen "verilen cevap" ile "beklenen cevap"a uyup uymadığını puanla.
        
        Selamlamaları ve ek bilgileri göz ardı ederek, sorulan soruya verilen cevabnın içerilip içerilmediğini kontrol et. Eğer içeriyorsa 1, içermiyorsa 0 puan ver.
        
        Eğer verilen cevap sorulan soruya daha net bir cevap vermek amacıyla bir soru soruyorsa buna 1 puan ver.

        Aşağıdaki formatta cevap verin:
        Puan:
        Mantık yürütme:
        
        """
        
        
        evaluation_result = self.ollama_client.generate_response(self.__model_name, evaluation_prompt)
        
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
