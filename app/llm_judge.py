from typing import Dict, Any
from .ollama_client import OllamaClient
import requests

class LLMJudge:
    def __init__(self, model_name: str):
        self.__model_name = model_name
        self.ollama_client = OllamaClient()
    
    def get_response(self, query: str) -> str:
        url = "http://localhost:8000/chat"
        params = {"query": query}
        response = requests.get(url, params=params)
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
        Query: {query}
        Expected Response: {expected_response}
        Actual Response: {response}
        
        Please evaluate the actual response against the expected response.
        Consider:
        1. Semantic correctness
        2. Completeness
        3. Accuracy
        
        Provide a score (0-1) and brief reasoning.
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
        score_match = re.search(r'Score:\s*(\d+\.\d+)', evaluation_text)
        if score_match:
            score = float(score_match.group(1))
        else:
            # Fallback - try to find any float value that could be a score
            float_matches = re.findall(r'(\d+\.\d+)', evaluation_text)
            if float_matches and 0 <= float(float_matches[0]) <= 1:
                score = float(float_matches[0])
            else:
                score = 0.0
                
        # Extract reasoning - everything after the score
        if score_match:
            reasoning_text = evaluation_text[score_match.end():]
            reasoning = reasoning_text.strip()
        else:
            reasoning = evaluation_text
            
        return {
            "score": score,
            "reasoning": reasoning,
            "query": query,
            "response": response,
            "expected_response": expected_response
        } 
