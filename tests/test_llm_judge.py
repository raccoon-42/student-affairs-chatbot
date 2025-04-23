import pytest
import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from tests.utils.llm_judge import LLMJudge
from app.client.api_client import ChatbotClient

@pytest.fixture
def llm_judge(model_name="gemma3:4b"):
    # Initialize a new instance of the LLM Judge for each test
    return LLMJudge(model_name=model_name)

@pytest.fixture
def test_cases():
    return {
        "yaz okulu kayıtları ne zaman": "8-9 temmuz 2025'de başlar",
        "dersten çekilmek için son tarih nedir?" : "hangi dönem çekilmek istediği sorusunu sorar ya da 25 nisan 2025 diye söyler.",
        "bahar dönemi için dersten çekilmek için son tarih nedir?" : "25 nisan 2025 diye söyler.",
        "okul ne zaman kapanıyor?": "hangi dönem için sorduğunu sorar ya da sınavların ne zaman bittiğini söyler.",
        "bahar dönemi sınavları ne zaman bitiyor?": "10-23 haziran 2025 arası",
        "mezuniyet ne zaman?": "18 temmuz 2025"
    }

def test_llm_responses(llm_judge, test_cases):
    client = ChatbotClient()

    results = []
    failures = []
    
    for i, (query, expected_response) in enumerate(test_cases.items(), 1):
        # Get response from LLM
        response = client.get_response_openai(query) # openAI

        
        # Evaluate the response
        evaluation = llm_judge.evaluate_response(query, response, expected_response)
        results.append(evaluation)
        
        # Always print the evaluation details
        print(f"\n----- Test Case {i}-----")
        print(f"Sorulan soru: {query}")
        print(f"Beklenen cevap: {expected_response}")
        print(f"Verilen cevap: {response}")
        print(f"Puan: {evaluation['score']}")
        print(f"Mantık yürütme: {evaluation['reasoning']}")
        print("-" * 30)
        
        # Check if it passes the threshold
        if evaluation["score"] == 0:
            failures.append(evaluation)
            print(f"❌ FAILED")
        else:
            print(f"✅ PASSED")
            
        assert evaluation["score"] == 1, \
                f"Query: {query}\nExpected: {expected_response}\nGot: {response}\nScore: {evaluation['score']}\nReasoning: {evaluation['reasoning']}"
    
    # This will be displayed only if all tests pass
    print("\n===== LLM EVALUATION RESULTS =====")
    print(f"Total test cases: {len(results)}")
    print(f"Passed: {len(results) - len(failures)}")
    print(f"Failed: {len(failures)}")
    print(f"Average score: {sum(r['score'] for r in results) / len(results):.2f}")