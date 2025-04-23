import pytest
import sys
import os
import argparse
import json
import glob

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from tests.utils.llm_judge import LLMJudge
from app.client.api_client import ChatbotClient

<<<<<<< Updated upstream
=======
OPENAI_MODEL_TO_TEST = "google/gemini-2.0-flash-001"
LOCAL_MODEL_TO_TEST = "llama3.1:latest"

>>>>>>> Stashed changes
@pytest.fixture
def llm_judge(model_name="gemma3:4b"):
    # Initialize a new instance of the LLM Judge for each test
    return LLMJudge(model_name=model_name)

@pytest.fixture
def test_cases():
    """
    Load test cases from JSON files in the test_cases directory.
    Each JSON file contains a list of test cases in the format:
    [
        {
            "query": "Question to ask",
            "expected": "Expected response",
            "description": "Description of the test"
        }
    ]
    """
    test_cases_dir = os.path.join(os.path.dirname(__file__), 'test_cases')
    json_files = glob.glob(os.path.join(test_cases_dir, '*.json'))
    
    all_test_cases = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                file_test_cases = json.load(f)
                # Add the category based on the filename
                category = os.path.splitext(os.path.basename(json_file))[0]
                for test_case in file_test_cases:
                    test_case["category"] = category
                all_test_cases.extend(file_test_cases)
        except Exception as e:
            print(f"Error loading test cases from {json_file}: {e}")
    
    if not all_test_cases:
        print("Warning: No test cases found. Using fallback test cases.")
        # Fallback to hardcoded test cases if no files are found or loaded
        all_test_cases = [
            {
                "query": "yaz okulu kayıtları ne zaman",
                "expected": "8-9 temmuz 2025'de başlar",
                "description": "Yaz okulu kayıt tarihleri",
                "category": "fallback"
            }
        ]
    
    return all_test_cases

def test_llm_responses(llm_judge, test_cases):
    client = ChatbotClient()

    results = []
    failures = []
    categories = {}
    
<<<<<<< Updated upstream
    for i, (query, expected_response) in enumerate(test_cases.items(), 1):
        # Get response from LLM
        response = client.get_response_openai(query) # openAI
        #response = client.get_response_local(query, "gemma3:4b") # local
=======
    use_local_model = True  # Set to False to test with OpenAI API instead
    
    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        expected_response = test_case["expected"]
        description = test_case["description"]
        category = test_case.get("category", "uncategorized")
        
        # Track results by category
        if category not in categories:
            categories[category] = {"total": 0, "passed": 0}
        categories[category]["total"] += 1
        
        # Get response from LLM based on selected model
        if use_local_model:
            response = client.get_response_local(query, LOCAL_MODEL_TO_TEST)
        else:
            response = client.get_response_openai(query, OPENAI_MODEL_TO_TEST)
>>>>>>> Stashed changes
        
        # Evaluate the response
        evaluation = llm_judge.evaluate_response(query, response, expected_response)
        evaluation["category"] = category
        results.append(evaluation)
        
        # Always print the evaluation details
        print(f"\n----- Test Case {i}: {description} [{category}] -----")
        print(f"Sorulan soru: {query}")
        print(f"Beklenen cevap: {expected_response}")
        print(f"Verilen cevap: {response}")
        print(f"Puan: {evaluation['score']}")
        print(f"Mantık yürütme: {evaluation['reasoning']}")
        print("-" * 50)
        
        # Check if it passes the threshold
        if evaluation["score"] == 0:
            failures.append(evaluation)
            print(f"❌ FAILED")
        else:
            categories[category]["passed"] += 1
            print(f"✅ PASSED")
            
        #assert evaluation["score"] == 1, \
        #        f"Query: {query}\nExpected: {expected_response}\nGot: {response}\nScore: {evaluation['score']}\nReasoning: {evaluation['reasoning']}"
    
    # This will be displayed only if all tests pass
    print("\n===== LLM EVALUATION RESULTS =====")
    model_info = f"LOCAL: {LOCAL_MODEL_TO_TEST}" if use_local_model else f"OPENAI: {OPENAI_MODEL_TO_TEST}"
    print(f"Model tested: {model_info}")
    print(f"Total test cases: {len(results)}")
    print(f"Passed: {len(results) - len(failures)}")
    print(f"Failed: {len(failures)}")
    print(f"Average score: {sum(r['score'] for r in results) / len(results):.2f}")
    
    # Print category-specific results
    print("\n----- Results by Category -----")
    for category, stats in categories.items():
        success_rate = (stats["passed"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        print(f"{category.capitalize()}: {stats['passed']}/{stats['total']} passed ({success_rate:.1f}%)")