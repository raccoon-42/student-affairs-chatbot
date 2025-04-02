import pytest
import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from app.llm_judge import LLMJudge

@pytest.fixture
def llm_judge():
    # Initialize a new instance of the LLM Judge for each test
    return LLMJudge()

@pytest.fixture
def test_cases():
    return {
        "yaz okulu kayıtları ne zaman": "8-9 temmuz'da başlar",
        # Add more test cases as needed
    }

def test_llm_responses(llm_judge, test_cases):
    results = []
    failures = []
    
    for query, expected_response in test_cases.items():
        # Get response from LLM
        response = llm_judge.get_response(query)
        
        # Evaluate the response
        evaluation = llm_judge.evaluate_response(query, response, expected_response)
        results.append(evaluation)
        
        # Always print the evaluation details
        print(f"\n----- Test Case -----")
        print(f"Query: {query}")
        print(f"Expected: {expected_response}")
        print(f"Got: {response}")
        print(f"Score: {evaluation['score']}")
        print(f"Reasoning: {evaluation['reasoning']}")
        print("-" * 30)
        
        # Check if it passes the threshold
        if evaluation["score"] < 0.7:
            failures.append(evaluation)
            print(f"❌ FAILED - Score below threshold (0.7)")
        else:
            print(f"✅ PASSED - Score meets threshold (0.7)")
        
        # Still perform the assertion for test pass/fail
        assert evaluation["score"] >= 0.7, \
            f"Query: {query}\nExpected: {expected_response}\nGot: {response}\nScore: {evaluation['score']}\nReasoning: {evaluation['reasoning']}"
    
    # This will be displayed only if all tests pass
    print("\n===== LLM EVALUATION RESULTS =====")
    print(f"Total test cases: {len(results)}")
    print(f"Passed: {len(results) - len(failures)}")
    print(f"Failed: {len(failures)}")
    print(f"Average score: {sum(r['score'] for r in results) / len(results):.2f}")

if __name__ == "__main__":
    # Create an instance of LLMJudge
    judge = LLMJudge(model_name="gemma3:4b")
    
    # Define test cases
    test_cases = {
        "yaz okulu kayıtları ne zaman": "8-9 temmuz'da başlar",
        # Add more test cases as needed
    }
    
    # Run tests manually and collect results
    results = []
    failures = []
    
    print("Running LLM evaluation tests...")
    
    for query, expected_response in test_cases.items():
        print(f"\n----- Test Case -----")
        print(f"Query: {query}")
        print(f"Expected: {expected_response}")
        
        # Get response from LLM
        try:
            response = judge.get_response(query)
            print(f"Got: {response}")
            
            # Evaluate the response
            evaluation = judge.evaluate_response(query, response, expected_response)
            results.append(evaluation)
            
            print(f"Score: {evaluation['score']}")
            print(f"Reasoning: {evaluation['reasoning']}")
            print("-" * 30)
            
            # Check if it passes the threshold
            if evaluation["score"] < 0.7:
                failures.append(evaluation)
                print(f"❌ FAILED - Score below threshold (0.7)")
            else:
                print(f"✅ PASSED - Score meets threshold (0.7)")
                
        except Exception as e:
            print(f"Error during test: {str(e)}")
            print("❌ FAILED - Test execution error")
    
    # Display summary results
    if results:
        print("\n===== LLM EVALUATION RESULTS =====")
        print(f"Total test cases: {len(test_cases)}")
        print(f"Completed: {len(results)}")
        print(f"Passed: {len(results) - len(failures)}")
        print(f"Failed: {len(failures)}")
        
        if len(results) > 0:
            print(f"Average score: {sum(r['score'] for r in results) / len(results):.2f}")