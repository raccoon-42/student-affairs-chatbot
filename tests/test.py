import unittest
from ..app import chatbot  # Import your LLM function

class TestLLMAsJudge(unittest.TestCase):
    
    def test_accuracy(self):
        test_cases = {
            "What is the capital of France?": "Paris",
            "Who wrote 'Hamlet'?": "William Shakespeare",
            "What is the largest planet in our solar system?": "Jupiter",
        }
        
        for query, expected_response in test_cases.items():
            with self.subTest(query=query):
                response = llm_judge(query)  # Call your LLM function
                self.assertEqual(response, expected_response)

    def test_decision_making(self):
        decision_cases = {
            "Is it better to save or invest money?": "It depends on your financial goals.",
            "Should I take a job offer or stay at my current job?": "Consider your career goals and job satisfaction.",
        }
        
        for query, expected_response in decision_cases.items():
            with self.subTest(query=query):
                response = llm_judge(query)
                self.assertIn(expected_response, response)  # Check if expected response is part of the output

if __name__ == '__main__':
    unittest.main()