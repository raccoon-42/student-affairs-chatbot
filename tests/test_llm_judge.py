"""End-to-end evaluation: real API server -> real retrieval -> real LLM,
judged by an LLM. Needs the whole stack running, so it's marked
`integration` and excluded from a plain `pytest` run.

Run with: pytest -m integration tests/test_llm_judge.py
"""
import glob
import json
import os

import pytest
import requests

from config import settings
from app.llm import OpenRouterLLM, OllamaLLM
from tests.evaluators.llm_judge import LLMJudge

OPENROUTER_MODEL_TO_TEST = settings.OPENROUTER_MODEL
LOCAL_MODEL_TO_TEST = settings.OLLAMA_MODEL

USE_LOCAL_MODEL = False  # True: chat via Ollama endpoint instead of OpenRouter
USE_LOCAL_JUDGE = False  # True: judge with Ollama instead of OpenRouter

pytestmark = pytest.mark.integration


def ask_chatbot(query: str, model_name: str, local: bool) -> str:
    endpoint = "/chat_local" if local else "/chat"
    response = requests.get(
        f"{settings.API_URL}{endpoint}",
        params={"query": query, "model_name": model_name},
    )
    response.raise_for_status()
    return response.json()["response"]


@pytest.fixture
def llm_judge():
    if USE_LOCAL_JUDGE:
        return LLMJudge(OllamaLLM(), settings.OLLAMA_MODEL)
    return LLMJudge(OpenRouterLLM(), settings.JUDGE_MODEL)


@pytest.fixture
def test_cases():
    """Load test cases from JSON files in the test_cases directory."""
    test_cases_dir = os.path.join(os.path.dirname(__file__), 'test_cases')
    all_test_cases = []

    for json_file in glob.glob(os.path.join(test_cases_dir, '*.json')):
        with open(json_file, 'r', encoding='utf-8') as f:
            file_test_cases = json.load(f)
        category = os.path.splitext(os.path.basename(json_file))[0]
        for test_case in file_test_cases:
            test_case["category"] = category
        all_test_cases.extend(file_test_cases)

    assert all_test_cases, "No test cases found in tests/test_cases/"
    return all_test_cases


def test_llm_responses(llm_judge, test_cases):
    results = []
    failures = []

    model_name = LOCAL_MODEL_TO_TEST if USE_LOCAL_MODEL else OPENROUTER_MODEL_TO_TEST

    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        expected_response = test_case["expected"]

        response = ask_chatbot(query, model_name, local=USE_LOCAL_MODEL)
        evaluation = llm_judge.evaluate_response(query, response, expected_response)
        evaluation["category"] = test_case["category"]
        results.append(evaluation)

        print(f"\n----- Test Case {i}: {test_case['description']} [{test_case['category']}] -----")
        print(f"Sorulan soru: {query}")
        print(f"Beklenen cevap: {expected_response}")
        print(f"Verilen cevap: {response}")
        print(f"Puan: {evaluation['score']}")
        print(f"Mantık yürütme: {evaluation['reasoning']}")

        if evaluation["score"] != 1:
            failures.append(evaluation)

    print("\n===== LLM EVALUATION RESULTS =====")
    print(f"Model tested: {model_name}")
    print(f"Passed: {len(results) - len(failures)}/{len(results)}")

    assert not failures, (
        f"{len(failures)}/{len(results)} cases failed: "
        + "; ".join(f"{f['query']} (score={f['score']})" for f in failures)
    )
