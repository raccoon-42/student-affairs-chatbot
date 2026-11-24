"""End-to-end evaluation: real API server -> real retrieval -> real LLM,
judged by an LLM. Needs the whole stack running, so it's marked
`integration` and excluded from a plain `pytest` run.

Run with: pytest -m integration tests/test_llm_judge.py -v
Each test case is its own test, so one bad answer doesn't hide the rest;
select one with -k, e.g. -k "ilişik".
"""
import glob
import json
import os

import pytest
import requests

from config import settings
from app.llm import OpenRouterLLM
from tests.evaluators.llm_judge import RUN_COSTS, LLMJudge, cited_reference, log_failure

OPENROUTER_MODEL_TO_TEST = settings.OPENROUTER_MODEL

pytestmark = pytest.mark.integration


def ask_chatbot(query: str, model_name: str) -> tuple[str, list, dict]:
    """Returns (answer, reference entries, usage): the numbered chunks the
    model saw, so the judge can verify facts instead of guessing, and what
    the answer call cost."""
    # generous but finite: a stalled upstream call should fail this one
    # case, not freeze the whole run (the server retries internally too)
    response = requests.get(
        f"{settings.API_URL}/chat",
        params={"query": query, "model_name": model_name},
        timeout=300,
    )
    response.raise_for_status()
    body = response.json()
    entries, usage = [], None
    debug = requests.get(
        f"{settings.API_URL}/chat/debug",
        params={"session_id": body["session_id"]},
        timeout=30,
    )
    if debug.ok:
        entries = debug.json().get("reference", [])
        usage = debug.json().get("usage")
    return body["response"], entries, usage


def load_test_cases():
    """Collected at import time so pytest can parametrize one test per case."""
    test_cases_dir = os.path.join(os.path.dirname(__file__), 'test_cases')
    all_test_cases = []

    for json_file in sorted(glob.glob(os.path.join(test_cases_dir, '*.json'))):
        with open(json_file, 'r', encoding='utf-8') as f:
            file_test_cases = json.load(f)
        category = os.path.splitext(os.path.basename(json_file))[0]
        for test_case in file_test_cases:
            test_case["category"] = category
        all_test_cases.extend(file_test_cases)

    assert all_test_cases, "No test cases found in tests/test_cases/"
    return all_test_cases


TEST_CASES = load_test_cases()


@pytest.fixture(scope="module")
def llm_judge():
    return LLMJudge(OpenRouterLLM(reasoning_effort=settings.JUDGE_REASONING_EFFORT),
                    settings.JUDGE_MODEL)


@pytest.mark.parametrize(
    "test_case", TEST_CASES,
    ids=[f"{c['category']}: {c['description']}" for c in TEST_CASES])
def test_llm_response(llm_judge, test_case):
    model_name = OPENROUTER_MODEL_TO_TEST
    query = test_case["query"]
    expected_response = test_case["expected"]

    response, entries, answer_usage = ask_chatbot(query, model_name)
    reference = cited_reference(response, entries)
    evaluation = llm_judge.evaluate_response(query, response, expected_response,
                                             reference=reference)
    RUN_COSTS["cases"] += 1
    RUN_COSTS["chatbot"] += (answer_usage or {}).get("cost") or 0.0
    RUN_COSTS["judge"] += (evaluation["usage"] or {}).get("cost") or 0.0

    print(f"\nSorulan soru: {query}")
    print(f"Beklenen cevap: {expected_response}")
    print(f"Verilen cevap: {response}")
    print(f"Judge'a giden kaynak maddeleri: {len(reference.split(chr(10) * 2)) if reference else 0}")
    print(f"Puan: {evaluation['score']}")
    print(f"Mantık yürütme: {evaluation['reasoning']}")

    if evaluation["score"] != 1:
        RUN_COSTS["failed"] += 1
        log_failure({
            "case": f"{test_case['category']}: {test_case['description']}",
            "query": query, "expected": expected_response, "response": response,
            "cited_reference": reference, "score": evaluation["score"],
            "reasoning": evaluation["reasoning"], "verdict": evaluation["verdict"],
            "chat_model": model_name, "judge_model": settings.JUDGE_MODEL,
        })
    assert evaluation["score"] == 1, (
        f"{test_case['description']}: judge scored {evaluation['score']} — "
        f"{evaluation['reasoning']}"
    )
