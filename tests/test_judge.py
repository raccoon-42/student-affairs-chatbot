"""The judge's parsing logic, tested offline with a fake LLM."""
from tests.evaluators.llm_judge import LLMJudge, parse_verdict


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    def chat(self, model, messages):
        self.prompt = messages[-1]["content"]
        return self.reply


def test_parses_score_and_reasoning():
    judge = LLMJudge(FakeLLM("Puan: 1\nMantık yürütme: Cevap doğru."), "any-model")
    result = judge.evaluate_response("soru", "cevap", "beklenen")
    assert result["score"] == 1
    assert result["reasoning"] == "Cevap doğru."


def test_prompt_contains_the_three_inputs():
    llm = FakeLLM("Puan: 0\nMantık yürütme: yanlış")
    LLMJudge(llm, "any-model").evaluate_response("soru?", "verilen", "beklenen")
    assert "soru?" in llm.prompt
    assert "verilen" in llm.prompt
    assert "beklenen" in llm.prompt


def test_verdict_with_extra_prose_still_parses():
    verdict = parse_verdict("Değerlendirme:\nPuan: 0 \nMantık yürütme: Tarih yanlış verilmiş.\nSon.")
    assert verdict["score"] == 0
    assert verdict["reasoning"] == "Tarih yanlış verilmiş."


def test_unparseable_verdict_returns_none_score():
    assert parse_verdict("hiçbir format yok")["score"] is None
