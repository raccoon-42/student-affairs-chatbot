"""The judge's parsing logic, tested offline with a fake LLM."""
from tests.evaluators.llm_judge import LLMJudge, cited_reference, parse_verdict


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
    verdict = parse_verdict("Değerlendirme:\nMantık yürütme: Tarih yanlış verilmiş.\nPuan: 0 \nSon.")
    assert verdict["score"] == 0
    assert verdict["reasoning"] == "Tarih yanlış verilmiş."


def test_reasoning_over_several_lines_is_kept_whole():
    # the one-line capture showed only praise for a case that scored 0
    verdict = parse_verdict("Mantık yürütme: Tarihler doğru [2].\nAncak bir cümlede kaynak yok.\nPuan: 0")
    assert verdict["score"] == 0
    assert verdict["reasoning"] == "Tarihler doğru [2].\nAncak bir cümlede kaynak yok."


def test_unparseable_verdict_returns_none_score():
    assert parse_verdict("hiçbir format yok")["score"] is None


def test_judge_sees_only_the_chunks_the_answer_cites():
    entries = ["[1] Final sınavları 11-22 Ocak 2027", "[2] Yılbaşı tatili 1 Ocak 2027",
               "[3] CENG 322 — 4 kredi, 8 AKTS", "[12] Bahar sınavları 14-25 Haziran 2027"]
    reference = cited_reference("Finaller 11-22 Ocak'ta [1][12]. Krediyi sormadın.", entries)
    assert reference == "[1] Final sınavları 11-22 Ocak 2027\n\n[12] Bahar sınavları 14-25 Haziran 2027"
    assert cited_reference("Hangi dönemi soruyorsun?", entries) == ""


def test_an_answer_citing_nothing_is_flagged_to_the_judge():
    llm = FakeLLM("Puan: 0\nMantık yürütme: kaynaksız")
    LLMJudge(llm, "any-model").evaluate_response("kaç kredi?", "4 kredi", "4 kredi")
    assert "hiçbir referans maddesine kaynak göstermiyor" in llm.prompt

