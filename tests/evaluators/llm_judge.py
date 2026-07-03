import re
from typing import Any, Dict

JUDGE_PROMPT_TEMPLATE = """Sen bir akademik takvim asistanı chatbot'unun cevaplarını değerlendiren bir uzman değerlendiricisin.

Değerlendirme Kuralları:
1. Selamlamaları ve ek bilgileri göz ardı et
2. Sadece sorulan soruya verilen cevabın doğruluğuna odaklan
3. ÖNEMLİ: Eğer soru belirsiz ise (örn. hangi dönem olduğu belli değilse) ve chatbot bu belirsizliği gidermek için soru soruyorsa, bu yanıtı DOĞRU kabul et
4. Chatbot hem belirsizliği gidermek için soru sorup hem de genel bilgi veriyorsa, bu özellikle takdir edilmeli

Örnek Değerlendirmeler:

Örnek 1:
Soru: "okul ne zaman kapanıyor?"
Beklenen Cevap: "hangi dönem için sorduğunu sorar ya da sınavların ne zaman bittiğini söyler."
Verilen Cevap: "Hangi dönemi kastettiğinizi öğrenebilir miyim? Güz ve bahar dönemlerinin kapanış tarihleri farklıdır."
Puan: 1
Mantık yürütme: Belirsizliği gidermek için soru sorulmuş ve ek bilgi de verilmiş.

Örnek 2:
Soru: "dersten çekilmek için son tarih nedir?"
Beklenen Cevap: "hangi dönem çekilmek istediği sorusunu sorar ya da 25 nisan 2025 diye söyler."
Verilen Cevap: "Hangi dönem için çekilmek istediğinizi belirtir misiniz? Bahar dönemi için son tarih 25 Nisan 2025'tir."
Puan: 1
Mantık yürütme: Hem dönem belirsizliğini gidermek için soru sorulmuş hem de yararlı bilgi verilmiş.

Şimdi şu yanıtı değerlendir:
Sorulan soru: {query}
Beklenen cevap: {expected_response}
Verilen cevap: {response}

Aşağıdaki formatta cevap ver:
Puan: (0 veya 1)
Mantık yürütme: (Puanlamanın nedeni)
"""


class LLMJudge:
    """Scores a chatbot answer against an expected answer.

    Takes any LLM adapter with chat(model, messages) -> str: OpenRouter
    (the default for evaluation), Ollama, or a fake in unit tests.
    """

    def __init__(self, llm, model_name: str):
        self._llm = llm
        self._model_name = model_name

    def evaluate_response(self, query: str, response: str, expected_response: str) -> Dict[str, Any]:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            query=query, expected_response=expected_response, response=response
        )
        verdict = self._llm.chat(self._model_name, [{"role": "user", "content": prompt}])

        return {
            **parse_verdict(verdict),
            "query": query,
            "response": response,
            "expected_response": expected_response,
        }


def parse_verdict(text: str) -> Dict[str, Any]:
    """Extract score and reasoning from the judge's free-text verdict."""
    score_match = re.search(r'Puan:\s*(\d)', text)
    reasoning_match = re.search(r'Mantık yürütme:\s*(.*)', text)
    return {
        "score": int(score_match.group(1)) if score_match else None,
        "reasoning": reasoning_match.group(1).strip() if reasoning_match else "",
    }
