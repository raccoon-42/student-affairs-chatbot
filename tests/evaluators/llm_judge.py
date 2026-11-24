import re
from typing import Any, Dict, List

from app.llm import chat_with_usage

# what a run spent, summed across cases; conftest prints it at the end
RUN_COSTS = {"cases": 0, "judge": 0.0, "chatbot": 0.0}

JUDGE_PROMPT_TEMPLATE = """Sen bir akademik takvim asistanı chatbot'unun cevaplarını değerlendiren bir uzman değerlendiricisin.

Değerlendirme Kuralları:
1. Selamlamaları ve ek bilgileri göz ardı et
2. Sadece sorulan soruya verilen cevabın doğruluğuna odaklan
3. ÖNEMLİ: Eğer soru belirsiz ise (örn. hangi dönem olduğu belli değilse) ve chatbot bu belirsizliği gidermek için soru soruyorsa, bu yanıtı DOĞRU kabul et
4. Chatbot hem belirsizliği gidermek için soru sorup hem de genel bilgi veriyorsa, bu özellikle takdir edilmeli
5. Chatbot her bilginin sonuna kaynağını [n] işaretiyle koymak zorundadır. Referans veri bölümünde yalnızca cevabın işaretlediği maddeler vardır. Cevaptaki somut bilgileri (tarih, sayı, kredi, oda numarası, telefon, isim) bu maddelere karşı DOĞRULA: bilgi işaretlenen maddede geçiyorsa doğru say; işaretlenen maddelerde geçmiyorsa ya da somut bilginin olduğu cümlede hiç işaret yoksa desteksiz say ve puanı 0 ver. Bilgi tabanından gelmeyen bilgiler işaret gerektirmez: bugünün tarihi (chatbot'a sistemden verilir), öğrencinin kendi sorusunda geçenler ve somut bilgi içermeyen cevaplar (netleştirme sorusu, "elimdeki verilerde yok"). Referans veriyi görebildiğin için "doğrulayamıyorum" deme.

Örnek Değerlendirmeler:

Örnek 1:
Soru: "okul ne zaman kapanıyor?"
Beklenen Cevap: "hangi dönem için sorduğunu sorar ya da sınavların ne zaman bittiğini söyler."
Verilen Cevap: "Hangi dönemi kastettiğinizi öğrenebilir miyim? Güz ve bahar dönemlerinin kapanış tarihleri farklıdır."
Mantık yürütme: Belirsizliği gidermek için soru sorulmuş ve ek bilgi de verilmiş.
Puan: 1

Örnek 2:
Soru: "dersten çekilmek için son tarih nedir?"
Beklenen Cevap: "hangi dönem çekilmek istediği sorusunu sorar ya da 25 nisan 2025 diye söyler."
Verilen Cevap: "Hangi dönem için çekilmek istediğinizi belirtir misiniz? Bahar dönemi için son tarih 25 Nisan 2025'tir."
Mantık yürütme: Hem dönem belirsizliğini gidermek için soru sorulmuş hem de yararlı bilgi verilmiş.
Puan: 1

Şimdi şu yanıtı değerlendir:
Sorulan soru: {query}
Beklenen cevap: {expected_response}
Verilen cevap: {response}
{reference_block}

Aşağıdaki formatta cevap ver; önce gerekçeyi yaz, puanı en sona koy:
Mantık yürütme: (değerlendirmen)
Puan: (0 veya 1)
"""


class LLMJudge:
    """Scores a chatbot answer against an expected answer.

    Takes any LLM adapter with chat(model, messages) -> str: OpenRouter
    (the default for evaluation) or a fake in unit tests.
    """

    def __init__(self, llm, model_name: str):
        self._llm = llm
        self._model_name = model_name

    def evaluate_response(self, query: str, response: str, expected_response: str,
                          reference: str = "") -> Dict[str, Any]:
        reference_block = (
            f"\nCevabın kaynak gösterdiği referans maddeleri:\n{reference}\n"
            if reference else "\nCevap hiçbir referans maddesine kaynak göstermiyor.\n"
        )
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            query=query, expected_response=expected_response, response=response,
            reference_block=reference_block,
        )
        verdict, usage = chat_with_usage(self._llm, self._model_name,
                                         [{"role": "user", "content": prompt}])

        return {
            **parse_verdict(verdict),
            "query": query,
            "response": response,
            "expected_response": expected_response,
            "usage": usage,
        }


def cited_reference(response: str, entries: List[str]) -> str:
    """The reference entries ("[n] chunk text") the answer cites with [n]
    markers. The judge needs only these: the full block (~38 chunks, ~15k
    tokens) was most of a judge run's cost, and an uncited fact fails
    anyway."""
    cited = set(re.findall(r"\[(\d+)\]", response))
    return "\n\n".join(entry for entry in entries
                       if (m := re.match(r"\[(\d+)\]", entry)) and m.group(1) in cited)


def parse_verdict(text: str) -> Dict[str, Any]:
    """Extract score and reasoning from the judge's free-text verdict. The
    reasoning comes first and runs until the score line, over several lines
    if need be (a one-line capture hid why a case scored 0); the last score
    line wins."""
    scores = re.findall(r'Puan:\s*(\d)', text)
    reasoning_match = re.search(r'Mantık yürütme:\s*(.*?)\s*(?:\n\s*Puan:|\Z)', text, re.DOTALL)
    return {
        "score": int(scores[-1]) if scores else None,
        "reasoning": reasoning_match.group(1).strip() if reasoning_match else "",
    }
