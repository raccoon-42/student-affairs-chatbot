"""The FAQ line parser, tested on text shaped like the real sources."""
from preprocessing.scrapers.faq_scraper import parse_faq_lines, _is_question

SAMPLE_LINES = [
    "LİSANS-SIKÇA SORULAN SORULAR",
    "ÖĞRENCİ KİMLİK KARTI",
    "Öğrenci Kimlik kartım kaybolursa ne yapmalıyım?",
    "Zayi dilekçesini doldurup imzaladıktan sonra teslim edin,",
    "₺50 ücreti IBAN'a yatırmanız gerekmektedir.",
    "",
    "BELGE TALEPLERİ",
    "Öğrenci Belgesi nasıl alınır?",
    "e-Devlet üzerinden alınabilir.",
]


def test_parses_questions_answers_and_categories():
    faqs = parse_faq_lines(SAMPLE_LINES, "lisans", "http://example.edu")

    assert len(faqs) == 2
    assert faqs[0]["question"] == "Öğrenci Kimlik kartım kaybolursa ne yapmalıyım?"
    assert faqs[0]["answer"] == (
        "Zayi dilekçesini doldurup imzaladıktan sonra teslim edin, "
        "₺50 ücreti IBAN'a yatırmanız gerekmektedir."
    )
    assert faqs[0]["category"] == "ÖĞRENCİ KİMLİK KARTI"
    assert faqs[1]["category"] == "BELGE TALEPLERİ"
    assert all(f["audience"] == "lisans" for f in faqs)


def test_intro_lines_before_first_question_are_dropped():
    lines = ["Hoş geldiniz, bu sayfada sorular var.", "Soru bir nedir?", "Cevap bir."]
    faqs = parse_faq_lines(lines, "lisans", "url")
    assert len(faqs) == 1
    assert faqs[0]["question"] == "Soru bir nedir?"


def test_question_without_answer_is_skipped():
    lines = ["Cevapsız soru?", "BAŞLIK", "Cevaplı soru?", "Cevap."]
    faqs = parse_faq_lines(lines, "lisans", "url")
    assert [f["question"] for f in faqs] == ["Cevaplı soru?"]


def test_question_with_trailing_parenthetical():
    # real case from the YDYO page that a plain endswith('?') missed
    assert _is_question("İngilizce Yeterlik Sınavı nasıl yapılır? (Eylül SBS)")
    assert _is_question("Sınava kimler katılabilirler? (Akademik takvime göre ay değişikliği olabilir**)")
    assert not _is_question("₺50 ücreti IBAN'a yatırmanız gerekmektedir.")
    assert not _is_question("Detaylar? için web sayfasına bakınız buradan devam eder")


def test_turkish_all_caps_headers_detected():
    faqs = parse_faq_lines(["ÇİFT ANADAL / YANDAL", "Soru mu?", "Cevap."], "lisans", "url")
    assert faqs[0]["category"] == "ÇİFT ANADAL / YANDAL"
