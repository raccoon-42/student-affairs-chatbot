"""The link check: addresses no chunk contains are removed from the answer."""
from app.conversation import Conversation
from app.links import check_links
from tests.test_conversation import FakeLLM, FakeRetriever

TURNA = ("2026 Erasmus+ KA131 Öğrenim Hareketliliği Başvuruları\n"
         "1. 17 Mart 2026 tarihinden itibaren TurnaPortal'a girin (bağlantı: http://turnaportal.ua.gov.tr/).")


def test_an_invented_address_written_as_its_own_link_text_goes_entirely():
    # a judge failure (makine dersleri): no chunk had this address
    answer = ("Bölümün kendisine veya Öğrenci İşleri Daire Başkanlığına "
              "([https://oidb.iyte.edu.tr](https://oidb.iyte.edu.tr)) başvurabilirsin.")
    cleaned, removed = check_links(answer, [TURNA])
    assert cleaned == "Bölümün kendisine veya Öğrenci İşleri Daire Başkanlığına başvurabilirsin."
    assert removed == ["https://oidb.iyte.edu.tr"]


def test_an_invented_link_keeps_its_text():
    # a judge failure (yaz okulu): the homepage was in no chunk
    answer = "Ayrıntı için [Öğrenci İşleri Daire Başkanlığı](https://www.iyte.edu.tr) ile iletişime geç."
    cleaned, removed = check_links(answer, [TURNA])
    assert cleaned == "Ayrıntı için Öğrenci İşleri Daire Başkanlığı ile iletişime geç."
    assert removed == ["https://www.iyte.edu.tr"]


def test_a_bare_invented_address_goes_and_the_sentence_keeps_its_stop():
    cleaned, removed = check_links("Güncel liste https://www.iyte.edu.tr.", [TURNA])
    assert cleaned == "Güncel liste."
    assert removed == ["https://www.iyte.edu.tr"]


def test_an_address_from_a_chunk_is_kept():
    # the chunk writes it with a trailing slash, the answer without
    answer = "1. [TurnaPortal](http://turnaportal.ua.gov.tr)'a girin [3]. Ya da http://turnaportal.ua.gov.tr/ adresine git."
    assert check_links(answer, [TURNA]) == (answer, [])


def test_the_root_of_a_longer_link_is_not_backed_by_it():
    # a judge run: the chunk had "https://uio.iyte.edu.tr/wp-content/...pdf",
    # and the answer's invented "https://uio.iyte.edu.tr/" matched it as text
    guide = "Online Öğrenim Başvuru Formu Rehberi (bağlantı: https://uio.iyte.edu.tr/wp-content/uploads/rehber.pdf)"
    answer = "Ayrıntılar için [UİO](https://uio.iyte.edu.tr/) sayfasına bak. Rehber: https://uio.iyte.edu.tr/wp-content/uploads/rehber.pdf"
    cleaned, removed = check_links(answer, [guide])
    assert cleaned == "Ayrıntılar için UİO sayfasına bak. Rehber: https://uio.iyte.edu.tr/wp-content/uploads/rehber.pdf"
    assert removed == ["https://uio.iyte.edu.tr/"]


def test_a_bare_address_written_in_a_chunk_backs_the_same_in_the_answer():
    chunk = "Başvurular obs.iyte.edu.tr adresinden yapılır."
    answer = "Başvuruyu obs.iyte.edu.tr üzerinden yapabilirsin [1]."
    assert check_links(answer, [chunk]) == (answer, [])


def test_an_address_without_its_scheme_is_caught_too():
    # a judge run: "(oidb.iyte.edu.tr)" got past a check for https:// only
    answer = "Bölüme veya Öğrenci İşleri Daire Başkanlığı'na (oidb.iyte.edu.tr) başvurun."
    cleaned, removed = check_links(answer, [TURNA])
    assert cleaned == "Bölüme veya Öğrenci İşleri Daire Başkanlığı'na başvurun."
    assert removed == ["oidb.iyte.edu.tr"]


def test_a_link_to_no_address_keeps_its_text():
    # a judge run: "[Mezuniyet Durumu Formu](mezuniyet-durumu-formu)"
    answer = "İşlemi başlatmak için [Mezuniyet Durumu Formu](mezuniyet-durumu-formu) doldurulur [4]."
    cleaned, removed = check_links(answer, [TURNA])
    assert cleaned == "İşlemi başlatmak için Mezuniyet Durumu Formu doldurulur [4]."
    assert removed == ["mezuniyet-durumu-formu"]


def test_a_link_target_with_spaces_keeps_its_text_and_markers_stay():
    # a judge run: "[Öğrenci Ders Kayıt Kılavuzu (PDF)](Öğrenci Ders Kayıt Kılavuzu)"
    answer = "Kılavuz: [Öğrenci Ders Kayıt Kılavuzu (PDF)](Öğrenci Ders Kayıt Kılavuzu) [38]. Ekler [38](PDF)."
    cleaned, removed = check_links(answer, [TURNA])
    assert cleaned == "Kılavuz: Öğrenci Ders Kayıt Kılavuzu (PDF) [38]. Ekler [38](PDF)."
    assert removed == ["Öğrenci Ders Kayıt Kılavuzu"]


def test_emails_names_and_numbers_are_not_addresses():
    answer = ("Koordinatör: Dr. Emir Mobedi (emirmobedi@iyte.edu.tr), not ortalaması 2.20, "
              "rapor.pdf ve turnaportal.ua.gov.tr [3].")
    assert check_links(answer, [TURNA]) == (answer, [])


def test_an_answer_without_links_is_untouched():
    answer = "- **Başvuru:**  TurnaPortal üzerinden [3].\n  - alt madde ."
    assert check_links(answer, [TURNA]) == (answer, [])


def test_respond_returns_the_answer_without_the_invented_link():
    reply = "Kimlik kartı kaybolursa [Öğrenci İşleri](https://oidb.iyte.edu.tr) dilekçe ister [3]."
    conversation = Conversation(FakeLLM(reply=reply), FakeRetriever(), model="m")

    answer = conversation.respond("kimlik kaybettim")

    assert answer == "Kimlik kartı kaybolursa Öğrenci İşleri dilekçe ister [3]."
    assert conversation.last_answer == answer
    assert any(line.startswith("links") and "oidb" in line for line in conversation.last_debug)
