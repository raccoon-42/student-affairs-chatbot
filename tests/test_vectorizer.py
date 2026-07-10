"""Chunk builders tested offline — pure JSON-to-chunks logic, no Qdrant."""
import json
from datetime import date

from preprocessing.indexing.vectorizer import (
    _page_corpus_chunks, courses_chunks_from_dir, people_chunks_from_dir,
    program_chunks_from_json, plan_sync, point_id)


def make_programs(tmp_path):
    programs = [
        {"title": "Bilgisayar Mühendisliği", "level": "lisans",
         "faculty": "Mühendislik Fakültesi",
         "source_url": "http://ceng.iyte.edu.tr/", "page_url": "https://iyte.edu.tr/lisans/",
         "description": "Yazılım ve donanım çalışır.", "aliases": ["ceng", "computer engineering"]},
        {"title": "Fizik", "level": "lisans", "faculty": "Fen Fakültesi",
         "source_url": "http://physics.iyte.edu.tr/", "page_url": "https://iyte.edu.tr/lisans/",
         "description": "Fizik çalışır.", "aliases": ["physics"]},
        {"title": "Fizik", "level": "doktora", "faculty": None,
         "source_url": "http://physics.iyte.edu.tr/", "page_url": "https://lee.iyte.edu.tr/",
         "description": "Fizik doktorası.", "aliases": ["physics phd"]},
    ]
    path = tmp_path / "programs.json"
    path.write_text(json.dumps(programs, ensure_ascii=False), encoding="utf-8")
    return path


def test_program_chunks_carry_aliases_and_metadata(tmp_path):
    chunks = program_chunks_from_json(make_programs(tmp_path))
    ceng = next(c for c in chunks if "Bilgisayar" in c["text"])
    assert "ceng" in ceng["text"]  # aliases feed both BM25 and the embedding
    assert ceng["metadata"]["source_url"] == "http://ceng.iyte.edu.tr/"
    assert ceng["metadata"]["level"] == "lisans"


def test_people_chunks_stack_identity_bio_and_contact(tmp_path):
    people = [
        {"name": "Selma Tekir", "title": "Associate Professor", "role": "akademik",
         "department": "Bilgisayar Mühendisliği",
         "contact": {"e-posta": "selmatekir@iyte.edu.tr", "ofis": "014"},
         "bio": "Selma Tekir leads the Data Analytics Research Group.",
         "areas": ["yapay zeka", "doğal dil işleme"],
         "source_url": "https://ceng.iyte.edu.tr/people/selma-tekir/"},
        {"name": "Bengisu Şahin", "title": None, "role": "arastirma-gorevlisi",
         "department": "Bilgisayar Mühendisliği", "contact": {}, "bio": None,
         "source_url": "https://ceng.iyte.edu.tr/people/bengisu-sahin/"},
    ]
    (tmp_path / "ceng.json").write_text(json.dumps(people, ensure_ascii=False), encoding="utf-8")

    chunks = people_chunks_from_dir(tmp_path)

    tekir = next(c for c in chunks if "Tekir" in c["text"] and "tam listesi" not in c["text"])
    assert "Associate Professor, Bilgisayar Mühendisliği" in tekir["text"]
    assert "selmatekir@iyte.edu.tr" in tekir["text"]  # contact must be retrievable
    assert tekir["metadata"]["department"] == "Bilgisayar Mühendisliği"
    sahin = next(c for c in chunks if "Şahin" in c["text"] and "tam listesi" not in c["text"])
    assert "Araştırma Görevlisi" in sahin["text"]  # role fills in for a missing title

    # "hocalar kimler" needs the whole roster in one chunk, like programs
    roster = next(c for c in chunks if "tam listesi" in c["text"])
    assert "Selma Tekir" in roster["text"] and "Bengisu Şahin" in roster["text"]
    assert "Öğretim üyeleri (1)" in roster["text"]
    assert "kayıtlı değildir" in roster["text"]  # grounds absence answers too

    # "AI çalışan hocalar kimler" = filtered enumeration -> per-area chunk
    ai = next(c for c in chunks if c["metadata"].get("area") == "yapay zeka")
    assert "Selma Tekir (Bilgisayar Mühendisliği)" in ai["text"]
    assert "biyografilerindeki bilgiye dayanır" in ai["text"]  # lossy tags: hedge, no absence claim

    # kind partitions person / roster / area chunks for tiered retrieval
    assert tekir["metadata"]["kind"] == "person"
    assert roster["metadata"]["kind"] == "roster" and ai["metadata"]["kind"] == "area"
    # the roster doubles as the group-by-area table: areas ride inline
    assert "Selma Tekir (Associate Professor; yapay zeka, doğal dil işleme)" in roster["text"]


def test_course_chunks_carry_prereqs_and_per_level_full_lists(tmp_path):
    courses = [
        {"code": "CENG 311", "name": "Computer Architecture",
         "description": "Basic computer organization concepts. Pipelining.",
         "prerequisites": "CENG 214", "levels": ["lisans"],
         "department": "Bilgisayar Mühendisliği",
         "source_url": "https://ceng.iyte.edu.tr/courses/ceng-311/"},
        {"code": "CENG 524", "name": "Advanced Computer Architecture",
         "description": "Advanced topics.", "prerequisites": None,
         "levels": ["yukseklisans", "doktora"],
         "department": "Bilgisayar Mühendisliği",
         "source_url": "https://ceng.iyte.edu.tr/courses/ceng-524/"},
    ]
    (tmp_path / "ceng.json").write_text(json.dumps(courses, ensure_ascii=False), encoding="utf-8")

    chunks = courses_chunks_from_dir(tmp_path)

    c311 = next(c for c in chunks if c["metadata"]["code"] == "CENG 311")
    assert "Önkoşul: CENG 214" in c311["text"]
    assert c311["metadata"]["kind"] == "course"
    assert c311["metadata"]["source_url"].endswith("ceng-311/")
    c524 = next(c for c in chunks if c["metadata"]["code"] == "CENG 524")
    assert "yüksek lisans, doktora dersi" in c524["text"]
    assert "Önkoşul" not in c524["text"]

    # absence/enumeration questions need the complete per-level catalog
    lists = [c for c in chunks if c["metadata"]["kind"] == "list"]
    assert len(lists) == 3  # lisans, yükseklisans, doktora
    lisans = next(c for c in lists if c["metadata"]["levels"] == ["lisans"])
    assert "CENG 311" in lisans["text"] and "CENG 524" not in lisans["text"]
    assert "olmayan" in lisans["text"]  # the completeness sentence


def test_page_corpus_link_entries_become_chunks_with_the_pdf_as_source(tmp_path):
    manifest = [
        {"title": "Ders Seçimi Bilgilendirme", "topic": "ders-secimi",
         "source_url": "https://x/ders-secimi/", "file": "ders-secimi.html"},
        {"title": "Ders Kayıtlanma Adımları (PDF)", "topic": "ders-secimi",
         "source_url": "https://x/uploads/adimlar.pdf", "kind": "link",
         "description": "Ders kayıt adımlarını özetleyen belge."},
    ]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False),
                                            encoding="utf-8")
    (tmp_path / "ders-secimi.json").write_text(json.dumps(
        [{"baslik": "İşlem Adımları", "metin": "1. katkı payı 2. seçim 3. onay"}],
        ensure_ascii=False), encoding="utf-8")

    chunks = _page_corpus_chunks(tmp_path, tmp_path / "manifest.json",
                                 lambda c: f"{c['baslik']}\n{c['metin']}", "rehber")

    link = next(c for c in chunks if "PDF" in c["metadata"]["document_title"])
    assert link["metadata"]["source_url"].endswith(".pdf")  # the chip links the PDF
    assert "kayıt adımlarını" in link["text"]  # description is the retrieval text
    page = next(c for c in chunks if c["metadata"]["section"] == "İşlem Adımları")
    assert "katkı payı" in page["text"]


def test_point_id_is_stable_for_equal_payloads_and_changes_with_content():
    payload = {"text": "MADDE 5 ...", "metadata": {"article": "5", "source_url": "http://x"}}
    same_other_order = {"metadata": {"source_url": "http://x", "article": "5"}, "text": "MADDE 5 ..."}
    assert point_id(payload) == point_id(same_other_order)  # key order must not matter
    assert point_id(payload) != point_id({**payload, "text": "MADDE 5 değişti"})
    # calendar payloads carry date objects inside metadata; ids must not crash on them
    assert point_id({"text": "kayıt", "metadata": {"parsed_date1": date(2026, 9, 14)}})


def test_plan_sync_diffs_desired_against_existing():
    new, stale = plan_sync({"a", "b", "c"}, {"b", "c", "d"})
    assert new == ["a"]  # only the new chunk gets embedded
    assert stale == ["d"]  # the vanished chunk gets deleted
    assert plan_sync({"a"}, {"a"}) == ([], [])  # unchanged corpus: no work


def test_program_full_list_chunk_per_level_enables_absence_answers(tmp_path):
    chunks = program_chunks_from_json(make_programs(tmp_path))
    lists = [c for c in chunks if "tam listesi" in c["text"]]
    assert len(lists) == 2  # lisans + doktora; no yükseklisans entries -> no list
    lisans = next(c for c in lists if c["metadata"]["level"] == "lisans")
    assert "Bilgisayar Mühendisliği" in lisans["text"] and "Fizik" in lisans["text"]
    # the completeness sentence is what grounds "İYTE'de X yok" answers
    assert "olmayan" in lisans["text"]
