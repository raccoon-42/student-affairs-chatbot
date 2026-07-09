"""Chunk builders tested offline — pure JSON-to-chunks logic, no Qdrant."""
import json

from preprocessing.indexing.vectorizer import people_chunks_from_dir, program_chunks_from_json


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


def test_program_full_list_chunk_per_level_enables_absence_answers(tmp_path):
    chunks = program_chunks_from_json(make_programs(tmp_path))
    lists = [c for c in chunks if "tam listesi" in c["text"]]
    assert len(lists) == 2  # lisans + doktora; no yükseklisans entries -> no list
    lisans = next(c for c in lists if c["metadata"]["level"] == "lisans")
    assert "Bilgisayar Mühendisliği" in lisans["text"] and "Fizik" in lisans["text"]
    # the completeness sentence is what grounds "İYTE'de X yok" answers
    assert "olmayan" in lisans["text"]
