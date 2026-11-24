"""The guide-question step's file handling, with the model faked."""
import json

from preprocessing.parsers import guide_questions_llm as step

PAGE = [{"baslik": "Başvuru Tarihleri", "metin": "17 Mart - 2 Nisan arası."},
        {"baslik": "Asgari Şartlar", "metin": "Lisans için en az 2.20 ortalama."}]


def test_questions_land_in_their_own_chunk(tmp_path, monkeypatch):
    path = tmp_path / "page.json"
    path.write_text(json.dumps(PAGE, ensure_ascii=False), encoding="utf-8")
    sent = []

    def fake(prompt):
        sent.append(prompt)
        # out of order, and padded: matched by number, trimmed
        return [{"parca": 2, "sorular": ["erasmus icin kac ortalama lazim "]},
                {"parca": 1, "sorular": ["erasmus başvuruları ne zaman"]}]

    monkeypatch.setattr(step, "json_array", fake)
    step.add_questions(path)

    chunks = json.loads(path.read_text(encoding="utf-8"))
    assert chunks[0]["sorular"] == ["erasmus başvuruları ne zaman"]
    assert chunks[1]["sorular"] == ["erasmus icin kac ortalama lazim"]
    # the model sees the whole page, so it can tell siblings apart
    assert "[1] Başvuru Tarihleri" in sent[0] and "[2] Asgari Şartlar" in sent[0]
    assert not step.needs_questions(chunks)


def test_a_chunk_the_model_skipped_is_asked_for_again(tmp_path, monkeypatch):
    path = tmp_path / "page.json"
    path.write_text(json.dumps(PAGE, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(step, "json_array", lambda prompt: [{"parca": 1, "sorular": ["ne zaman"]}])

    step.add_questions(path)

    chunks = json.loads(path.read_text(encoding="utf-8"))
    assert chunks[0]["sorular"] == ["ne zaman"]
    assert "sorular" not in chunks[1]
    assert step.needs_questions(chunks)  # --all picks the page up again
