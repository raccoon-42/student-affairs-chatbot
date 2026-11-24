"""The retrieval eval's scoring, offline: label matching, hit@k at each
collection's production top_k, and offline RRF."""
from scripts.eval_retrieval import TOP_K, matches, rrf, score


def test_label_matches_text_and_metadata():
    text = "CENG 322 — Operating Systems (Bilgisayar Mühendisliği, lisans dersi)"
    assert matches({"contains": ["operating systems"]}, text, {})  # case-insensitive
    assert matches({"metadata": {"code": "CENG 322"}}, text, {"code": "CENG 322"})
    assert not matches({"contains": ["Operating Systems"], "metadata": {"code": "CENG 311"}},
                       text, {"code": "CENG 322"})


def test_hit_counts_only_within_the_collections_production_top_k():
    case = {"relevant": [{"collection": "forms", "contains": ["Dersten Çekilme"]}]}
    others = [("başka form", {})] * TOP_K["forms"]
    assert score(case, {"forms": [("Dersten Çekilme Talep Formu", {})] + others})["hit@1"]
    late = score(case, {"forms": others + [("Dersten Çekilme Talep Formu", {})]})
    assert not late["hit@k"] and late["rr"] == 1 / (TOP_K["forms"] + 1)
    assert score(case, {})["rr"] == 0.0


def test_any_relevant_chunk_is_a_hit():
    case = {"relevant": [{"collection": "programs", "contains": ["Bilgisayar Mühendisliği (Lisans"]},
                         {"collection": "programs", "contains": ["tam listesi"]}]}
    result = score(case, {"programs": [("İYTE lisans programlarının tam listesi", {})]})
    assert result["hit@1"] and result["recall@k"] == 0.5


def test_rrf_rewards_agreement_between_rankings():
    dense = [("a", "A", {}), ("b", "B", {}), ("c", "C", {})]
    sparse = [("c", "C", {}), ("b", "B", {})]
    # A: 1/2; B: 1/3 + 1/3; C: 1/4 + 1/2 — dense's #1 loses to chunks both rank
    assert [text for text, _ in rrf([dense, sparse], k=2)] == ["C", "B", "A"]
