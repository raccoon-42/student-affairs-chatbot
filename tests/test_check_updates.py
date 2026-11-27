"""index_state.json across runs, with the corpus syncs faked: no scraping,
no parsing, no indexing."""
import json
import sys
from pathlib import Path

import pytest

from preprocessing import check_updates


@pytest.fixture
def state(tmp_path, monkeypatch):
    path = tmp_path / "index_state.json"
    monkeypatch.setattr(check_updates, "STATE_PATH", path)
    return path


def run(monkeypatch, syncs, *corpora):
    monkeypatch.setattr(check_updates, "SYNCS", syncs)
    monkeypatch.setattr(sys, "argv", ["check_updates", *corpora])
    check_updates.main()


def test_a_corrupt_state_file_re_baselines_instead_of_crashing(state, monkeypatch, capsys):
    # a run stopped mid-write left half a file, and every later run died on it
    state.write_text('{"faq": {"faq.json": "ab', encoding="utf-8")
    seen = []
    run(monkeypatch, {"faq": lambda old, dry_run: seen.append(old) or {"faq.json": "h1"}})
    assert seen == [{}]  # no recorded state: the baseline path
    assert json.loads(state.read_text(encoding="utf-8")) == {"faq": {"faq.json": "h1"}}
    assert "index_state.json is corrupt" in capsys.readouterr().err


def test_a_run_keeps_what_a_parallel_run_recorded(state, monkeypatch):
    state.write_text(json.dumps({"faq": {"faq.json": "h0"}, "takvim": {"m": "t0"}}), encoding="utf-8")

    def sync_takvim(old, dry_run):
        # meanwhile another run re-indexed faq and recorded it
        state.write_text(json.dumps({"faq": {"faq.json": "h1"}, "takvim": {"m": "t0"}}), encoding="utf-8")
        return {"m": "t1"}

    run(monkeypatch, {"takvim": sync_takvim}, "takvim")
    assert json.loads(state.read_text(encoding="utf-8")) == {"faq": {"faq.json": "h1"},
                                                             "takvim": {"m": "t1"}}


def test_a_run_stopped_mid_write_leaves_the_old_state_whole(state, monkeypatch):
    class Stopped(BaseException):  # a container stop, not an error main() handles
        pass

    old = {"faq": {"faq.json": "h0"}}
    state.write_text(json.dumps(old), encoding="utf-8")
    real_write = Path.write_text

    def stopped_mid_write(self, text, *args, **kwargs):
        real_write(self, text[:len(text) // 2], *args, **kwargs)
        raise Stopped

    monkeypatch.setattr(Path, "write_text", stopped_mid_write)
    with pytest.raises(Stopped):
        run(monkeypatch, {"faq": lambda old, dry_run: {"faq.json": "h1"}})
    assert json.loads(state.read_text(encoding="utf-8")) == old
