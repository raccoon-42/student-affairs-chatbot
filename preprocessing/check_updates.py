"""Change check: re-scrape each corpus, hash the artifacts, and run the
expensive steps (LLM parsing, embedding) only for what actually changed.

Scraping is cheap and always runs; the hash diff decides everything else.
State lives in preprocessing/data/index_state.json, one hash map per
corpus, written only after that corpus indexed successfully — a failed run
is retried whole next time. On the first run for a corpus the current
files are recorded as the baseline WITHOUT scraping, so the pipeline isn't
re-run for data that is already indexed. The vectorizer itself is
incremental (point id = content hash), so "reindex" embeds only the
chunks that changed.

Usage:
    uv run python -m preprocessing.check_updates             # all corpora
    uv run python -m preprocessing.check_updates faq sks     # a subset
    uv run python -m preprocessing.check_updates --dry-run   # scrape + report, change nothing

Meant for a daily cron once trusted:
    0 6 * * * cd <repo> && uv run python -m preprocessing.check_updates
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from config import settings

DATA = settings.ROOT / "preprocessing" / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
STATE_PATH = DATA / "index_state.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_module(module, *args):
    command = [sys.executable, "-m", module, *[str(arg) for arg in args]]
    print(f"  $ python -m {module} " + " ".join(str(arg) for arg in args))
    subprocess.run(command, check=True, cwd=settings.ROOT)


def reindex(content_type, path):
    run_module("preprocessing.indexing.vectorizer", path, "--type", content_type)


def diff(old, new):
    changed = sorted(key for key, value in new.items() if old.get(key) != value)
    removed = sorted(key for key in old if key not in new)
    return changed, removed


def baseline(name, fingerprint, index_args, dry_run, scrape=None):
    """First run for a corpus: trust the files on disk, don't re-scrape.

    Exception: on a fresh clone the raw PDFs are missing (gitignored,
    only their manifests are tracked) — scrape once to restore them.
    The processed output from git stays authoritative either way."""
    if dry_run:
        print(f"[{name}] no recorded state — run once without --dry-run to record a baseline")
        return None
    print(f"[{name}] no recorded state — recording current files as the baseline")
    reindex(*index_args)
    try:
        return fingerprint()
    except FileNotFoundError:
        if scrape is None:
            raise
        print(f"[{name}] raw files missing (fresh clone?) — scraping once to restore them")
        scrape()
        return fingerprint()


def report(name, changed, removed):
    print(f"[{name}] changed: {', '.join(changed) or '—'}"
          + (f" | removed: {', '.join(removed)}" if removed else ""))


def delete_processed(directory, stems):
    """Drop stale parser output so the --all/skip-existing parsers redo it."""
    for stem in stems:
        for suffix in (".json", ".txt"):
            path = directory / f"{stem}{suffix}"
            if path.exists():
                path.unlink()
                print(f"  removed stale {path.relative_to(settings.ROOT)}")


def manifest_fingerprint(raw_dir, with_files=True):
    """Hash of the manifest plus (optionally) every file it lists.
    Entries without a file (kind: "link") are covered by the manifest
    hash itself — a changed PDF href shows up there."""
    manifest_path = raw_dir / "manifest.json"
    hashes = {"manifest.json": sha256(manifest_path)}
    if with_files:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        hashes.update({entry["file"]: sha256(raw_dir / entry["file"])
                       for entry in manifest if entry.get("file")})
    return hashes


# --- one sync function per corpus ------------------------------------------
# Each takes (old_hashes, dry_run) and returns the new hashes when the
# corpus was (re)indexed, or None when nothing should be recorded.

def sync_faq(old, dry_run):
    # the FAQ scraper writes processed/faq/faq.json directly — no LLM step
    faq_json = PROCESSED / "faq" / "faq.json"
    fingerprint = lambda: {"faq.json": sha256(faq_json)}
    index_args = ("faq", faq_json)
    if not old:
        return baseline("faq", fingerprint, index_args, dry_run)
    run_module("preprocessing.scrapers.faq_scraper")
    new = fingerprint()
    changed, removed = diff(old, new)
    if not changed and not removed:
        print("[faq] unchanged")
        return None
    report("faq", changed, removed)
    if dry_run:
        print("  would reindex faq (no LLM calls)")
        return None
    reindex(*index_args)
    return new


def sync_mevzuat(old, dry_run):
    raw_dir = RAW / "mevzuat"
    fingerprint = lambda: manifest_fingerprint(raw_dir)
    index_args = ("regulations", PROCESSED / "mevzuat")
    if not old:
        return baseline("mevzuat", fingerprint, index_args, dry_run,
                        scrape=lambda: run_module("preprocessing.scrapers.mevzuat_scraper"))
    run_module("preprocessing.scrapers.mevzuat_scraper")
    new = fingerprint()
    changed, removed = diff(old, new)
    if not changed and not removed:
        print("[mevzuat] unchanged")
        return None
    report("mevzuat", changed, removed)
    changed_pdfs = [key for key in changed if key != "manifest.json"]
    stale = [Path(key).stem for key in changed_pdfs + removed if key != "manifest.json"]
    if dry_run:
        print(f"  would re-parse {len(changed_pdfs)} PDF(s) (~{len(changed_pdfs)} LLM calls) and reindex")
        return None
    delete_processed(PROCESSED / "mevzuat", stale)
    if changed_pdfs:
        run_module("preprocessing.parsers.regulation_parser_llm", "--all")
    reindex(*index_args)
    return new


def sync_takvim(old, dry_run):
    raw_dir = RAW / "takvim"
    fingerprint = lambda: manifest_fingerprint(raw_dir)
    index_args = ("calendar", PROCESSED / "takvim")
    if not old:
        return baseline("takvim", fingerprint, index_args, dry_run,
                        scrape=lambda: run_module("preprocessing.scrapers.takvim_scraper"))
    run_module("preprocessing.scrapers.takvim_scraper")
    new = fingerprint()
    changed, removed = diff(old, new)
    if not changed and not removed:
        print("[takvim] unchanged")
        return None
    report("takvim", changed, removed)
    changed_pdfs = [key for key in changed if key != "manifest.json"]
    stale = [Path(key).stem for key in changed_pdfs + removed if key != "manifest.json"]
    if dry_run:
        print(f"  would re-parse {len(changed_pdfs)} calendar(s) (~{len(changed_pdfs)} LLM calls) and reindex")
        return None
    delete_processed(PROCESSED / "takvim", stale)
    # the calendar parser has no --all: parse whatever lacks its JSON
    manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest:
        stem = Path(entry["file"]).stem
        if not (PROCESSED / "takvim" / f"{stem}.json").exists():
            run_module("preprocessing.parsers.academic_calendar_parser_llm", raw_dir / entry["file"])
    reindex(*index_args)
    return new


def sync_forms(old, dry_run):
    # link catalog: the manifest IS the corpus; the describer is one LLM
    # call for the whole catalog. The scraper refreshes its default page —
    # entries scraped from other page URLs are preserved as-is.
    raw_dir = RAW / "formlar"
    fingerprint = lambda: manifest_fingerprint(raw_dir, with_files=False)
    index_args = ("forms", PROCESSED / "formlar" / "forms.json")
    if not old:
        return baseline("forms", fingerprint, index_args, dry_run)
    run_module("preprocessing.scrapers.forms_scraper")
    new = fingerprint()
    changed, removed = diff(old, new)
    if not changed and not removed:
        print("[forms] unchanged")
        return None
    report("forms", changed, removed)
    if dry_run:
        print("  would re-describe the catalog (1 LLM call) and reindex")
        return None
    run_module("preprocessing.parsers.form_describer_llm")
    reindex(*index_args)
    return new


def sync_programs(old, dry_run):
    raw_dir = RAW / "programlar"
    fingerprint = lambda: manifest_fingerprint(raw_dir, with_files=False)
    index_args = ("programs", PROCESSED / "programlar" / "programs.json")
    if not old:
        return baseline("programs", fingerprint, index_args, dry_run)
    run_module("preprocessing.scrapers.programs_scraper")
    new = fingerprint()
    changed, removed = diff(old, new)
    if not changed and not removed:
        print("[programs] unchanged")
        return None
    report("programs", changed, removed)
    if dry_run:
        print("  would re-describe the catalog (1 LLM call) and reindex")
        return None
    run_module("preprocessing.parsers.program_describer_llm")
    reindex(*index_args)
    return new


def sync_sks(old, dry_run):
    raw_dir = RAW / "sks"
    fingerprint = lambda: manifest_fingerprint(raw_dir)
    index_args = ("sks", PROCESSED / "sks")
    if not old:
        return baseline("sks", fingerprint, index_args, dry_run)
    run_module("preprocessing.scrapers.sks_scraper")
    new = fingerprint()
    changed, removed = diff(old, new)
    if not changed and not removed:
        print("[sks] unchanged")
        return None
    report("sks", changed, removed)
    changed_pages = [key for key in changed if key != "manifest.json"]
    stale = [Path(key).stem for key in changed_pages + removed if key != "manifest.json"]
    if dry_run:
        print(f"  would re-parse {len(changed_pages)} page(s) (~{len(changed_pages)} LLM calls) and reindex")
        return None
    delete_processed(PROCESSED / "sks", stale)
    if changed_pages:
        run_module("preprocessing.parsers.sks_parser_llm", "--all")
    reindex(*index_args)
    return new


def _people_normalized(text):
    """Scraper output, ignoring the LLM-added area tags — comparing the
    re-scrape against the tagged file must not count areas as a change."""
    people = json.loads(text)
    stripped = [{key: value for key, value in person.items() if key != "areas"}
                for person in people]
    return json.dumps(stripped, ensure_ascii=False, sort_keys=True)


def sync_people(old, dry_run):
    # the people scraper writes processed/people/<dept>.json directly and
    # KNOWS NOTHING of the areas the tagger wrote into those files — so
    # snapshot first, re-scrape, and restore any file whose scraped
    # content didn't change, keeping its tags
    people_dir = PROCESSED / "people"
    fingerprint = lambda: {
        path.name: hashlib.sha256(_people_normalized(path.read_text(encoding="utf-8"))
                                  .encode()).hexdigest()
        for path in sorted(people_dir.glob("*.json"))}
    index_args = ("people", people_dir)
    if not old:
        return baseline("people", fingerprint, index_args, dry_run)

    snapshots = {path.name: path.read_text(encoding="utf-8")
                 for path in people_dir.glob("*.json")}
    run_module("preprocessing.scrapers.people_scraper")
    new = fingerprint()
    changed, removed = diff(old, new)

    unchanged = [name for name in snapshots if name not in changed]
    restore = snapshots if dry_run else {name: snapshots[name] for name in unchanged}
    for name, text in restore.items():
        (people_dir / name).write_text(text, encoding="utf-8")

    if not changed and not removed:
        print("[people] unchanged")
        return None
    report("people", changed, removed)
    if dry_run:
        print(f"  would re-tag areas for {len(changed)} department(s) "
              f"({len(changed)} LLM calls) and reindex (files restored)")
        return None
    if changed:
        run_module("preprocessing.parsers.people_areas_llm",
                   *[people_dir / name for name in changed])
    reindex(*index_args)
    return new


def sync_guides(old, dry_run):
    raw_dir = RAW / "rehber"
    fingerprint = lambda: manifest_fingerprint(raw_dir)
    index_args = ("guides", PROCESSED / "rehber")
    if not old:
        return baseline("guides", fingerprint, index_args, dry_run)
    run_module("preprocessing.scrapers.guides_scraper")
    new = fingerprint()
    changed, removed = diff(old, new)
    if not changed and not removed:
        print("[guides] unchanged")
        return None
    report("guides", changed, removed)
    changed_pages = [key for key in changed if key != "manifest.json"]
    stale = [Path(key).stem for key in changed_pages + removed if key != "manifest.json"]
    if dry_run:
        print(f"  would re-parse {len(changed_pages)} page(s) (~{len(changed_pages)} LLM calls) and reindex")
        return None
    delete_processed(PROCESSED / "rehber", stale)
    if changed_pages:
        run_module("preprocessing.parsers.guides_parser_llm", "--all")
    reindex(*index_args)
    return new


def sync_courses(old, dry_run):
    # like people, the scraper writes the processed JSON directly — but
    # with no LLM tags to preserve, so no snapshot/restore dance
    courses_dir = PROCESSED / "courses"
    fingerprint = lambda: {path.name: sha256(path)
                           for path in sorted(courses_dir.glob("*.json"))}
    index_args = ("courses", courses_dir)
    if not old:
        return baseline("courses", fingerprint, index_args, dry_run)
    run_module("preprocessing.scrapers.courses_scraper")
    new = fingerprint()
    changed, removed = diff(old, new)
    if not changed and not removed:
        print("[courses] unchanged")
        return None
    report("courses", changed, removed)
    if dry_run:
        print("  would reindex courses (no LLM calls)")
        return None
    reindex(*index_args)
    return new


SYNCS = {
    "faq": sync_faq,
    "takvim": sync_takvim,
    "mevzuat": sync_mevzuat,
    "forms": sync_forms,
    "sks": sync_sks,
    "programs": sync_programs,
    "people": sync_people,
    "courses": sync_courses,
    "guides": sync_guides,
}


def main():
    parser = argparse.ArgumentParser(
        description="Detect source changes per corpus and re-parse/re-index only what changed")
    parser.add_argument("corpora", nargs="*",
                        help=f"corpora to check (default: all — {', '.join(SYNCS)})")
    parser.add_argument("--dry-run", action="store_true",
                        help="scrape and report the diff, but don't parse, index or record state")
    args = parser.parse_args()

    unknown = [name for name in args.corpora if name not in SYNCS]
    if unknown:
        parser.error(f"unknown corpora {unknown}; choose from: {', '.join(SYNCS)}")

    state = json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else {}
    failed = []
    for name in args.corpora or SYNCS:
        print(f"\n=== {name} ===")
        try:
            new = SYNCS[name](state.get(name, {}), args.dry_run)
        except Exception as error:
            print(f"[{name}] FAILED: {error}", file=sys.stderr)
            failed.append(name)
            continue
        if new is not None:
            state[name] = new
            STATE_PATH.write_text(
                json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")

    if failed:
        sys.exit(f"\nfailed: {', '.join(failed)} — state kept, will retry next run")


if __name__ == "__main__":
    main()
