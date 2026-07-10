import argparse
import json
import re
import uuid
from datetime import date
from pathlib import Path

from config import settings

MEVZUAT_MANIFEST = settings.ROOT / "preprocessing" / "data" / "raw" / "mevzuat" / "manifest.json"
SKS_MANIFEST = settings.ROOT / "preprocessing" / "data" / "raw" / "sks" / "manifest.json"
GUIDES_MANIFEST = settings.ROOT / "preprocessing" / "data" / "raw" / "rehber" / "manifest.json"


def _calendar_payload(text, metadata):
    return {
        "text": text,
        "metadata": metadata,
        "keywords": metadata.get("keywords", []),
        "context": metadata.get("context", {}),
        "event_type": metadata.get("event_type"),
        "academic_period": metadata.get("academic_period"),
        "date1": metadata.get("date1"),
        "date2": metadata.get("date2"),
        "parsed_date1": metadata.get("parsed_date1").isoformat() if metadata.get("parsed_date1") else None,
        "parsed_date2": metadata.get("parsed_date2").isoformat() if metadata.get("parsed_date2") else None,
    }


def _regulation_payload(text, metadata):
    return {
        "text": text,
        "metadata": metadata,
        "keywords": metadata.get("keywords", []),
        "context": metadata.get("context", {}),
        "section": metadata.get("section", ""),
        "chapter": metadata.get("chapter", ""),
        "article": metadata.get("article", ""),
        "rule_type": metadata.get("rule_type", ""),
        "category": metadata.get("category", ""),
    }


def _people_payload(text, metadata):
    return {
        "text": text,  # name + title + bio + contact lines
        "metadata": metadata,  # carries document_title, department, role, source_url
        "department": metadata.get("department"),
        "role": metadata.get("role"),
    }


def _course_payload(text, metadata):
    return {
        "text": text,  # code + name + level/department + description + prereq
        "metadata": metadata,  # carries document_title, department, levels, kind, source_url
        "department": metadata.get("department"),
        "code": metadata.get("code"),
    }


def _program_payload(text, metadata):
    return {
        "text": text,  # name + level/faculty + description + aliases
        "metadata": metadata,  # carries document_title, level, faculty, source_url
        "level": metadata.get("level"),
    }


def _sks_payload(text, metadata):
    return {
        "text": text,  # page title + section title + flattened content
        "metadata": metadata,  # carries document_title, topic, source_url
        "topic": metadata.get("topic"),
    }


def _forms_payload(text, metadata):
    return {
        "text": text,  # title + use-case description + aliases — all retrieval signal
        "metadata": metadata,  # carries title, source_url, form_code, category
        "category": metadata.get("category"),
    }


def _faq_payload(text, metadata):
    return {
        "text": text,  # the question — that's what gets embedded and matched
        "metadata": metadata,  # carries the answer, audience, category, source_url
        "audience": metadata.get("audience"),
        "category": metadata.get("category"),
    }


def calendar_chunks_from_json(input_file):
    """Chunks from the LLM parser's schedule-llm.json (one event per entry).

    Dates are already ISO, so nothing is re-parsed from Turkish text; the
    embedded text is rendered the same way as the parser's .txt output.
    """
    from preprocessing.extraction import extract_event_type, extract_academic_period
    from preprocessing.parsers.academic_calendar_parser_llm import render_line

    with open(input_file, encoding="utf-8") as f:
        events = json.load(f)

    chunks = []
    for event in events:
        text = render_line(event)
        start, end = event.get("start_date"), event.get("end_date")
        # the term carries the academic year ("2025-2026 Güz Yarıyılı") —
        # it picks that year's PDF so citation chips link to the right one
        year = re.match(r"\d{4}-\d{4}", event.get("term") or "")
        chunks.append({
            "text": text,
            "metadata": {
                "date1": start,
                "date2": end,
                "event": event["description"],
                "event_type": extract_event_type(event["description"], default="event"),
                "academic_period": extract_academic_period(event.get("term") or ""),
                "parsed_date1": date.fromisoformat(start) if start else None,
                "parsed_date2": date.fromisoformat(end) if end else None,
                "source_url": settings.CALENDAR_SOURCE_URLS.get(year.group()) if year else None,
            },
        })
    return chunks


def regulation_chunks_from_json(input_file):
    """Chunks from an LLM-parsed regulation JSON (one article per entry)."""
    from preprocessing.parsers.regulation_parser_llm import render_article

    with open(input_file, encoding="utf-8") as f:
        articles = json.load(f)

    return [
        {
            "text": render_article(article),
            "metadata": {
                "article_number": str(article["madde"]) if article.get("madde") is not None else None,
                "article_title": article.get("baslik"),
                "section": article.get("bolum"),
                "article": str(article["madde"]) if article.get("madde") is not None else None,
                "content_type": "regulation",
            },
        }
        for article in articles
    ]


def mevzuat_chunks_from_dir(input_dir):
    """Chunks from every parsed mevzuat JSON, tagged with its document.

    The scrape manifest supplies title/category/source_url per document.
    The embedded text is prefixed with the document title: all 25
    regulations contain a "MADDE 5", so a chunk must say which document
    it belongs to or retrieval and citations blur across documents.
    source_url in metadata makes each citation chip link to its own PDF.
    """
    with open(MEVZUAT_MANIFEST, encoding="utf-8") as f:
        by_stem = {Path(entry["file"]).stem: entry for entry in json.load(f)}

    chunks = []
    parsed = sorted(Path(input_dir).glob("*.json"))
    for path in parsed:
        doc = by_stem.get(path.stem)
        if doc is None:
            print(f"Warning: {path.name} not in mevzuat manifest — skipping")
            continue
        for chunk in regulation_chunks_from_json(path):
            chunk["text"] = f"{doc['title']}\n{chunk['text']}"
            chunk["metadata"].update({
                "document_title": doc["title"],
                "category": doc["category"],
                "source_url": doc["source_url"],
            })
            chunks.append(chunk)

    missing = sorted(set(by_stem) - {p.stem for p in parsed})
    if missing:
        print(f"Warning: {len(missing)} scraped documents not parsed yet, "
              f"indexing without them: {', '.join(missing)}")
    return chunks


def forms_chunks_from_json(input_file):
    """Chunks from the describer's forms.json (one link-catalog entry each).

    The embedded text stacks title, description and alias keywords so both
    the semantic and BM25 channels see how students actually ask; the URL
    rides in metadata and surfaces as the citation chip."""
    with open(input_file, encoding="utf-8") as f:
        forms = json.load(f)

    chunks = []
    for form in forms:
        lines = [form["title"], form.get("description", "")]
        if form.get("aliases"):
            lines.append("Anahtar kelimeler: " + ", ".join(form["aliases"]))
        chunks.append({
            "text": "\n".join(line for line in lines if line),
            "metadata": {
                "document_title": form["title"],
                "source_url": form["source_url"],
                "form_code": form.get("form_code"),
                "category": form.get("category"),
            },
        })
    return chunks


def _page_corpus_chunks(input_dir, manifest_path, render_chunk, corpus):
    """Chunks from every parsed page of a curated-page corpus (sks,
    rehber), tagged with its page.

    Same shape as mevzuat: the scrape manifest supplies title/topic/
    source_url per page, the embedded text is prefixed with the page
    title, and source_url makes each citation chip link to its page."""
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    by_stem = {Path(entry["file"]).stem: entry
               for entry in manifest if entry.get("kind") != "link"}

    chunks = []
    # link entries (kind: "link") are authored in the manifest itself:
    # title + description are the retrieval text, source_url (a PDF or an
    # external page) becomes the citation chip — nothing to parse
    for entry in manifest:
        if entry.get("kind") == "link":
            chunks.append({
                "text": f"{entry['title']}\n{entry['description']}",
                "metadata": {
                    "document_title": entry["title"],
                    "topic": entry["topic"],
                    "source_url": entry["source_url"],
                    "section": None,
                },
            })
    parsed = sorted(Path(input_dir).glob("*.json"))
    for path in parsed:
        page = by_stem.get(path.stem)
        if page is None:
            print(f"Warning: {path.name} not in {corpus} manifest — skipping")
            continue
        with open(path, encoding="utf-8") as f:
            page_chunks = json.load(f)
        for chunk in page_chunks:
            chunks.append({
                "text": f"{page['title']}\n{render_chunk(chunk)}",
                "metadata": {
                    "document_title": page["title"],
                    "topic": page["topic"],
                    "source_url": page["source_url"],
                    "section": chunk.get("baslik"),
                },
            })

    missing = sorted(set(by_stem) - {p.stem for p in parsed})
    if missing:
        print(f"Warning: {len(missing)} scraped pages not parsed yet, "
              f"indexing without them: {', '.join(missing)}")
    return chunks


def sks_chunks_from_dir(input_dir):
    from preprocessing.parsers.sks_parser_llm import render_chunk
    return _page_corpus_chunks(input_dir, SKS_MANIFEST, render_chunk, "sks")


def guides_chunks_from_dir(input_dir):
    from preprocessing.parsers.guides_parser_llm import render_chunk
    return _page_corpus_chunks(input_dir, GUIDES_MANIFEST, render_chunk, "rehber")


PEOPLE_ROLES = {"akademik": None,  # academic members carry their own title
                "arastirma-gorevlisi": "Araştırma Görevlisi",
                "idari": "İdari Personel"}


PEOPLE_ROSTER_SECTIONS = (("akademik", "Öğretim üyeleri"),
                          ("arastirma-gorevlisi", "Araştırma görevlileri"),
                          ("idari", "İdari personel"))


def people_chunks_from_dir(input_dir):
    """One chunk per person from every department's people JSON, PLUS a
    full-roster chunk per department.

    The person chunk stacks name, title, department, bio and contact —
    research-area questions match the bio, "maili ne / ofisi nerede"
    questions match the contact lines, and the chip links the profile.
    The roster chunk exists for the same reason as the programs "tam
    liste" chunk: "bölümün hocaları kimler" needs the whole list in
    context, not the top-k nearest three people. Per-area chunks (from
    the areas tagger) answer the filtered variant — "AI çalışan hangi
    hocalar var" — which top-k over individual bios can never enumerate."""
    chunks = []
    by_area = {}
    for path in sorted(Path(input_dir).glob("*.json")):
        with open(path, encoding="utf-8") as f:
            people = json.load(f)
        if not people:
            continue
        department = people[0]["department"]

        # academics carry their areas inline, so the roster is the
        # data-complete table for ANY aggregation the student asks for
        # (group by area, count, filter) — the model reshapes in-context
        sections = []
        for role, label in PEOPLE_ROSTER_SECTIONS:
            of_role = [p for p in people if p["role"] == role]
            if not of_role:
                continue
            entries = []
            for p in of_role:
                detail = "; ".join(part for part in
                                   (p.get("title"), ", ".join(p.get("areas") or [])) if part)
                entries.append(f"{p['name']}" + (f" ({detail})" if detail else ""))
            sections.append(f"{label} ({len(of_role)}): " + ", ".join(entries) + ".")
        chunks.append({
            "text": (f"{department} bölümü kadrosunun tam listesi ve çalışma alanları.\n"
                     + "\n".join(sections) +
                     f"\nBu listede olmayan bir kişi {department} bölümü kadrosunda kayıtlı değildir."),
            "metadata": {
                "document_title": f"{department} Kadrosu (tam liste)",
                "department": department,
                "role": None,
                "kind": "roster",
                "source_url": people[0]["source_url"].rsplit("/", 2)[0] + "/",
            },
        })

        for person in people:
            title = person.get("title") or PEOPLE_ROLES.get(person["role"]) or ""
            header = f"{person['name']}" + (f" — {title}" if title else "")
            header += f", {person['department']}"
            lines = [header]
            if person.get("bio"):
                lines.append(person["bio"])
            if person.get("areas"):
                lines.append("Çalışma alanları: " + ", ".join(person["areas"]))
                for area in person["areas"]:
                    by_area.setdefault(area, []).append(person)
            contact = person.get("contact") or {}
            if contact:
                lines.append(" | ".join(f"{key}: {value}" for key, value in contact.items()))
            chunks.append({
                "text": "\n".join(lines),
                "metadata": {
                    "document_title": person["name"],
                    "department": person["department"],
                    "role": person["role"],
                    "kind": "person",
                    "source_url": person["source_url"],
                },
            })

    # one enumeration chunk per research area, across departments; the
    # hedge is deliberate — tags come from bios, which are lossy, so no
    # "nobody else works on this" claim (unlike the roster chunk)
    for area, tagged in sorted(by_area.items()):
        names = ", ".join(
            f"{p['name']} ({p['department']}"
            + (", arş. gör.)" if p["role"] == "arastirma-gorevlisi" else ")")
            for p in tagged)
        chunks.append({
            "text": (f"{area} alanında çalışan öğretim elemanları ({len(tagged)} kişi): {names}. "
                     f"Bu liste kişilerin biyografilerindeki bilgiye dayanır."),
            "metadata": {
                "document_title": f"{area} — çalışan öğretim üyeleri",
                "department": None,
                "role": None,
                "area": area,
                "kind": "area",
                "source_url": None,
            },
        })
    return chunks


PROGRAM_LEVELS = {"lisans": "Lisans", "yukseklisans": "Yüksek Lisans", "doktora": "Doktora"}


def courses_chunks_from_dir(input_dir):
    """One chunk per course from every department's courses JSON, PLUS a
    full-catalog list chunk per department and level.

    Same lesson as programs and people: "hangi dersler var / X dersi var
    mı" is an enumeration/absence question that top-k nearest courses can
    never ground — it needs the complete per-level catalog in context.
    metadata.kind (course|list) lets the retriever admit a list chunk
    only when it outscores the best individual course."""
    chunks = []
    for path in sorted(Path(input_dir).glob("*.json")):
        with open(path, encoding="utf-8") as f:
            courses = json.load(f)
        if not courses:
            continue
        department = courses[0]["department"]

        for course in courses:
            levels = ", ".join(PROGRAM_LEVELS[lv].lower() for lv in course["levels"])
            lines = [f"{course['code']} — {course['name']} ({department}, {levels} dersi)",
                     course.get("description") or ""]
            if course.get("prerequisites"):
                lines.append(f"Önkoşul: {course['prerequisites']}")
            chunks.append({
                "text": "\n".join(line for line in lines if line),
                "metadata": {
                    "document_title": f"{course['code']} {course['name']}",
                    "department": department,
                    "code": course["code"],
                    "levels": course["levels"],
                    "kind": "course",
                    "source_url": course["source_url"],
                },
            })

        for level, label in PROGRAM_LEVELS.items():
            of_level = [c for c in courses if level in c["levels"]]
            if not of_level:
                continue
            names = ", ".join(f"{c['code']} {c['name']}" for c in of_level)
            chunks.append({
                "text": (f"{department} {label.lower()} derslerinin tam listesi "
                         f"({len(of_level)} ders): {names}. "
                         f"Bu listede olmayan bir ders {department} {label.lower()} "
                         f"programının ders kataloğunda yoktur."),
                "metadata": {
                    "document_title": f"{department} {label} Dersleri (tam liste)",
                    "department": department,
                    "code": None,
                    "levels": [level],
                    "kind": "list",
                    "source_url": None,
                },
            })
    return chunks


def program_chunks_from_json(input_file):
    """Chunks from the describer's programs.json: one per program, PLUS a
    synthesized full-list chunk per level.

    The list chunks exist because absence questions ("İYTE'de endüstri
    mühendisliği var mı?") can't be answered from the top-k nearest
    programs — claiming something doesn't exist needs the whole list in
    context, so the list says explicitly that it is complete."""
    with open(input_file, encoding="utf-8") as f:
        programs = json.load(f)

    chunks = []
    for program in programs:
        level = PROGRAM_LEVELS[program["level"]]
        header = f"{program['title']} ({level}" + (
            f", {program['faculty']})" if program.get("faculty") else " programı)")
        lines = [header, program.get("description", "")]
        if program.get("aliases"):
            lines.append("Anahtar kelimeler: " + ", ".join(program["aliases"]))
        chunks.append({
            "text": "\n".join(line for line in lines if line),
            "metadata": {
                "document_title": program["title"],
                "level": program["level"],
                "faculty": program.get("faculty"),
                "source_url": program["source_url"],
            },
        })

    for level, label in PROGRAM_LEVELS.items():
        of_level = [p for p in programs if p["level"] == level]
        if not of_level:
            continue
        names = ", ".join(p["title"] for p in of_level)
        chunks.append({
            "text": (f"İYTE {label.lower()} programlarının tam listesi "
                     f"({len(of_level)} program): {names}. "
                     f"Bu listede olmayan bir {label.lower()} programı İYTE'de yoktur."),
            "metadata": {
                "document_title": f"{label} Programları (tam liste)",
                "level": level,
                "faculty": None,
                "source_url": of_level[0]["page_url"],
            },
        })
    return chunks


def calendar_chunks_from_dir(input_dir):
    """Chunks from every parsed calendar JSON (one file per academic year)."""
    chunks = []
    for path in sorted(Path(input_dir).glob("*.json")):
        chunks.extend(calendar_chunks_from_json(path))
    return chunks


# fixed namespace so the same payload always maps to the same point id
POINT_NAMESPACE = uuid.UUID("6c1de2a3-6f3e-4c7d-9a2b-1f4e8b0c5d17")


def point_id(payload):
    """Deterministic point id: UUID5 of the canonical payload JSON.

    The id IS the content hash — an unchanged chunk keeps its id across
    runs, any change (text or metadata) yields a new id. Sync then reduces
    to a set diff of ids, with no per-document bookkeeping."""
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return str(uuid.uuid5(POINT_NAMESPACE, canonical))


def plan_sync(desired_ids, existing_ids):
    """Split into (new ids to embed+upsert, stale ids to delete)."""
    desired, existing = set(desired_ids), set(existing_ids)
    return sorted(desired - existing), sorted(existing - desired)


def store_embedding(chunks, collection_name, build_payload=_calendar_payload, recreate=False):
    """Sync chunks into Qdrant: the given chunks are the desired state.

    Only new/changed chunks are embedded; points whose chunk is gone are
    deleted. A corpus whose source didn't change costs zero embedding
    calls. --recreate drops the collection first — needed when the
    embedding backend/dimension changes (ids don't cover the vector
    space). The payload shape is the only thing that differs between
    document types, so it's a parameter.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams, PointStruct, PointIdsList

    from app.embeddings import default_embedder

    if not chunks:
        raise SystemExit("No chunks produced — refusing to empty the collection. "
                         "If that is intended, delete it in Qdrant directly.")

    desired = {}  # id -> payload; identical chunks collapse to one point
    for chunk in chunks:
        payload = build_payload(chunk["text"], chunk["metadata"])
        desired[point_id(payload)] = payload
    if len(desired) < len(chunks):
        print(f"Note: {len(chunks) - len(desired)} duplicate chunks collapsed")

    print("Connecting to Qdrant...")
    qdrant = QdrantClient(settings.QDRANT_URL)

    if recreate and qdrant.collection_exists(collection_name=collection_name):
        print(f"Deleting existing collection: {collection_name}")
        qdrant.delete_collection(collection_name=collection_name)

    # keyed by str(id) for the diff, but deletion needs the original id —
    # pre-sync collections have integer ids, which are not valid as strings
    existing = {}
    if qdrant.collection_exists(collection_name=collection_name):
        offset = None
        while True:
            records, offset = qdrant.scroll(
                collection_name=collection_name, limit=1000, offset=offset,
                with_payload=False, with_vectors=False)
            existing.update({str(record.id): record.id for record in records})
            if offset is None:
                break

    new_ids, stale_keys = plan_sync(desired, existing)
    stale_ids = [existing[key] for key in stale_keys]
    if not new_ids and not stale_ids:
        print(f"{collection_name}: up to date ({len(desired)} chunks, nothing embedded)")
        return

    if new_ids:
        embedder = default_embedder()
        print(f"Encoding {len(new_ids)} new/changed chunks "
              f"({len(desired) - len(new_ids)} unchanged, {settings.EMBEDDING_BACKEND})...")
        vectors = embedder.embed_documents([desired[pid]["text"] for pid in new_ids])

        if qdrant.collection_exists(collection_name=collection_name):
            configured = qdrant.get_collection(collection_name).config.params.vectors.size
            if configured != len(vectors[0]):
                raise SystemExit(
                    f"{collection_name} holds {configured}-dim vectors but the embedder "
                    f"returned {len(vectors[0])} — rerun with --recreate.")
        else:
            print(f"Creating new collection: {collection_name} (dim {len(vectors[0])})")
            qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
            )

        qdrant.upsert(collection_name=collection_name, points=[
            PointStruct(id=pid, vector=vector, payload=desired[pid])
            for pid, vector in zip(new_ids, vectors)
        ])

    if stale_ids:
        qdrant.delete(collection_name=collection_name,
                      points_selector=PointIdsList(points=stale_ids))

    print(f"{collection_name}: {len(new_ids)} upserted, {len(stale_ids)} deleted, "
          f"{len(desired)} chunks total.")


def main():
    parser = argparse.ArgumentParser(description='Store text chunks in Qdrant vector database')
    parser.add_argument('input_file',
                      help='Input file, or a directory of parsed JSONs to index as one collection')
    parser.add_argument('--collection', default=None,
                      help='Qdrant collection name (defaults to the configured one for the type)')
    parser.add_argument('--type',
                      choices=['calendar', 'regulations', 'faq', 'forms', 'sks', 'programs',
                               'people', 'courses', 'guides'],
                      required=True,
                      help='Type of content being processed')
    parser.add_argument('--recreate', action='store_true',
                      help='Drop the collection first (needed when the embedding backend/dim changes)')

    args = parser.parse_args()

    from preprocessing.indexing.text_splitter import split_text, split_regulation

    print(f"Processing: {args.input_file}")

    # The run's chunks are synced as the collection's desired state, so a
    # multi-document corpus must be indexed from its directory in one go —
    # indexing file-by-file would delete every other file's chunks as stale.
    is_dir = Path(args.input_file).is_dir()
    is_json = args.input_file.endswith('.json')

    if args.type == 'calendar':
        if is_dir:
            chunks = calendar_chunks_from_dir(args.input_file)
        else:
            chunks = calendar_chunks_from_json(args.input_file) if is_json else split_text(args.input_file)
        collection = args.collection or settings.CALENDAR_COLLECTION
        payload = _calendar_payload
    elif args.type == 'forms':
        chunks = forms_chunks_from_json(args.input_file)
        collection = args.collection or settings.FORMS_COLLECTION
        payload = _forms_payload
    elif args.type == 'sks':
        chunks = sks_chunks_from_dir(args.input_file)
        collection = args.collection or settings.SKS_COLLECTION
        payload = _sks_payload
    elif args.type == 'programs':
        chunks = program_chunks_from_json(args.input_file)
        collection = args.collection or settings.PROGRAMS_COLLECTION
        payload = _program_payload
    elif args.type == 'people':
        chunks = people_chunks_from_dir(args.input_file)
        collection = args.collection or settings.PEOPLE_COLLECTION
        payload = _people_payload
    elif args.type == 'courses':
        chunks = courses_chunks_from_dir(args.input_file)
        collection = args.collection or settings.COURSES_COLLECTION
        payload = _course_payload
    elif args.type == 'guides':
        chunks = guides_chunks_from_dir(args.input_file)
        collection = args.collection or settings.GUIDES_COLLECTION
        payload = _sks_payload  # same shape: page title + topic + source_url
    elif args.type == 'faq':
        # faq.json from the scraper is already one Q&A per entry; embed the question
        with open(args.input_file, encoding='utf-8') as f:
            faqs = json.load(f)
        chunks = [{"text": faq["question"], "metadata": faq} for faq in faqs]
        collection = args.collection or settings.FAQ_COLLECTION
        payload = _faq_payload
    else:
        if is_dir:
            chunks = mevzuat_chunks_from_dir(args.input_file)
        else:
            chunks = regulation_chunks_from_json(args.input_file) if is_json else split_regulation(args.input_file)
        collection = args.collection or settings.REGULATIONS_COLLECTION
        payload = _regulation_payload

    print(f"Split text into {len(chunks)} chunks")
    store_embedding(chunks, collection, payload, recreate=args.recreate)


if __name__ == '__main__':
    main()
