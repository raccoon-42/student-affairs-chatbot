import argparse
import json
from datetime import date

from config import settings


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
            },
        })
    return chunks


def regulation_chunks_from_json(input_file):
    """Chunks from the LLM parser's yonerge-llm.json (one article per entry)."""
    from preprocessing.parsers.regulation_parser_llm import render_article

    with open(input_file, encoding="utf-8") as f:
        articles = json.load(f)

    return [
        {
            "text": render_article(article),
            "metadata": {
                "article_number": str(article["madde"]),
                "article_title": article.get("baslik"),
                "section": article.get("bolum"),
                "article": str(article["madde"]),
                "content_type": "regulation",
            },
        }
        for article in articles
    ]


def store_embedding(chunks, collection_name, build_payload=_calendar_payload):
    """Encode text chunks into embeddings & store them in Qdrant.

    Recreates the collection from scratch. The payload shape is the only
    thing that differs between document types, so it's a parameter.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams, PointStruct

    from app.embeddings import default_embedder

    embedder = default_embedder()
    texts = [chunk['text'] for chunk in chunks]
    metadata_list = [chunk["metadata"] for chunk in chunks]

    print(f"Encoding chunks into embeddings ({settings.EMBEDDING_BACKEND})...")
    chunk_vectors = embedder.embed_documents(texts)

    print("Connecting to Qdrant...")
    qdrant = QdrantClient(settings.QDRANT_URL)

    if qdrant.collection_exists(collection_name=collection_name):
        print(f"Deleting existing collection: {collection_name}")
        qdrant.delete_collection(collection_name=collection_name)

    vector_size = len(chunk_vectors[0])
    print(f"Creating new collection: {collection_name} (dim {vector_size})")
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    print("Storing chunks in Qdrant...")
    points = [
        PointStruct(id=i, vector=chunk_vectors[i], payload=build_payload(text, metadata))
        for i, (text, metadata) in enumerate(zip(texts, metadata_list))
    ]
    qdrant.upsert(collection_name=collection_name, points=points)

    print(f"Successfully stored {len(chunks)} chunks in Qdrant.")


def main():
    parser = argparse.ArgumentParser(description='Store text chunks in Qdrant vector database')
    parser.add_argument('input_file', help='Input text file to process')
    parser.add_argument('--collection', default=None,
                      help='Qdrant collection name (defaults to the configured one for the type)')
    parser.add_argument('--type', choices=['calendar', 'regulations', 'faq'], required=True,
                      help='Type of content being processed')

    args = parser.parse_args()

    from preprocessing.indexing.text_splitter import split_text, split_regulation

    print(f"Processing file: {args.input_file}")

    is_json = args.input_file.endswith('.json')

    if args.type == 'calendar':
        chunks = calendar_chunks_from_json(args.input_file) if is_json else split_text(args.input_file)
        collection = args.collection or settings.CALENDAR_COLLECTION
        payload = _calendar_payload
    elif args.type == 'faq':
        # faq.json from the scraper is already one Q&A per entry; embed the question
        with open(args.input_file, encoding='utf-8') as f:
            faqs = json.load(f)
        chunks = [{"text": faq["question"], "metadata": faq} for faq in faqs]
        collection = args.collection or settings.FAQ_COLLECTION
        payload = _faq_payload
    else:
        chunks = regulation_chunks_from_json(args.input_file) if is_json else split_regulation(args.input_file)
        collection = args.collection or settings.REGULATIONS_COLLECTION
        payload = _regulation_payload

    print(f"Split text into {len(chunks)} chunks")
    store_embedding(chunks, collection, payload)


if __name__ == '__main__':
    main()
