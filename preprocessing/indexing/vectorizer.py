import argparse

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


def store_embedding(chunks, collection_name, build_payload=_calendar_payload):
    """Encode text chunks into embeddings & store them in Qdrant.

    Recreates the collection from scratch. The payload shape is the only
    thing that differs between document types, so it's a parameter.
    """
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams, PointStruct

    print(f"Loading embedding model {settings.EMBEDDING_MODEL}...")
    embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL, trust_remote_code=True)

    texts = [chunk['text'] for chunk in chunks]
    metadata_list = [chunk["metadata"] for chunk in chunks]

    print("Encoding chunks into embeddings...")
    chunk_vectors = embedding_model.encode(texts)

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
        PointStruct(id=i, vector=chunk_vectors[i].tolist(), payload=build_payload(text, metadata))
        for i, (text, metadata) in enumerate(zip(texts, metadata_list))
    ]
    qdrant.upsert(collection_name=collection_name, points=points)

    print(f"Successfully stored {len(chunks)} chunks in Qdrant.")


def main():
    parser = argparse.ArgumentParser(description='Store text chunks in Qdrant vector database')
    parser.add_argument('input_file', help='Input text file to process')
    parser.add_argument('--collection', default=None,
                      help='Qdrant collection name (defaults to the configured one for the type)')
    parser.add_argument('--type', choices=['calendar', 'regulations'], required=True,
                      help='Type of content being processed (calendar or regulations)')

    args = parser.parse_args()

    from preprocessing.indexing.text_splitter import split_text, split_regulation

    print(f"Processing file: {args.input_file}")

    if args.type == 'calendar':
        chunks = split_text(args.input_file)
        collection = args.collection or settings.CALENDAR_COLLECTION
        payload = _calendar_payload
    else:
        chunks = split_regulation(args.input_file)
        collection = args.collection or settings.REGULATIONS_COLLECTION
        payload = _regulation_payload

    print(f"Split text into {len(chunks)} chunks")
    store_embedding(chunks, collection, payload)


if __name__ == '__main__':
    main()
