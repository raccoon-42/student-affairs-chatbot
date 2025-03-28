from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import argparse

from config import EMBEDDING_MODEL3, QDRANT_URL

def store_embedding(chunks, collection_name):
    """Encode text chunks into embeddings & store them in Qdrant"""
    
    # Loading embedding model and encoding chunks into multi-dimensional vectors
    print(f"Loading embedding model {EMBEDDING_MODEL3}...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL3, trust_remote_code=True)
    
    # Extract text and metadata from chunks
    # Add the required instruction prefix for Nomic embeddings
    texts = [chunk['text'] for chunk in chunks]
    metadata_list = [chunk["metadata"] for chunk in chunks]
    
    print("Encoding chunks into embeddings...")
    chunk_vectors = embedding_model.encode(texts)
    
    # Connect to Qdrant DB
    print("Connecting to Qdrant...")
    qdrant = QdrantClient(QDRANT_URL)
    
    # Create new collection or delete existing one
    if qdrant.collection_exists(collection_name=collection_name):
        print(f"Deleting existing collection: {collection_name}")
        qdrant.delete_collection(collection_name=collection_name)
        
    print(f"Creating new collection: {collection_name}")
    
    # Get vector dimension from the first encoded vector
    vector_size = len(chunk_vectors[0])
    print(f"Vector dimension: {vector_size}")
    
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    
    # Insert the embeddings with metadata
    print("Storing chunks in Qdrant...")
    points = []
    for i, (text, metadata) in enumerate(zip(texts, metadata_list)):
        # Create a more detailed payload
        payload = {
            "text": text,
            "metadata": metadata,
            "keywords": metadata.get("keywords", []),
            "context": metadata.get("context", {}),
            "event_type": metadata.get("event_type"),
            "academic_period": metadata.get("academic_period"),
            "date1": metadata.get("date1"),
            "date2": metadata.get("date2"),
            "parsed_date1": metadata.get("parsed_date1").isoformat() if metadata.get("parsed_date1") else None,
            "parsed_date2": metadata.get("parsed_date2").isoformat() if metadata.get("parsed_date2") else None
        }
        
        points.append(
            PointStruct(
                id=i,
                vector=chunk_vectors[i].tolist(),
                payload=payload
            )
        )
    
    qdrant.upsert(
        collection_name=collection_name,
        points=points
    )
    
    print(f"Successfully stored {len(chunks)} chunks in Qdrant.")

def store_regulation_embedding(chunks, collection_name):
    """Encode regulation text chunks into embeddings & store them in Qdrant"""
    
    # Loading embedding model and encoding chunks into multi-dimensional vectors
    print(f"Loading embedding model {EMBEDDING_MODEL3}...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL3, trust_remote_code=True)
    
    # Extract text and metadata from chunks
    texts = [chunk['text'] for chunk in chunks]
    metadata_list = [chunk["metadata"] for chunk in chunks]
    
    print("Encoding chunks into embeddings...")
    chunk_vectors = embedding_model.encode(texts)
    
    # Connect to Qdrant DB
    print("Connecting to Qdrant...")
    qdrant = QdrantClient(QDRANT_URL)
    
    # Create new collection or delete existing one
    if qdrant.collection_exists(collection_name=collection_name):
        print(f"Deleting existing collection: {collection_name}")
        qdrant.delete_collection(collection_name=collection_name)
        
    print(f"Creating new collection: {collection_name}")
    
    # Get vector dimension from the first encoded vector
    vector_size = len(chunk_vectors[0])
    print(f"Vector dimension: {vector_size}")
    
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    
    # Insert the embeddings with metadata
    print("Storing chunks in Qdrant...")
    points = []
    for i, (text, metadata) in enumerate(zip(texts, metadata_list)):
        # Create a more detailed payload for regulations
        payload = {
            "text": text,
            "metadata": metadata,
            "keywords": metadata.get("keywords", []),
            "context": metadata.get("context", {}),
            "section": metadata.get("section", ""),
            "chapter": metadata.get("chapter", ""),
            "article": metadata.get("article", ""),
            "rule_type": metadata.get("rule_type", ""),
            "category": metadata.get("category", "")
        }
        
        points.append(
            PointStruct(
                id=i,
                vector=chunk_vectors[i].tolist(),
                payload=payload
            )
        )
    
    qdrant.upsert(
        collection_name=collection_name,
        points=points
    )
    
    print(f"Successfully stored {len(chunks)} regulation chunks in Qdrant.")

def main():
    parser = argparse.ArgumentParser(description='Store text chunks in Qdrant vector database')
    parser.add_argument('input_file', help='Input text file to process')
    parser.add_argument('collection_name', help='Name of the Qdrant collection (required)')
    parser.add_argument('--type', choices=['calendar', 'regulations'], required=True,
                      help='Type of content being processed (calendar or regulations)')
    
    args = parser.parse_args()
    
    # Import here to avoid circular imports
    from text_splitter import split_text, split_regulation
    
    # Split text into chunks
    print(f"Processing file: {args.input_file}")
    
    
    # Store chunks in Qdrant based on type
    if args.type == 'calendar':
        chunks = split_text(args.input_file)
        print(f"Split text into {len(chunks)} chunks")
        store_embedding(chunks, args.collection_name)
        
    else:
        chunks = split_regulation(args.input_file)
        print(f"Split text into {len(chunks)} chunks")
        store_regulation_embedding(chunks, args.collection_name)

if __name__ == '__main__':
    main()