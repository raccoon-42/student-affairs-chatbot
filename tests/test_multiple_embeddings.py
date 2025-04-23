from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import argparse
import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")

def store_embedding(chunks):
    """Encode text chunks into embeddings & store them in Qdrant"""
    
    embedding_models = [
        "nomic-ai/nomic-embed-text-v1.5",
        "intfloat/multilingual-e5-large-instruct"
    ]

    collection_names = [
        "academic_calendar_2025_nomic",
        "academic_calendar_2025_intfloat"
    ]
    
    queries = [
        "lisans öğrencileri için bahar yarıyılı dersleri ne zaman bitecek?",
        "dersler ne zaman başlayacak?",
        "yatay geçiş tarihleri ne zaman?",
    ]
    
    print("Connecting to Qdrant...")
    qdrant = QdrantClient(QDRANT_URL)
    
    # Loading embedding model and encoding chunks into multi-dimensional vectors
    for model_name, collection_name in zip(embedding_models, collection_names):
        print("Loading embedding model...")
        embedding_model = SentenceTransformer(model_name, trust_remote_code=True)
    
        # Extract text and metadata from chunks
        # Add the required instruction prefix for Nomic embeddings
        if model_name == "nomic-embed-text-v1.5":   
            texts = [f"search_document: {chunk['text']}" for chunk in chunks]
        else:
            texts = [chunk['text'] for chunk in chunks]
        metadata_list = [chunk["metadata"] for chunk in chunks]
        
        print("Encoding chunks into embeddings...")
        chunk_vectors = embedding_model.encode(texts)
    
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
        
        for query in queries:
            results = qdrant.search(
                collection_name=collection_name,
                query_vector=embedding_model.encode(query),
                limit=10
            )
             
            print(f"Results for query: {query}")
            for result in results:
                print(result)
            print("\n")
        

def main():
    parser = argparse.ArgumentParser(description='Store text chunks in Qdrant vector database')
    parser.add_argument('input_file', help='Input text file to process')
    parser.add_argument('--collection-name', default='academic_calendar_2025', help='Name of the Qdrant collection')
    
    args = parser.parse_args()
    
    # Import here to avoid circular imports
    from preprocessing.indexing.text_splitter import split_text
    
    # Split text into chunks
    print(f"Processing file: {args.input_file}")
    chunks = split_text(args.input_file)
    print(f"Split text into {len(chunks)} chunks")
    
    # Store chunks in Qdrant
    store_embedding(chunks)

if __name__ == '__main__':
    main()