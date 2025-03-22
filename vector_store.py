from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from config import EMBEDDING_MODEL, QDRANT_URL

def store_embedding(chunks, collection_name):
    """Encode text chunks into embeddings & store them in Qdrant"""
    
    # Loading embedding model and encoding chunks into multi-dimensional vectors.
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    chunk_vectors = embedding_model.encode(chunks)
    
    # Connect to Qdrant DB
    qdrant = QdrantClient(QDRANT_URL)
    
    # Creating new collection with Cosine Similarity distance and 384 dimensionality configuration.
    if qdrant.collection_exists(collection_name=collection_name):
        qdrant.delete_collection(collection_name=collection_name)
        
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )
    print(f"Created new collection: {collection_name}.")

  
    # Inserting the embeddings with index and tagging with payload text.
    qdrant.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(id=i, vector=chunk_vectors[i].tolist(), payload={"text": chunks[i]}) for i in range(len(chunk_vectors))
        ]
    )
    
    print(f"{len(chunks)} chunks stored into Qdrant.")
    
    
if __name__ == '__main__':
    from text_splitter import split_text
    text_file_to_split = input("Enter txt file name: ")
    chunks = split_text(text_file_to_split)
    collection_name = input("Enter collection name: ")
    store_embedding(chunks, collection_name)