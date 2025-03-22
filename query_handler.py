from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from config import EMBEDDING_MODEL, QDRANT_URL

# Loading embedding model once globally to reduce time cost.
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

def query_qdrant(user_query, collection_name, top_k=10):
    # Convert user query into embedding to enable searching inside database with Cosine Similarity.
    query_vector = embedding_model.encode(user_query)
    
    qdrant = QdrantClient(QDRANT_URL)
    
    # Do similarity search inside QdrantDB
    search_results = qdrant.query_points(
        collection_name=collection_name,
        query=query_vector.tolist(),
        with_payload=True,
        limit=top_k
    )
    
    return [point.payload['text'] for point in search_results.points]
    

if __name__ == "__main__":
    collection_name = input("Enter collection name to query: ")
    
    while(1):
        user_question = input(">> ")
        results = query_qdrant(user_question, collection_name)
        print("\n".join(results))
    
