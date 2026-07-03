# ADR-0001: Vector store behind a seam in the Retriever

**Decision.** Retrieval goes through one deep module (`app/retrieval.py`). The
vector store is an adapter: `search(collection, vector, limit, filters) -> [Hit]`.
Two adapters exist: `QdrantVectorStore` (prod) and `InMemoryVectorStore` (tests).

**Why.** Before, query logic was spread over module-level functions with
import-time singletons (Qdrant client, embedding model, one shared BM25 whose
state accumulated across queries). Nothing could be tested without live
services. Two adapters make the seam real, and the BM25/collection-name bugs
now live in one testable place.

**Consequence.** Clients are created lazily — importing the app never loads
models or opens connections.
