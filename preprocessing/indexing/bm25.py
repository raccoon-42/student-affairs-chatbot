from typing import List, Dict
import numpy as np
from collections import Counter
import re

class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 with parameters
        k1: term saturation parameter
        b: length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self.doc_freqs = {}
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0
        self.doc_count = 0
        self.doc_lists = {}
        
    def fit(self, documents: List[str]):
        """
        Fit BM25 by computing document frequencies and other statistics
        """
        self.doc_count = len(documents)
        self.doc_len = [len(doc.split()) for doc in documents]
        if self.doc_count == 0:
            self.avgdl = 0  # or raise an exception, or set to None
        else:
            self.avgdl = sum(self.doc_len) / self.doc_count
        
        # Compute document frequencies
        for doc in documents:
            words = doc.split()
            word_freq = Counter(words)
            
            for word, freq in word_freq.items():
                if word not in self.doc_freqs:
                    self.doc_freqs[word] = 0
                self.doc_freqs[word] += 1
                
        # Compute IDF
        for word, freq in self.doc_freqs.items():
            self.idf[word] = np.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1)
            
    def score(self, query: str, documents: List[str]) -> List[float]:
        """
        Score documents based on the query
        Returns: List of BM25 scores for each document
        """
        query_words = query.split()
        scores = []
        
        for i, doc in enumerate(documents):
            score = 0
            words = doc.split()
            doc_len = len(words)
            
            for word in query_words:
                if word in self.idf:
                    # Term frequency in document
                    tf = words.count(word) / doc_len
                    # BM25 formula
                    score += self.idf[word] * (tf * (self.k1 + 1)) / (tf + self.k1)
                    
            # Length normalization
            score *= (1 - self.b + self.b * doc_len / self.avgdl)
            scores.append(score)
            
        return scores
    
    def rank_documents(self, query: str, documents: List[str], top_k: int = 5) -> List[tuple]:
        """
        Rank documents based on BM25 scores
        Returns: List of (document, score) tuples sorted by score
        """
        scores = self.score(query, documents)
        ranked_docs = list(zip(documents, scores))
        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        return ranked_docs[:top_k]

def preprocess_text(text: str) -> str:
    """
    Basic text preprocessing
    """
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text 