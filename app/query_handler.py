from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import re
from datetime import datetime
import argparse
from dotenv import load_dotenv
import os 
from typing import List, Dict
import sys
sys.path.append("..")  # Add parent directory to Python path
from indexing.bm25 import BM25, preprocess_text

load_dotenv()

EMBEDDING_MODEL3 = os.getenv("EMBEDDING_MODEL3")
QDRANT_URL = os.getenv("QDRANT_URL")

# Loading embedding model once globally to reduce time cost.
embedding_model = SentenceTransformer(EMBEDDING_MODEL3, trust_remote_code=True)

# Initialize clients
client = QdrantClient(QDRANT_URL)
model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')

# Initialize BM25
bm25 = BM25()

def parse_date(date_str):
    """Parse Turkish date format into datetime object"""
    try:
        # Handle Turkish month abbreviations
        turkish_months = {
            'Oca': 'January', 'Şub': 'February', 'Mar': 'March',
            'Nis': 'April', 'May': 'May', 'Haz': 'June',
            'Tem': 'July', 'Ağu': 'August', 'Eyl': 'September',
            'Eki': 'October', 'Kas': 'November', 'Ara': 'December'
        }
        
        # Remove day name if present
        date_str = re.sub(r'\s+[PÇCPSPÇC]\w+$', '', date_str)
        
        # Split the date string
        parts = date_str.split('.')
        if len(parts) != 3:
            return None
            
        day, month, year = parts
        
        # Convert Turkish month abbreviation to English
        month = turkish_months.get(month, month)
        
        # Convert 2-digit year to 4-digit year
        year = '20' + year if int(year) < 50 else '19' + year
        
        # Create date string in standard format
        date_str = f"{day} {month} {year}"
        
        return datetime.strptime(date_str, "%d %B %Y")
    except:
        return None

def extract_event_type_from_query(query):
    """Extract event type from user query"""
    query = query.lower()
    
    # Check for specific event types
    if "son gün" in query or "deadline" in query or "son tarih" in query:
        return "deadline"
    elif "arasında" in query or "period" in query or "dönem" in query:
        return "period"
    elif "tatil" in query or "bayram" in query:
        return "holiday"
    elif "sınav" in query:
        return "exam"
    elif "kayıt" in query or "kaydı" in query:
        return "registration"
    elif "ders" in query:
        return "course"
    elif "mezuniyet" in query:
        return "graduation"
    elif "başvur" in query:
        return "application"
    elif "duyuru" in query or "ilan" in query:
        return "announcement"
    
    return None

def extract_academic_period_from_query(query):
    """Extract academic period from query"""
    query = query.lower()
    
    if "güz" in query:
        return "fall"
    elif "bahar" in query:
        return "spring"
    elif "yaz" in query:
        return "summer"
    
    return None

def format_date_range(date1, date2):
    """Format date range for display"""
    if date1 and date2:
        return f"{date1} - {date2}"
    elif date1:
        return date1
    return ""

def determine_collection(query):
    """Determine which collection to query based on the user's question"""
    query = query.lower()
    
    # Keywords for academic calendar
    calendar_keywords = [
        "tarih", "sınav", "tatil", "kaydı", "kayıt", "ders", "final", "vize",
        "ne zaman", "hangi tarih", "bitiyor", "kapanıyor","başlangıç", "bitiş", "dönem",
        "güz", "bahar", "yaz", "semester", "exam", "holiday", "registration", "bitcek", 
        "bitecek", "takvim", "kapan", "bit", "aç"
    ]
    
    # Keywords for regulations
    regulation_keywords = [
        "kural", "yönetmelik", "madde", "bölüm", "şart", "koşul",
        "nasıl", "nedir", "neden", "kim", "hangi", "kaç", "ne kadar",
        "rule", "regulation", "article", "section", "requirement",
        "azami", "en fazla", "en az", "en kısa", "en uzun", "en küçük", "en büyük",
    ]       
    
    # Count matches for each type
    calendar_matches = sum(1 for keyword in calendar_keywords if keyword in query)
    regulation_matches = sum(1 for keyword in regulation_keywords if keyword in query)
    
    # If both types have matches, prefer the one with more matches  
    if calendar_matches > regulation_matches:
        return "academic_calendar_2025"
    elif regulation_matches > calendar_matches:
        return "regulations"
    else:
        return None # If no clear preference, query both collections    
    


def query_qdrant_academic_calendar(query: str, top_k: int = 10) -> List[Dict]:
    """Query academic calendar collection using both BM25 and semantic search"""
    # Extract event type and academic period from query
    event_type = extract_event_type_from_query(query)
    academic_period = extract_academic_period_from_query(query)
    
    # Create filter based on extracted information
    search_filter = None
    filter_conditions = []
    
    if event_type:
        filter_conditions.append(
            FieldCondition(
                key="metadata.event_type",
                match=MatchValue(value=event_type)
            )
        )
    
    if academic_period:
        filter_conditions.append(
            FieldCondition(
                key="metadata.academic_period",
                match=MatchValue(value=academic_period)
            )
        )
    
    if filter_conditions:
        search_filter = Filter(should=filter_conditions)
    
    # Get semantic search results with filters
    query_vector = model.encode(get_detailed_instruct(query), convert_to_tensor=True)
    semantic_results = client.search(
        collection_name="academic_calendar_2025",
        query_vector=query_vector.tolist(),
        limit=top_k * 2,  # Get more results for BM25 filtering
        query_filter=search_filter
    )
    
    # Use BM25 to rank and filter the semantic results
    documents = [hit.payload["text"] for hit in semantic_results]
    bm25.fit(documents)
    bm25_scores = bm25.score(query, documents)
    
    # Combine semantic and BM25 scores
    combined_scores = []
    for i, (semantic_score, bm25_score) in enumerate(zip(semantic_results, bm25_scores)):
        combined_score = 0.7 * semantic_score.score + 0.3 * bm25_score
        combined_scores.append((i, combined_score))
    
    # Sort by combined score and take top_k
    combined_scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [i for i, _ in combined_scores[:top_k]]
    
    # Format results
    results = []
    for idx in top_indices:
        hit = semantic_results[idx]
        metadata = hit.payload.get("metadata", {})
        
        # Format the result based on event type
        if metadata.get('event_type') == 'holiday':
            formatted_text = f"🎉 {hit.payload['text']}"
        elif metadata.get('event_type') == 'exam':
            formatted_text = f"📝 {hit.payload['text']}"
        elif metadata.get('event_type') == 'deadline':
            formatted_text = f"⏰ {hit.payload['text']}"
        else:
            date_range = format_date_range(metadata.get('date1'), metadata.get('date2'))
            if date_range:
                formatted_text = f"{date_range}: {metadata.get('event', hit.payload['text'])}"
            else:
                formatted_text = hit.payload['text']
        
        results.append({
            "text": formatted_text,
            "score": combined_scores[idx][1],
            "metadata": metadata
        })
    
    return results

def query_qdrant_regulations(query: str, top_k: int = 10) -> List[Dict]:
    """Query regulations collection using both BM25 and semantic search"""
    # Get semantic search results
    query_vector = model.encode(get_detailed_instruct(query), convert_to_tensor=True)
    semantic_results = client.search(
        collection_name="regulation",
        query_vector=query_vector.tolist(),
        limit=top_k * 2  # Get more results for BM25 filtering
    )
    
    # Use BM25 to rank and filter the semantic results
    documents = [hit.payload["text"] for hit in semantic_results]
    bm25.fit(documents)
    bm25_scores = bm25.score(query, documents)
    
    # Combine semantic and BM25 scores
    combined_scores = []
    for i, (semantic_score, bm25_score) in enumerate(zip(semantic_results, bm25_scores)):
        combined_score = 0.7 * semantic_score.score + 0.3 * bm25_score
        combined_scores.append((i, combined_score))
    
    # Sort by combined score and take top_k
    combined_scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [i for i, _ in combined_scores[:top_k]]
    
    # Format results
    results = []
    for idx in top_indices:
        hit = semantic_results[idx]
        metadata = hit.payload.get("metadata", {})
        section = metadata.get('section', '')
        chapter = metadata.get('chapter', '')
        
        if section and chapter:
            formatted_text = f"📖 Chapter {chapter}, Section {section}: {hit.payload['text']}"
        else:
            formatted_text = f"📖 {hit.payload['text']}"
        
        results.append({
            "text": formatted_text,
            "score": combined_scores[idx][1],
            "metadata": metadata
        })
    
    return results

def get_detailed_instruct(query: str) -> str:
    task = "Üniversite yönetmeliği veya akademik takvim ile ilgili Türkçe bir soruya yanıt verebilecek ilgili pasajları getir"
    return f'Instruct: {task}\nQuery: {query}'

def main():
    parser = argparse.ArgumentParser(description='Query academic calendar events and regulations')
    parser.add_argument('--top-k', type=int, default=10, help='Number of results to return (default: 10)')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--query', help='Single query to execute (non-interactive mode)')
    parser.add_argument('--join', action='store_true', help='Join results as a single string')
    parser.add_argument('--collection', choices=['calendar', 'regulations', 'both'], default='both',
                      help='Which collection to query (default: both)')
    
    args = parser.parse_args()
    
    def process_query(query):
        results = []
        if args.collection in ['calendar', 'both']:
            calendar_results = query_qdrant_academic_calendar(query, args.top_k)
            results.extend(calendar_results)
        if args.collection in ['regulations', 'both']:
            regulation_results = query_qdrant_regulations(query, args.top_k)
            results.extend(regulation_results)
        return results
    
    if args.interactive:
        while True:
            user_question = input(">> ")
            if user_question.lower() in ['exit', 'quit', 'q']:
                break
            results = process_query(user_question)
            
            if args.join:
                print("\n=== Results ===")
                print("\n".join(results))
                print("=" * 50)
            else:
                for i, result in enumerate(results):
                    print(f"\nResult {i+1}:")
                    print(result)
                    print("-" * 50)
            
    elif args.query:
        results = process_query(args.query)
        
        if args.join:
            print("\n=== Results ===")
            print("\n".join(results))
            print("=" * 50)
        else:
            for i, result in enumerate(results):
                print(f"\nResult {i+1}:")
                print(result)
                print("-" * 50)
    else:
        print("Please specify either --interactive or --query")

if __name__ == "__main__":
    main()
    
