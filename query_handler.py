from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from config import EMBEDDING_MODEL3, QDRANT_URL
import re
from datetime import datetime
import argparse

# Loading embedding model once globally to reduce time cost.
embedding_model = SentenceTransformer(EMBEDDING_MODEL3, trust_remote_code=True)

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
    elif "kayıt" in query:
        return "registration"
    elif "ders" in query:
        return "course"
    elif "mezuniyet" in query:
        return "graduation"
    elif "başvuru" in query:
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
        "tarih", "sınav", "tatil", "kayıt", "ders", "final", "vize",
        "ne zaman", "hangi tarih", "başlangıç", "bitiş", "dönem",
        "güz", "bahar", "yaz", "semester", "exam", "holiday", "registration"
    ]
    
    # Keywords for regulations
    regulation_keywords = [
        "kural", "yönetmelik", "madde", "bölüm", "şart", "koşul",
        "nasıl", "nedir", "neden", "kim", "hangi", "kaç", "ne kadar",
        "rule", "regulation", "article", "section", "requirement"
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
    


def query_qdrant_academic_calendar(user_query, collection_name="academic_calendar_2025", top_k=10):
    print("\n===== QUERY DEBUG INFO =====")
    
    # Convert user query into embedding
    print(f"Query: '{user_query}'")
    query_vector = embedding_model.encode(user_query)
    print(f"Vector dimension: {len(query_vector)}")
    
    qdrant = QdrantClient(QDRANT_URL)
    
    # Extract event type and academic period from query
    event_type = extract_event_type_from_query(user_query)
    academic_period = extract_academic_period_from_query(user_query)
    
    print(f"Detected event_type: {event_type}")
    print(f"Detected academic_period: {academic_period}")
    
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
        print(f"Applied filters: {filter_conditions}")
        search_filter = Filter(should=filter_conditions)
    else:
        print("No filters applied - using semantic search only")
    
    # Do similarity search inside QdrantDB with optional filtering
    search_results = qdrant.query_points(
        collection_name=collection_name,
        query=query_vector.tolist(),
        with_payload=True,
        limit=top_k,
        query_filter=search_filter
    )
    
    print(f"Number of results: {len(search_results.points)}")
    print("============================\n")
    
    # Process and sort results
    results = []
    for point in search_results.points:
        text = point.payload.get('text', '')
        metadata = point.payload.get('metadata', {})
        
        # Format the result based on event type
        if metadata.get('event_type') == 'holiday':
            formatted_text = f"🎉 {text} (Score: {point.score:.2f})"
        elif metadata.get('event_type') == 'exam':
            formatted_text = f"📝 {text} (Score: {point.score:.2f})"
        elif metadata.get('event_type') == 'deadline':
            formatted_text = f"⏰ {text} (Score: {point.score:.2f})"
        else:
            date_range = format_date_range(metadata.get('date1'), metadata.get('date2'))
            if date_range:
                formatted_text = f"{date_range}: {metadata.get('event', text)} (Score: {point.score:.2f})"
            else:
                formatted_text = f"{text} (Score: {point.score:.2f})"
        
        results.append(formatted_text)
        
    print(results)
    return results

def query_qdrant_regulations(user_query, collection_name="regulation", top_k=10):
    """Query regulations collection with section and rule-specific handling"""
    print("\n===== REGULATIONS QUERY =====")
    print(f"Query: '{user_query}'")
    
    query_vector = embedding_model.encode(user_query)
    qdrant = QdrantClient(QDRANT_URL)
    
    # Simple search for regulations (can be enhanced with regulation-specific filters)
    search_results = qdrant.query_points(
        collection_name=collection_name,
        query=query_vector.tolist(),
        with_payload=True,
        limit=top_k
    )
    
    # Format regulation-specific results
    results = []
    for point in search_results.points:
        text = point.payload.get('text', '')
        metadata = point.payload.get('metadata', {})
        
        # Add section/chapter information if available
        section = metadata.get('section', '')
        chapter = metadata.get('chapter', '')
        if section and chapter:
            formatted_text = f"📖 Chapter {chapter}, Section {section}: {text} (Score: {point.score:.2f})"
        else:
            formatted_text = f"📖 {text} (Score: {point.score:.2f})"
        
        results.append(formatted_text)
    
    return results

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
    
