from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from config import EMBEDDING_MODEL2, QDRANT_URL
import re
from datetime import datetime
import argparse

# Loading embedding model once globally to reduce time cost.
embedding_model = SentenceTransformer(EMBEDDING_MODEL2, trust_remote_code=True)

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

def query_qdrant(user_query, collection_name, top_k=10):
    print("\n===== QUERY DEBUG INFO =====")
    print(f"Query: '{user_query}'")
    
    # Convert user query into embedding
    # Add the required instruction prefix for Nomic embeddings
    formatted_query = f"search_query: {user_query}"
    print(f"Formatted query with prefix: '{formatted_query}'")
    query_vector = embedding_model.encode(formatted_query)
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
        text = point.payload.get('text', '').replace('search_document: ', '')
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
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Query academic calendar events')
    parser.add_argument('collection_name', help='Name of the Qdrant collection to query')
    parser.add_argument('--top-k', type=int, default=10, help='Number of results to return (default: 10)')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--query', help='Single query to execute (non-interactive mode)')
    parser.add_argument('--join', action='store_true', help='Join results as a single string')
    
    args = parser.parse_args()
    
    if args.interactive:
        while True:
            user_question = input(">> ")
            if user_question.lower() in ['exit', 'quit', 'q']:
                break
            results = query_qdrant(user_question, args.collection_name, args.top_k)
            
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
        results = query_qdrant(args.query, args.collection_name, args.top_k)
        
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
    
