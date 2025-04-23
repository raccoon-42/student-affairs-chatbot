# Academic Information RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that provides information about academic calendars and university regulations. This chatbot searches through academic documents using vector embeddings and provides accurate, context-aware responses.

## Features

- Simultaneous querying of both academic calendar and university regulations
- Intelligent context retrieval using vector embeddings
- FastAPI-based REST API
- Interactive command-line interface
- Conversation history management
- Support for Turkish language queries
- Program-specific information (undergraduate/graduate/preparatory)
- Two model options (OpenAI API or local Ollama)

## Project Structure

```
rag-chatbot/
├── app/
│   ├── __init__.py
│   ├── chatbot.py             # Main chatbot logic (OpenAI)
│   ├── chatbot_local.py       # Local model integration (Ollama)
│   ├── query_handler.py       # Qdrant query processing
│   ├── api.py                 # FastAPI wrapper
│   ├── api_client.py          # API client
│   ├── ollama_client.py       # Ollama model client
│   └── llm_judge.py           # Model response evaluation
├── config/
│   ├── __init__.py
│   └── prompts/
│       └── system_prompt.txt  # System instructions
├── preprocessing/
│   ├── __init__.py
│   ├── data/                  # Data storage
│   │   ├── raw/               # Original PDF files
│   │   └── processed/         # Processed text files
│   ├── indexing/              # Indexing utilities
│   │   ├── __init__.py
│   │   ├── text_splitter.py   # Document chunking
│   │   ├── vectorizer.py      # Document vectorization
│   │   └── bm25.py            # BM25 implementation
│   └── parsers/               # PDF parsers
│       ├── academic_calendar_parser.py  # Calendar PDF parser
│       └── regulation_parser.py         # Regulations PDF parser
├── tests/
│   ├── __init__.py
│   └── test_llm_judge.py      # LLM response evaluation
├── .env                       # Environment variables
└── requirements.txt           # Dependencies
```

## Requirements

- Python 3.8+
- Qdrant vector database
- OpenAI API key or Ollama local model

## Installation

1. Clone the repository:
```bash
git clone https://github.com/raccoon-42/rag-chatbot.git
cd rag-chatbot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory:
```
OPENAI_API_KEY=your-openai-api-key
QDRANT_URL=http://localhost:6333
EMBEDDING_MODEL=intfloat/multilingual-e5-large-instruct
```

5. Set up and start the Qdrant database:
```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

## Usage

> **Note:** All commands below assume you start from the project root directory (`rag-chatbot/`). Each section includes navigation commands to move to the appropriate subdirectory and back to the root as needed.

### 1. Document Processing

First, process your PDF documents:
```bash
# Navigate to parsers directory
cd preprocessing/parsers

# Process regulations
python regulation_parser.py --input ../data/raw/regulations.pdf --output ../data/processed/regulations.txt

# Process academic calendar
python academic_calendar_parser.py --input ../data/raw/calendar.pdf --output ../data/processed/calendar.txt

# Return to root directory
cd ../..
```

### 2. Document Indexing

Index the processed documents:
```bash
# Navigate to indexing directory
cd preprocessing/indexing

# Index regulations
python vectorizer.py --input ../data/processed/regulations.txt --collection regulations

# Index academic calendar
python vectorizer.py --input ../data/processed/calendar.txt --collection academic_calendar_2025

# Return to root directory
cd ../..
```

### 3. Running the Chatbot

#### Command Line Interface
```bash
# Navigate to app directory
cd app

# With OpenAI API
python chatbot.py

# With local model (Ollama)
python chatbot_local.py

# Return to root directory
cd ..
```

#### API Server
```bash
# IMPORTANT: Make sure you're in the project root directory (rag-chatbot/)
python -m uvicorn app.api:app --reload
```

Then access the API at `http://localhost:8000`.

### 4. Example Queries

- Academic Calendar:
  ```
  >> When are the spring semester exams?
  >> Which days are holidays?
  >> What is the last date to withdraw from a course for undergraduate students in the spring semester?
  ```

- Regulations:
  ```
  >> What are the requirements for horizontal transfer?
  >> What are the rules for course retakes?
  >> What are the conditions for course withdrawal?
  ```

## How It Works

1. **Document Processing**:
   - PDFs are converted to text using preprocessing scripts
   - Documents are split into chunks using `text_splitter.py`
   - Chunks are vectorized using a multilingual embedding model
   - Vectors are stored in the Qdrant database

2. **Query Processing**:
   - User query is converted to an embedding
   - Similar chunks are retrieved from both collections
   - Context is formatted and sent to the LLM
   - Hybrid search strategy (BM25 + semantic search) is used

3. **Response Generation**:
   - LLM generates a response based on the retrieved context
   - Conversation history is maintained for context
   - Response is returned to the user

## API Endpoints

- `GET /`: Welcome message
- `GET /chat`: Send a message to the OpenAI API-based chatbot
- `GET /chat_local`: Send a message to the local model-based chatbot

## Code Reference

### Using the Chatbot Client

```python
from app.api_client import ChatbotClient

# Initialize the client
client = ChatbotClient()

# Get a response using OpenAI API
response = client.get_response_openai("When does the spring semester end?")
print(response)

# Get a response using local model
response = client.get_response_local("What is the course withdrawal date for undergraduate students?")
print(response)
```

### Integration in Your Own Application

```python
import requests

def get_chatbot_response(query):
    url = "http://localhost:8000/chat"
    response = requests.get(url, params={"query": query})
    
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return f"Error: {response.status_code}"

# Usage
print(get_chatbot_response("When do classes start in the fall semester?"))
```

## Development

This project uses a modular structure:
- `app/`: Core application logic
- `config/`: Configuration and environment variables
- `preprocessing/`: PDF processing utilities
- `tests/`: Test suite

### New Model Integration

To add a new LLM model:
1. Create a new client module specific to the model
2. Implement a new chatbot function
3. Add a new endpoint to the API

## License

This project is licensed under the MIT License - see the LICENSE file for details.