# Academic Information RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that provides information about academic calendar events and university regulations. The chatbot uses vector embeddings to search through academic documents and provides accurate, context-aware responses.

## Features

- Query both academic calendar and university regulations simultaneously
- Intelligent context retrieval using vector embeddings
- FastAPI-based REST API
- Interactive command-line interface
- Conversation history management
- Support for Turkish language queries

## Project Structure

```
.
├── app.py                 # FastAPI server implementation
├── chatbot.py            # Core chatbot logic and conversation management
├── config.py             # Configuration settings
├── pdf_parser.py         # PDF document parsing utilities
├── query_handler.py      # Vector search and query processing
├── text_splitter.py      # Document chunking and preprocessing
├── vector_store.py       # Vector database operations
├── system_prompt.txt     # LLM system prompt
├── schedule.txt          # Academic calendar data
└── yonerge.txt          # University regulations data
```

## Prerequisites

- Python 3.8+
- Qdrant vector database
- OpenAI API key (for OpenRouter)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/raccoon-42/rag-chatbot.git
cd rag-chatbot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your environment variables in `config.py`:
```python
QDRANT_URL = "http://localhost:6333"
OPENAI_API_KEY = "your-api-key"
```

## Usage

### 1. Setting up the Vector Database

First, process and store your documents in the vector database:

```bash
# Store academic calendar
python vector_store.py schedule.txt academic_calendar_2025 --type calendar

# Store regulations
python vector_store.py yonerge.txt regulations --type regulations
```

### 2. Running the Chatbot

#### Command Line Interface
```bash
python chatbot.py
```

#### API Server
```bash
uvicorn app:app --reload
```

Then access the API at `http://localhost:8000`

### 3. Example Queries

- Academic Calendar:
  ```
  >> Bahar dönemi sınavları ne zaman?
  >> Tatil günleri hangileri?
  ```

- Regulations:
  ```
  >> Yatay geçiş şartları nelerdir?
  >> Ders tekrarı kuralları nedir?
  ```

## How It Works

1. **Document Processing**:
   - Documents are split into chunks using `text_splitter.py`
   - Chunks are embedded using a multilingual embedding model
   - Vectors are stored in Qdrant database

2. **Query Processing**:
   - User query is converted to embedding
   - Similar chunks are retrieved from both collections
   - Context is formatted and sent to LLM

3. **Response Generation**:
   - LLM generates response based on retrieved context
   - Conversation history is maintained for context
   - Response is returned to user

## API Endpoints

- `GET /`: Welcome message
- `GET /chat?query=your_question`: Get chatbot response

## License

This project is licensed under the MIT License - see the LICENSE file for details.