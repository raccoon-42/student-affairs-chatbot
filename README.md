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
rag-chatbot/
├── app/
│   ├── __init__.py
│   ├── chatbot.py          # Main chatbot logic
│   ├── query_handler.py    # Qdrant querying
│   └── api.py             # FastAPI wrapper
├── config/
│   ├── __init__.py
│   └── prompts/
│       └── system_prompt.txt
├── data/
│   ├── processed/         # Processed text files
│   └── raw/              # Original PDFs
├── preprocessing/
│   ├── __init__.py
│   ├── academic_calendar_parser.py     # Regulations PDF parser
│   └── regulation_parser.py    # Academic calendar PDF parser
├── indexing/
│   ├── __init__.py
│   ├── text_splitter.py  # Document chunking
│   └── vectorizer.py     # Document vectorization
├── tests/
│   ├── __init__.py
│   └── test_chatbot.py
├── .env                  # Environment variables
└── requirements.txt
```

## Prerequisites

- Python 3.8+
- Qdrant vector database
- OpenAI API key

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

4. Create a `.env` file in the root directory:
```
OPENAI_API_KEY=your-api-key-here
```

## Usage

### 1. Document Processing

First, process your PDF documents:
```bash
# Process regulations PDF
python preprocessing/pdf_parser.py

# Process academic calendar PDF
python preprocessing/pdf_parser2.py
```

### 2. Document Indexing

Index the processed documents:
```bash
python indexing/vectorizer.py
```

### 3. Running the Chatbot

#### Command Line Interface
```bash
python app/chatbot.py
```

#### API Server
```bash
uvicorn app.api:app --reload
```

Then access the API at `http://localhost:8000`

### 4. Example Queries

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
   - PDFs are converted to text using preprocessing scripts
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
- `POST /chat`: Send a message to the chatbot
- `POST /reset`: Reset conversation history

## Development

The project uses a modular structure:
- `app/`: Core application logic
- `config/`: Configuration and environment variables
- `preprocessing/`: PDF processing utilities
- `indexing/`: Document vectorization and indexing
- `tests/`: Test suite

## License

This project is licensed under the MIT License - see the LICENSE file for details.