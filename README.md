# Hardware Chatbot - RAG-Based PDF Question Answering

A Streamlit application that answers questions from timber construction tool catalogs using Retrieval-Augmented Generation (RAG) with LangChain, ChromaDB, and OpenAI.

## Features

- **Semantic Search**: Intelligently search across 50+ PDF catalogs (2000+ pages)
- **Accurate Citations**: Every answer includes page references to the source documents
- **Fast Performance**: Persistent vector database loads in 2-5 seconds after initial indexing
- **Technical Expertise**: GPT-4 optimized for technical documentation about construction tools
- **User-Friendly UI**: Clean Streamlit interface with chat history and source document exploration

## Architecture

```
Hardware Chatbot/
├── PDF/                          # Source PDF catalogs
├── chroma_db/                    # Persistent vector database (auto-created)
├── src/
│   ├── config.py                 # Configuration constants
│   ├── pdf_processor.py          # PDF loading and chunking
│   ├── vector_store.py           # ChromaDB operations
│   ├── rag_pipeline.py           # RAG query pipeline
│   └── utils.py                  # Helper functions (optional)
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── .env                          # API keys (not in version control)
└── .env.example                  # Environment template
```

## Installation

### Prerequisites

- Python 3.9+
- OpenAI API key (get from https://platform.openai.com/api-keys)

### Setup Steps

1. **Clone or navigate to the project directory**
   ```bash
   cd "Hardware Chatbot"
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-api-key-here
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

The app will automatically:
- Load your OpenAI API key from `.env`
- Process all PDFs from the `PDF/` directory on first run (5-10 minutes)
- Save the vector database to `chroma_db/` for fast subsequent loads
- Open in your browser at `http://localhost:8501`

## Usage

### Asking Questions

1. Type your question in the chat input box
2. The chatbot will search the PDF catalogs and generate an answer
3. Click "View Sources" to see which documents the answer came from with page numbers

### Example Questions

- "What are the technical specifications for HBS connectors?"
- "What are the load-bearing capacities mentioned in the catalogs?"
- "Compare different fastening systems available"
- "What materials are used in timber construction tools?"

### Managing the Database

**Reprocess PDFs**: Click the "Reprocess PDFs" button in the sidebar to:
- Force re-indexing of all PDF documents
- Useful when adding new PDFs to the `PDF/` directory
- Takes 5-10 minutes depending on catalog size

**Clear Chat**: Click "Clear chat" to reset the conversation history

## Configuration

### Tunable Parameters

Edit `src/config.py` to adjust:

```python
# PDF Processing
CHUNK_SIZE = 1000              # Size of text chunks (characters)
CHUNK_OVERLAP = 200            # Overlap between chunks (characters)

# RAG Pipeline
LLM_MODEL = "gpt-4"            # Model: "gpt-4" or "gpt-3.5-turbo"
LLM_TEMPERATURE = 0.1          # Lower = more factual, Higher = more creative
LLM_MAX_TOKENS = 1000          # Max response length
TOP_K_RESULTS = 5              # Number of documents to retrieve

# Embeddings
EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI embedding model

# Feature Flags
FORCE_REPROCESS = False        # Set True to ignore existing database
```

### Sidebar Settings (Runtime)

- **Number of results to retrieve**: 3-10 documents
- **Temperature**: 0.0-1.0 (creativity level)

## Technical Details

### PDF Processing Pipeline

1. **Loading**: PyPDFLoader reads PDFs and preserves page metadata
2. **Chunking**: RecursiveCharacterTextSplitter creates overlapping chunks
3. **Enrichment**: Adds source filename and page numbers to each chunk

### Vector Database

- **Engine**: ChromaDB with cosine similarity
- **Embeddings**: OpenAI's `text-embedding-ada-002` (1536 dimensions)
- **Persistence**: Saves to disk at `chroma_db/`
- **Collection**: `timber_tools_catalog`

### RAG Pipeline

1. **Retrieval**: Semantic search returns top-5 most relevant chunks
2. **Ranking**: Based on cosine similarity to query embedding
3. **Generation**: GPT-4 reads retrieved context and generates answer
4. **Citation**: Page references extracted from document metadata

## Performance & Costs

### Initial Setup

- **Processing Time**: 5-10 minutes for 2000 pages (one-time)
- **Embedding Cost**: ~$0.80 (one-time)
- **Storage**: 500MB-1GB for vector database

### Runtime

- **Startup**: 2-5 seconds (loads cached database)
- **Query Response**: 3-8 seconds
- **Query Cost**: ~$0.06 per query with GPT-4

## Troubleshooting

### "Missing OPENAI_API_KEY"
- Copy `.env.example` to `.env`
- Add your OpenAI API key to `.env`
- Restart the application

### Database initialization fails
- Delete the `chroma_db/` directory
- Click "Reprocess PDFs" to rebuild from scratch
- Check that all PDF files are readable

### No relevant answers
- Try rephrasing your question more specifically
- Adjust the "Number of results to retrieve" slider higher
- Check that the answer exists in the PDF catalogs

### API rate limits
- Wait a few moments before sending another query
- The app includes automatic retry logic

## Adding New PDFs

1. Add PDF files to the `PDF/` directory
2. Click "Reprocess PDFs" button in the app sidebar
3. The app will re-index all documents (5-10 minutes)

## Advanced Usage

### Custom Prompt Template

Edit the `SYSTEM_PROMPT` in `src/rag_pipeline.py` to customize:
- Answer style and tone
- Citation format
- Specific instructions for technical accuracy

### Batch Processing

For large-scale query processing, use:
```python
from src.vector_store import initialize_vector_store
from src.rag_pipeline import create_rag_chain, query_rag

vector_store = initialize_vector_store()
chain = create_rag_chain(vector_store)

questions = ["question1", "question2", ...]
for q in questions:
    result = query_rag(chain, q)
    print(result["answer"])
```

## Development

### Project Structure

- **src/config.py**: Centralized configuration
- **src/pdf_processor.py**: PDF loading and chunking (most complex)
- **src/vector_store.py**: ChromaDB initialization and management
- **src/rag_pipeline.py**: RAG chain and query processing
- **app.py**: Streamlit UI and orchestration

### Logging

Logs are printed to console and can help debug issues:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

Potential improvements for future versions:

- [ ] Query expansion for better retrieval
- [ ] Confidence scoring for answers
- [ ] Analytics dashboard with query logging
- [ ] Incremental indexing for new PDFs without full reprocessing
- [ ] Multi-collection support by document type
- [ ] Streaming responses for faster perceived performance
- [ ] User feedback mechanism (thumbs up/down)
- [ ] Cache frequent queries

## Dependencies

Key packages:
- **streamlit**: Web UI framework
- **langchain**: RAG orchestration
- **chromadb**: Vector database
- **openai**: LLM and embeddings API
- **pypdf**: PDF processing
- **python-dotenv**: Environment management

See `requirements.txt` for complete list and versions.

## License

This project is provided as-is for educational and commercial use.

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review the logs in the console
3. Verify your `.env` file has the correct OpenAI API key
4. Check that PDFs are accessible in the `PDF/` directory

## Cost Disclaimer

This application uses OpenAI's API, which incurs costs:
- **Initial embedding**: ~$0.80 for 2000 pages
- **Per query**: ~$0.06 with GPT-4 (varies based on context)

For cost-sensitive usage, consider switching to `gpt-3.5-turbo` in `src/config.py`:
```python
LLM_MODEL = "gpt-3.5-turbo"  # Lower cost: ~$0.002 per query
```

---

**Created with Streamlit, LangChain, ChromaDB, and OpenAI**
