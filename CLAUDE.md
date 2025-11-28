# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Hardware Chatbot** is a Retrieval-Augmented Generation (RAG) application that answers questions from PDF catalogs using semantic search. It's built with Streamlit for the UI, LangChain for RAG orchestration, ChromaDB for vector storage, and OpenAI for embeddings and LLM inference.

**Key characteristic**: A production-ready RAG system optimized for technical documentation with persistent vector database (process PDFs once, reuse forever).

## Core Architecture

### RAG Pipeline Flow

```
User Question
    ↓
[app.py] - Streamlit UI & Session Management
    ↓
[rag_pipeline.py] - Query validation & chain execution
    ↓
Vector similarity search (ChromaDB)
    ↓
Retrieve top-K relevant chunks with page numbers
    ↓
[rag_pipeline.py] - Feed to GPT-4 with custom prompt
    ↓
Format answer + extract page citations
    ↓
Return to user with sources
```

### Four Core Modules

1. **`src/pdf_processor.py`** - PDF ingestion pipeline
   - `load_pdfs_from_directory()`: Reads PDFs from `PDF/` directory with PyPDFLoader, preserves page metadata
   - `chunk_documents()`: Splits with RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
   - `enrich_metadata()`: Adds chunk IDs and size tracking
   - Returns `List[Document]` with source filename and page numbers in metadata

2. **`src/vector_store.py`** - Vector database operations
   - `check_database_exists()`: Checks if `chroma_db/` persisted data exists
   - `initialize_vector_store()`: Creates or loads ChromaDB with OpenAI embeddings
   - `add_documents_to_store()`: Batch adds documents (100 at a time for API efficiency)
   - `get_retriever()`: Creates configured VectorStoreRetriever
   - Key: Uses `Chroma.from_documents()` for creation and `Chroma(persist_directory=...)` for loading

3. **`src/rag_pipeline.py`** - RAG chain and query processing
   - `create_rag_chain()`: Builds RetrievalQA with custom system prompt
   - `format_sources()`: Formats page references from Document metadata
   - `query_rag()`: Orchestrates query execution and response formatting
   - Custom prompt emphasizes citation requirement and technical accuracy

4. **`src/config.py`** - Centralized configuration
   - All tunable parameters in one place
   - Key parameters:
     - `CHUNK_SIZE=1000`, `CHUNK_OVERLAP=200` (PDF processing)
     - `LLM_MODEL="gpt-4"`, `LLM_TEMPERATURE=0.1` (LLM behavior)
     - `TOP_K_RESULTS=5` (retrieval count)
     - `FORCE_REPROCESS=False` (flag to rebuild database)

### Data Flow Details

**Initialization (first run)**:
1. `app.py` calls `initialize_vector_store()` (cached with `@st.cache_resource`)
2. `vector_store.py` checks if `chroma_db/` exists
3. If not, calls `pdf_processor.process_pdfs()` → Load + Chunk + Enrich
4. Creates embeddings via OpenAI API and persists to `chroma_db/`
5. Initialization cached, so subsequent loads just load from disk (2-5 seconds)

**Query (user asks question)**:
1. `app.py` receives question via `st.chat_input()`
2. Validates with `validate_query()`
3. Passes to `rag_pipeline.query_rag(chain, question)`
4. Chain retrieves top-K chunks, passes with question to GPT-4
5. Returns answer + `format_sources()` extracts page numbers
6. Display in chat with expandable source section

### Key Design Decisions

- **Persistent Vector Database**: Process PDFs once (~5-10 min), reuse forever (2-5 sec load)
- **Page Metadata**: Tracked through entire pipeline (Document → Chunk → Vector → Result)
- **Modular Architecture**: Each module has single responsibility, easy to test
- **GPT-4 by Default**: Better for technical documentation (switchable to gpt-3.5-turbo for cost)
- **Batch Embeddings**: Process 100 docs at a time to respect API limits

## Common Development Tasks

### Running the Application

```bash
# Start the app (runs on http://localhost:8501)
streamlit run app.py
```

**First run**:
- Will process all PDFs from `PDF/` directory
- Takes 5-10 minutes depending on PDF size
- Creates `chroma_db/` directory with persisted vectors

**Subsequent runs**:
- Loads cached database from `chroma_db/`
- Takes 2-5 seconds to start

### Modifying Configuration

Edit `src/config.py`:
```python
# Cost optimization: switch to cheaper model
LLM_MODEL = "gpt-3.5-turbo"  # ~99% cheaper per query

# Retrieve more context for complex questions
TOP_K_RESULTS = 8

# Adjust PDF chunking for longer documents
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
```

### Customizing the Prompt

Edit `SYSTEM_PROMPT` in `src/rag_pipeline.py`:
- Controls how LLM formats answers
- Can emphasize specific citation format or domain expertise
- Current prompt requires page numbers and source filename

### Adding New PDFs

1. Drop PDF files in `PDF/` directory
2. In Streamlit app, click "Reprocess PDFs" button (or set `FORCE_REPROCESS=True`)
3. App rebuilds vector database with all PDFs
4. Takes 5-10 minutes depending on total size

### Debugging Issues

**"Missing OPENAI_API_KEY"**:
- Create `.env` file (copy from `.env.example`)
- Add your OpenAI API key
- Restart the app

**"ChromaDB initialization fails"**:
- Delete `chroma_db/` directory
- Click "Reprocess PDFs" or restart app
- Check PDFs are readable with `pdf_processor.process_pdfs()` directly

**"No relevant answers"**:
- Try rephrasing question more specifically
- Increase `TOP_K_RESULTS` in config to retrieve more context
- Check PDF content actually contains answer

**Empty PDFs or encoding issues**:
- PyPDFLoader handles most encoding automatically
- If specific PDFs fail, error logged with filename

### Testing Changes

To test a specific module in isolation:

```python
# Test PDF processing
from src.pdf_processor import process_pdfs
docs = process_pdfs()
print(f"Processed {len(docs)} chunks")

# Test vector store
from src.vector_store import initialize_vector_store
vs = initialize_vector_store()
retriever = vs.as_retriever(search_kwargs={"k": 3})

# Test RAG pipeline
from src.rag_pipeline import create_rag_chain, query_rag
chain = create_rag_chain(vs)
result = query_rag(chain, "What are technical specs?")
print(result["answer"])
```

### Performance Tuning

**Faster responses**:
- Reduce `TOP_K_RESULTS` (default 5, try 3)
- Use gpt-3.5-turbo instead of gpt-4
- Reduce `LLM_MAX_TOKENS` if answers too long

**Better answer quality**:
- Increase `TOP_K_RESULTS` (try 7-8)
- Keep `LLM_TEMPERATURE` low (0.0-0.2 for factual)
- Ensure `CHUNK_SIZE` large enough (1000-1500 chars)
- Customize system prompt in `src/rag_pipeline.py`

**Lower API costs**:
- Switch to gpt-3.5-turbo (~$0.002 per query vs $0.06)
- Reduce `TOP_K_RESULTS` (fewer tokens for context)
- Use smaller embedding model (currently ada-002)

## Dependencies & Versions

Key packages (see `requirements.txt` for full list):
- **streamlit==1.28.0**: Web UI framework
- **langchain==0.1.0**: RAG orchestration
- **chromadb==0.4.18**: Vector database
- **openai==1.6.1**: API client
- **pypdf==3.17.1**: PDF loading
- **python-dotenv==1.0.0**: Environment variable management
- **tiktoken==0.5.2**: Token counting for OpenAI

### Compatibility Notes

- Requires Python 3.9+
- Streamlit breaks on major version updates (stay on 1.x)
- LangChain API unstable, stick with 0.1.0
- ChromaDB disk format may not be backwards compatible (regenerate `chroma_db/` if needed after upgrades)

## Project Structure Details

```
Hardware Chatbot/
├── PDF/                    # Input: User's 50+ PDF catalogs (not in git)
├── chroma_db/              # Auto-created: Persisted vector database (not in git)
├── src/                    # Source modules
│   ├── __init__.py
│   ├── config.py          # ⭐ Central config - adjust here first
│   ├── pdf_processor.py   # PDF loading/chunking (most I/O intensive)
│   ├── vector_store.py    # ChromaDB operations (requires OpenAI API)
│   └── rag_pipeline.py    # RAG chain & prompting (customize here)
├── app.py                 # ⭐ Main entry point - Streamlit UI
├── requirements.txt       # Dependencies
├── .env                   # User's API keys (create from .env.example)
├── .env.example           # Template (in git, .env is not)
├── .gitignore             # Excludes chroma_db/, .env, __pycache__/
└── README.md              # User documentation
```

## Key Files to Edit for Common Changes

| Goal | File | What to Change |
|------|------|-----------------|
| Adjust LLM behavior | `src/config.py` | `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`, `LLM_MODEL` |
| Change answer format | `src/rag_pipeline.py` | `SYSTEM_PROMPT` template |
| Tune retrieval | `src/config.py` | `TOP_K_RESULTS`, `CHUNK_SIZE`, `CHUNK_OVERLAP` |
| Add PDF processing logic | `src/pdf_processor.py` | Modify `process_pdfs()` or add custom loading |
| Modify UI/UX | `app.py` | Edit display functions, sidebar content |
| Database operations | `src/vector_store.py` | Change persistence path or embeddings model |

## Streaming & Caching Strategy

**Streamlit caching** (in `app.py`):
```python
@st.cache_resource
def initialize_database(_force_reprocess: bool = False):
    # Cached across reruns, cleared only when explicit button clicked

@st.cache_resource
def initialize_chain(_vector_store):
    # Cached RAG chain (underscore prefix prevents hashing unhashable type)
```

**Session state** (in `app.py`):
```python
st.session_state.messages  # Chat history persists across reruns
st.session_state.vector_store  # Reference to cached vector store
st.session_state.db_initialized  # Status flag
```

This prevents re-initialization on every Streamlit rerun (which happens when user types).

## API Cost Estimation

**One-time (processing 2000 pages)**:
- Embeddings: ~$0.80 (2000 pages × 500 avg tokens × $0.0004/1K)

**Per-query (ongoing)**:
- GPT-4: ~$0.06 (5 chunks × 500 tokens = 2500 tokens context)
- GPT-3.5: ~$0.002 (significantly cheaper)

Switch models in `src/config.py` depending on budget.

## Important Notes for Future Development

1. **Don't break the persistence logic**: The `chroma_db/` directory is created once and reused. Don't delete it or change collection names unless intentionally rebuilding.

2. **Page metadata is critical**: The entire citation system depends on Document.metadata["page"] being preserved through PDF loading → chunking → embedding → retrieval. Test this when modifying `pdf_processor.py`.

3. **API rate limits**: OpenAI has rate limits. Batch processing (100 docs) in `vector_store.py:add_documents_to_store()` respects this. Adjust batch size if hitting limits.

4. **Temperature matters**: For technical documentation, keep `LLM_TEMPERATURE=0.1` or lower. Higher values may hallucinate specifications.

5. **Chunk overlap is intentional**: 200-char overlap prevents splitting relevant context across chunk boundaries. Test before changing.

6. **Streamlit limitations**:
   - No persistent background tasks (PDF processing blocks UI)
   - Cache is session-specific (each browser tab has own cache)
   - Reruns happen on every input change (why caching is critical)
