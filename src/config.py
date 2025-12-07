"""Configuration constants for Hardware Chatbot"""

import os
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent.parent
PDF_DIRECTORY = BASE_DIR / "PDF"
CHROMA_PERSIST_DIRECTORY = BASE_DIR / "chroma_db"

# ChromaDB Configuration
COLLECTION_NAME = "timber_tools_catalog"
EMBEDDING_MODEL = "text-embedding-ada-002"

# PDF Processing Configuration
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# RAG Pipeline Configuration
LLM_MODEL = "gpt-5-mini"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1000
TOP_K_RESULTS = 10

# Streamlit Configuration
STREAMLIT_PAGE_TITLE = "Hardware Chatbot"
STREAMLIT_PAGE_ICON = "📚"
STREAMLIT_LAYOUT = "wide"
STREAMLIT_INITIAL_SIDEBAR_STATE = "expanded"

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Feature Flags
FORCE_REPROCESS = False  # Set to True to ignore existing database and reprocess all PDFs
