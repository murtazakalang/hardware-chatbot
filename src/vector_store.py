"""Vector store operations using ChromaDB"""

import logging
from pathlib import Path
from typing import List, Optional

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.config import CHROMA_PERSIST_DIRECTORY, COLLECTION_NAME, EMBEDDING_MODEL
from src.pdf_processor import process_pdfs

# Set up logging
logger = logging.getLogger(__name__)


def check_database_exists(persist_directory: str = None) -> bool:
    """
    Check if ChromaDB already exists and has data.

    Args:
        persist_directory: Path to ChromaDB directory. Defaults to config value.

    Returns:
        True if database exists with data, False otherwise.
    """
    if persist_directory is None:
        persist_directory = CHROMA_PERSIST_DIRECTORY

    persist_path = Path(persist_directory)

    if not persist_path.exists():
        logger.info(f"ChromaDB directory does not exist: {persist_directory}")
        return False

    # Check if there are any Chroma files
    chroma_files = list(persist_path.glob("*"))
    if not chroma_files:
        logger.info(f"ChromaDB directory is empty: {persist_directory}")
        return False

    logger.info(f"ChromaDB directory found with {len(chroma_files)} files")
    return True


def initialize_vector_store(
    persist_directory: str = None,
    force_reprocess: bool = False
) -> Chroma:
    """
    Initialize or load existing ChromaDB vector store.

    Args:
        persist_directory: Path to ChromaDB directory. Defaults to config value.
        force_reprocess: If True, reprocess PDFs even if database exists.

    Returns:
        Initialized Chroma vector store instance.
    """
    if persist_directory is None:
        persist_directory = CHROMA_PERSIST_DIRECTORY

    persist_directory = Path(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)

    logger.info(f"Initializing vector store at {persist_directory}")

    # Check if database already exists
    if check_database_exists(persist_directory) and not force_reprocess:
        logger.info("Loading existing ChromaDB...")
        try:
            embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
            vector_store = Chroma(
                collection_name=COLLECTION_NAME,
                persist_directory=str(persist_directory),
                embedding_function=embeddings
            )
            logger.info("Successfully loaded existing ChromaDB")
            return vector_store
        except Exception as e:
            logger.warning(f"Error loading existing database: {str(e)}. Reprocessing PDFs...")

    # Create new database
    logger.info("Creating new ChromaDB from PDFs...")
    documents = process_pdfs()

    if not documents:
        raise ValueError("No documents processed from PDFs")

    logger.info(f"Creating vector embeddings for {len(documents)} chunks...")

    try:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

        # Create vector store with persistence
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=str(persist_directory)
        )

        logger.info("Successfully created and persisted new ChromaDB")
        return vector_store

    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise


def add_documents_to_store(
    vector_store: Chroma,
    documents: List[Document],
    batch_size: int = 100
) -> None:
    """
    Add documents to existing vector store in batches.

    Args:
        vector_store: Chroma vector store instance.
        documents: List of Document objects to add.
        batch_size: Number of documents to process at once.
    """
    if not documents:
        logger.warning("No documents provided to add to store")
        return

    logger.info(f"Adding {len(documents)} documents to vector store in batches of {batch_size}")

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        try:
            vector_store.add_documents(batch)
            logger.info(f"Added batch {i // batch_size + 1}: {len(batch)} documents")
        except Exception as e:
            logger.error(f"Error adding batch starting at index {i}: {str(e)}")
            raise

    # Persist after adding
    vector_store.persist()
    logger.info("Documents added and persisted successfully")


def get_retriever(
    vector_store: Chroma,
    k: int = 5,
    score_threshold: float = 0.0
):
    """
    Create a retriever from the vector store.

    Args:
        vector_store: Chroma vector store instance.
        k: Number of top documents to retrieve.
        score_threshold: Minimum similarity score threshold.

    Returns:
        Configured VectorStoreRetriever.
    """
    logger.info(f"Creating retriever with k={k}, score_threshold={score_threshold}")

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            "score_threshold": score_threshold
        }
    )

    return retriever


def get_vector_store_stats(vector_store: Chroma) -> dict:
    """
    Get statistics about the vector store.

    Args:
        vector_store: Chroma vector store instance.

    Returns:
        Dictionary with store statistics.
    """
    try:
        collection = vector_store._collection
        count = collection.count()
        logger.info(f"Vector store contains {count} documents")
        return {"total_documents": count}
    except Exception as e:
        logger.error(f"Error getting vector store stats: {str(e)}")
        return {"total_documents": "unknown"}
