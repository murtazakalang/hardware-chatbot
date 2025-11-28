"""PDF processing module for loading, chunking, and enriching PDF documents"""

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import PDF_DIRECTORY, CHUNK_SIZE, CHUNK_OVERLAP

# Set up logging
logger = logging.getLogger(__name__)


def load_pdfs_from_directory(directory_path: str = None) -> List[Document]:
    """
    Load all PDFs from specified directory with page metadata.

    Args:
        directory_path: Path to directory containing PDFs. Defaults to config PDF_DIRECTORY.

    Returns:
        List of LangChain Document objects with page metadata preserved.

    Raises:
        FileNotFoundError: If directory doesn't exist or contains no PDFs.
    """
    if directory_path is None:
        directory_path = PDF_DIRECTORY

    directory = Path(directory_path)
    if not directory.exists():
        error_msg = (
            f"❌ PDF directory not found: {directory}\n\n"
            f"Please ensure the PDF/ directory exists in your project root with PDF files.\n"
            f"You can create it and add PDF files, then restart the application."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Find all PDF files
    pdf_files = list(directory.glob("*.pdf"))
    if not pdf_files:
        error_msg = (
            f"⚠️ No PDF files found in {directory}\n\n"
            f"Please add PDF files to the PDF/ directory and restart the application.\n"
            f"Supported format: .pdf files"
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info(f"Found {len(pdf_files)} PDF files in {directory}")

    documents = []
    for pdf_file in sorted(pdf_files):
        try:
            logger.info(f"Loading PDF: {pdf_file.name}")
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()

            # PyPDFLoader automatically adds 'page' metadata to each document
            # Each page is a separate document
            for doc in docs:
                doc.metadata["source"] = pdf_file.name
                doc.metadata["source_path"] = str(pdf_file)

            documents.extend(docs)
            logger.info(f"Successfully loaded {pdf_file.name} ({len(docs)} pages)")

        except Exception as e:
            logger.error(f"Error loading {pdf_file.name}: {str(e)}")
            raise

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into chunks with overlap using RecursiveCharacterTextSplitter.
    Preserves page number metadata.

    Args:
        documents: List of Document objects to chunk.

    Returns:
        List of chunked Document objects with preserved metadata.
    """
    if not documents:
        raise ValueError("No documents provided for chunking")

    logger.info(f"Chunking {len(documents)} documents...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = []
    for doc in documents:
        try:
            split_texts = splitter.split_text(doc.page_content)

            for i, chunk_text in enumerate(split_texts):
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata={
                        "source": doc.metadata.get("source", "unknown"),
                        "source_path": doc.metadata.get("source_path", ""),
                        "page": doc.metadata.get("page", 0),
                        "chunk_index": i,
                        "original_doc_page": doc.metadata.get("page", 0)
                    }
                )
                chunks.append(chunk_doc)

        except Exception as e:
            logger.error(f"Error chunking document with source {doc.metadata.get('source')}: {str(e)}")
            raise

    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks


def enrich_metadata(chunks: List[Document]) -> List[Document]:
    """
    Add additional metadata to chunks for better tracking and retrieval.

    Args:
        chunks: List of chunked Document objects.

    Returns:
        Enhanced Document objects with additional metadata.
    """
    logger.info(f"Enriching metadata for {len(chunks)} chunks...")

    for i, chunk in enumerate(chunks):
        # Add chunk ID
        chunk.metadata["chunk_id"] = f"{chunk.metadata['source']}_page_{chunk.metadata['page']}_chunk_{chunk.metadata['chunk_index']}"

        # Add character count for reference
        chunk.metadata["chunk_size"] = len(chunk.page_content)

    logger.info("Metadata enrichment completed")
    return chunks


def process_pdfs() -> List[Document]:
    """
    Main entry point: Load, chunk, and enrich PDF documents.

    Returns:
        Processed and enriched Document objects ready for vector storage.
    """
    logger.info("Starting PDF processing...")

    # Load PDFs
    documents = load_pdfs_from_directory()

    # Chunk documents
    chunks = chunk_documents(documents)

    # Enrich metadata
    enriched_chunks = enrich_metadata(chunks)

    logger.info(f"PDF processing complete. Ready to store {len(enriched_chunks)} chunks in vector database")
    return enriched_chunks
