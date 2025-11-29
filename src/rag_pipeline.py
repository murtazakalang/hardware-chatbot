"""RAG pipeline for querying documents with LangChain"""

import logging
import re
from typing import Any, Dict, List

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from src.config import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, TOP_K_RESULTS

# Set up logging
logger = logging.getLogger(__name__)

# Custom prompt template for the RAG chain
SYSTEM_PROMPT = """You are an expert assistant for timber construction tools and hardware.
Use the following pieces of context from technical catalogs to answer the question.
If you don't know the answer based on the context, say so clearly instead of making something up.

Context:
{context}

Question: {question}

Provide a detailed and accurate answer without including source information or page references.
Answer:"""


def create_rag_chain(
    vector_store: Chroma,
    llm_model: str = None,
    temperature: float = None,
    top_k: int = None
) -> RetrievalQA:
    """
    Create a RAG chain with custom prompt for querying documents.

    Args:
        vector_store: Chroma vector store instance.
        llm_model: LLM model name. Defaults to config value.
        temperature: LLM temperature. Defaults to config value.
        top_k: Number of documents to retrieve. Defaults to config value.

    Returns:
        Configured RetrievalQA chain.
    """
    if llm_model is None:
        llm_model = LLM_MODEL
    if temperature is None:
        temperature = LLM_TEMPERATURE
    if top_k is None:
        top_k = TOP_K_RESULTS

    logger.info(f"Creating RAG chain with model={llm_model}, temperature={temperature}, top_k={top_k}")

    try:
        # Initialize LLM
        llm = ChatOpenAI(
            model_name=llm_model,
            temperature=temperature,
            max_tokens=LLM_MAX_TOKENS,
            streaming=False
        )

        # Create custom prompt
        prompt = PromptTemplate(
            template=SYSTEM_PROMPT,
            input_variables=["context", "question"]
        )

        # Create retriever
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )

        # Create RAG chain
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": prompt,
                "verbose": False
            }
        )

        logger.info("RAG chain created successfully")
        return rag_chain

    except Exception as e:
        error_msg = str(e).lower()
        if "api_key" in error_msg or "authentication" in error_msg:
            logger.error(f"OpenAI authentication error: {str(e)}")
            raise RuntimeError(
                "❌ OpenAI API key error: Invalid or missing API key.\n"
                "Please set a valid OPENAI_API_KEY in your .env file and restart the app."
            )
        elif "api" in error_msg or "openai" in error_msg:
            logger.error(f"OpenAI API error during chain creation: {str(e)}")
            raise RuntimeError(
                f"❌ OpenAI API error: {str(e)}\n"
                "Please verify your API key and configuration."
            )
        logger.error(f"Error creating RAG chain: {str(e)}")
        raise


def strip_source_citations(text: str) -> str:
    """
    Remove source citations from answer text.
    Removes patterns like (Source: ...), (Page: ...), etc.

    Args:
        text: Text containing potential source citations.

    Returns:
        Text with source citations removed.
    """
    # Remove patterns like (Source: filename, Page: X)
    text = re.sub(r'\s*\(Source:[^)]*\)', '', text)
    # Remove patterns like [Source: filename], [filename], etc.
    text = re.sub(r'\s*\[[^\]]*\]', '', text)
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def format_sources(source_documents: List[Document]) -> str:
    """
    Format source documents with page references for display.

    Args:
        source_documents: List of source Document objects from retrieval.

    Returns:
        Formatted string with source references.
    """
    if not source_documents:
        return "No sources found."

    formatted_sources = []
    seen_sources = set()

    for doc in source_documents:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        chunk_id = doc.metadata.get("chunk_id", "")

        # Create unique source key to avoid duplicates
        source_key = f"{source}_{page}"
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            formatted_sources.append(f"- **{source}** (Page {page})")

    if formatted_sources:
        return "**Sources:**\n" + "\n".join(formatted_sources)
    else:
        return "No sources available."


def query_rag(
    chain: RetrievalQA,
    question: str,
    temperature: float = None,
    top_k: int = None
) -> Dict[str, Any]:
    """
    Query the RAG pipeline and format response with sources.

    Args:
        chain: RetrievalQA chain instance.
        question: User question/query.
        temperature: Optional temperature override for this query.
        top_k: Optional top_k override for this query.

    Returns:
        Dictionary with:
            - answer: The generated answer
            - sources: Formatted source references
            - source_documents: Raw source Document objects
    """
    if not question or not isinstance(question, str):
        raise ValueError("Invalid question provided")

    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty")

    logger.info(f"Processing query: {question[:100]}...")

    try:
        # Run the RAG chain
        result = chain({"query": question})

        # Extract components
        answer = result.get("result", "")
        source_documents = result.get("source_documents", [])

        # Strip source citations from answer
        answer = strip_source_citations(answer)

        # Format sources
        formatted_sources = format_sources(source_documents)

        logger.info(f"Query processed successfully. Retrieved {len(source_documents)} sources")

        return {
            "answer": answer,
            "sources": formatted_sources,
            "source_documents": source_documents
        }

    except ValueError as e:
        logger.error(f"Invalid input to RAG chain: {str(e)}")
        raise ValueError(f"Invalid query: {str(e)}")
    except RuntimeError as e:
        if "API" in str(e) or "openai" in str(e).lower():
            logger.error(f"OpenAI API error: {str(e)}")
            raise RuntimeError(
                "❌ OpenAI API error: Unable to process your question.\n"
                "This could be due to:\n"
                "- Invalid or expired API key\n"
                "- Rate limiting from OpenAI\n"
                "- Network connectivity issues\n"
                "Please check your API key and try again."
            )
        raise
    except Exception as e:
        error_str = str(e).lower()
        if "timeout" in error_str or "connection" in error_str:
            logger.error(f"Network error: {str(e)}")
            raise RuntimeError(
                "❌ Network error: Unable to reach OpenAI API.\n"
                "Please check your internet connection and try again."
            )
        elif "api" in error_str or "openai" in error_str:
            logger.error(f"OpenAI API error: {str(e)}")
            raise RuntimeError(
                f"❌ Error communicating with OpenAI: {str(e)}\n"
                "Please verify your API key is valid and try again."
            )
        logger.error(f"Error querying RAG chain: {str(e)}")
        raise


def validate_query(question: str) -> bool:
    """
    Validate if a query is appropriate for the RAG system.

    Args:
        question: User question to validate.

    Returns:
        True if query is valid, False otherwise.
    """
    if not question:
        return False

    question = question.strip()
    if len(question) < 3:
        return False

    return True
