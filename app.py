"""Main Streamlit application for Hardware Chatbot"""

import logging
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.config import (
    STREAMLIT_PAGE_TITLE,
    STREAMLIT_PAGE_ICON,
    STREAMLIT_LAYOUT,
    STREAMLIT_INITIAL_SIDEBAR_STATE,
    FORCE_REPROCESS,
)
from src.vector_store import initialize_vector_store, get_vector_store_stats
from src.rag_pipeline import create_rag_chain, query_rag, validate_query

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Streamlit page configuration
st.set_page_config(
    page_title=STREAMLIT_PAGE_TITLE,
    page_icon=STREAMLIT_PAGE_ICON,
    layout=STREAMLIT_LAYOUT,
    initial_sidebar_state=STREAMLIT_INITIAL_SIDEBAR_STATE
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def validate_environment() -> bool:
    """
    Validate that required environment variables are set.

    Returns:
        True if all required variables are set, False otherwise.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error("❌ Missing OPENAI_API_KEY environment variable")
        st.info("""
        Please set up your OpenAI API key:
        1. Copy `.env.example` to `.env`
        2. Add your OpenAI API key to `.env`
        3. Restart the application

        Get your API key from: https://platform.openai.com/api-keys
        """)
        return False

    return True


@st.cache_resource
def initialize_database(_force_reprocess: bool = False):
    """
    Initialize or load the vector database (cached).

    Args:
        _force_reprocess: If True, reprocess PDFs (note underscore to avoid hashing).

    Returns:
        Initialized Chroma vector store.
    """
    try:
        with st.spinner("📚 Initializing database..."):
            vector_store = initialize_vector_store(force_reprocess=_force_reprocess)
        return vector_store
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        st.error(f"❌ Error initializing database: {str(e)}")
        st.info("💡 Try deleting the `chroma_db/` directory and restarting the app")
        st.stop()


@st.cache_resource
def initialize_chain(_vector_store):
    """
    Initialize the RAG chain (cached).

    Args:
        _vector_store: The vector store (note underscore to avoid hashing).

    Returns:
        Initialized RetrievalQA chain.
    """
    try:
        with st.spinner("🔗 Loading RAG chain..."):
            chain = create_rag_chain(_vector_store)
        return chain
    except Exception as e:
        logger.error(f"Error creating RAG chain: {str(e)}")
        st.error(f"❌ Error creating RAG chain: {str(e)}")
        st.stop()


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None

    if "db_initialized" not in st.session_state:
        st.session_state.db_initialized = False

    if "last_temperature" not in st.session_state:
        st.session_state.last_temperature = 0.1

    if "last_k_value" not in st.session_state:
        st.session_state.last_k_value = 5


def display_sidebar():
    """Display and manage the sidebar."""
    with st.sidebar:
        st.title("⚙️ Settings")

        # Database status
        st.subheader("Database Status")
        if st.session_state.db_initialized:
            st.success("✅ Database loaded")

            try:
                stats = get_vector_store_stats(st.session_state.vector_store)
                st.metric("Documents", stats.get("total_documents", "Unknown"))
            except Exception as e:
                logger.warning(f"Could not get database stats: {str(e)}")
                st.metric("Documents", "Unknown")
        else:
            st.warning("⏳ Initializing...")

        st.divider()

        # Reprocess button
        st.subheader("Database Management")
        if st.button("🔄 Reprocess PDFs", use_container_width=True):
            st.session_state.db_initialized = False
            st.session_state.vector_store = None
            st.session_state.rag_chain = None
            # Clear cache to force reprocessing
            st.cache_resource.clear()
            st.success("✅ Cache cleared. PDFs will be reprocessed on next query.")
            st.rerun()

        st.divider()

        # RAG settings
        st.subheader("RAG Configuration")
        k_value = st.slider(
            "Number of results to retrieve",
            min_value=3,
            max_value=10,
            value=5,
            step=1,
            help="Higher values retrieve more context but may include less relevant info"
        )

        temperature = st.slider(
            "Temperature (Creativity)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get("last_temperature", 0.1),
            step=0.1,
            help="Lower values make the response more focused and deterministic"
        )

        st.divider()

        # About section
        st.subheader("ℹ️ About")
        st.markdown("""
        **Hardware Chatbot** is a RAG-based system that answers questions from timber construction tool catalogs.

        **Features:**
        - Semantic search across 50+ PDF catalogs
        - Page-accurate citations
        - Powered by GPT-4 and ChromaDB
        """)

        st.markdown("---")
        st.caption("Made with Streamlit & LangChain")

        # Store current values in session state for use in queries
        st.session_state.current_temperature = temperature
        st.session_state.current_k_value = k_value
        st.session_state.last_temperature = temperature
        st.session_state.last_k_value = k_value


def display_chat_interface():
    """Display and manage the chat interface."""
    st.title("📚 Hardware Chatbot")
    st.caption("Ask questions about timber construction tools and get answers from the catalogs with page references")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Display sources if available
            # Temporarily commented out - sources display disabled
            # if "sources" in message and message["sources"]:
            #     with st.expander("📖 View Sources"):
            #         st.markdown(message["sources"], unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Ask me about timber construction tools..."):
        # Validate query
        if not validate_query(prompt):
            st.warning("⚠️ Please enter a valid question (at least 3 characters)")
            st.rerun()

        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get RAG response
        with st.chat_message("assistant"):
            try:
                with st.spinner("🔍 Searching catalogs..."):
                    # Get current settings from session state
                    current_temperature = st.session_state.get("current_temperature", 0.1)
                    current_k_value = st.session_state.get("current_k_value", 5)
                    response = query_rag(
                        st.session_state.rag_chain,
                        prompt,
                        temperature=current_temperature,
                        top_k=current_k_value
                    )

                answer = response["answer"]
                sources = response["sources"]

                # Display answer
                st.markdown(answer)

                # Display sources in expander
                # Temporarily commented out - sources display disabled
                # if sources and sources != "No sources found.":
                #     with st.expander("📖 View Sources"):
                #         st.markdown(sources, unsafe_allow_html=True)

                # Add to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                error_message = f"❌ Error processing your question: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })

        st.rerun()

    # Clear chat button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def display_example_questions():
    """Display example questions for users."""
    if not st.session_state.messages:
        st.info("💡 **Example questions you can ask:**")
        examples = [
            "What are the technical specifications for HBS connectors?",
            "What are the load-bearing capacities mentioned in the catalogs?",
            "Compare different fastening systems available",
            "What materials are used in timber construction tools?",
        ]

        cols = st.columns(2)
        for i, example in enumerate(examples):
            with cols[i % 2]:
                if st.button(example, use_container_width=True, key=f"example_{i}"):
                    st.session_state.messages.append({
                        "role": "user",
                        "content": example
                    })
                    st.rerun()


def main():
    """Main application entry point."""
    # Initialize session state
    initialize_session_state()

    # Validate environment
    if not validate_environment():
        st.stop()

    # Initialize database on first run
    if not st.session_state.db_initialized:
        st.session_state.vector_store = initialize_database(_force_reprocess=FORCE_REPROCESS)
        st.session_state.rag_chain = initialize_chain(st.session_state.vector_store)
        st.session_state.db_initialized = True

    # Display sidebar and get settings
    display_sidebar()

    # Display chat interface
    display_chat_interface()

    # Display example questions if no chat history
    display_example_questions()


if __name__ == "__main__":
    main()
