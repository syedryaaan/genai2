"""Streamlit Web Application for GER-3620-P RAG System.

Provides:
- Input query interface with submit button.
- Clear grounded answer display.
- Verified PDF page citations.
- Strict 'Not found' display for unsupported queries.
- Diagnostic panel displaying retrieved chunks and metadata.
"""

import streamlit as st
from config import (
    GEMINI_API_KEY,
    NOT_FOUND_RESPONSE,
    PDF_FILENAME,
    TOP_K,
)
from src.answer_generator import AnswerGenerator
from src.embeddings import get_embedding_function
from src.retriever import ChromaVectorStoreManager

st.set_page_config(
    page_title="GER-3620-P Gas Turbine Assistant",
    page_icon="⚙️",
    layout="centered",
)

# Custom minimal styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .answer-box {
        background-color: #F8FAFC;
        border-left: 4px solid #2563EB;
        padding: 1rem;
        border-radius: 4px;
        margin-top: 1rem;
    }
    .citation-tag {
        display: inline-block;
        background-color: #EFF6FF;
        color: #1D4ED8;
        padding: 0.3rem 0.6rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    .not-found-box {
        background-color: #FEF2F2;
        border-left: 4px solid #DC2626;
        padding: 1rem;
        border-radius: 4px;
        color: #991B1B;
        font-weight: 600;
        font-size: 1.1rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_rag_pipeline():
    """Initialize vector store and answer generator once per session."""
    embed_fn = get_embedding_function()
    retriever_mgr = ChromaVectorStoreManager(embedding_function=embed_fn)
    generator = AnswerGenerator()
    return retriever_mgr, generator


st.markdown('<div class="main-header">GE Vernova GER-3620-P Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Strictly grounded Question-Answering system over <i>Heavy-Duty Gas Turbine Operating and Maintenance Considerations (GER-3620P)</i>. '
    'Answers are verified against source PDF pages. If sufficient evidence is not present, the system returns <code>Not found</code>.'
    '</div>',
    unsafe_allow_html=True,
)

if not GEMINI_API_KEY:
    st.error("⚠️ GEMINI_API_KEY is not configured in `.env`. Please supply an API key to run queries.")
    st.stop()

try:
    retriever_mgr, generator = load_rag_pipeline()
    total_indexed = retriever_mgr.count()
    if total_indexed == 0:
        st.warning("⚠️ No documents indexed yet in ChromaDB. Run `python ingest.py` in your terminal to index the PDF.")
except Exception as e:
    st.error(f"Failed to load pipeline: {e}")
    st.stop()

# Query Input
with st.form(key="qa_form"):
    question = st.text_input(
        "Ask a question about gas turbine operating and maintenance considerations:",
        placeholder="e.g. What are the typical baseline inspection intervals for 7E.03?",
    )
    submit_button = st.form_submit_button(label="Submit Question", use_container_width=True)

if submit_button:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving evidence and verifying citations..."):
            retrieved_chunks = retriever_mgr.retrieve(question.strip(), top_k=TOP_K)
            result = generator.generate_answer(question.strip(), retrieved_chunks)

        answer_text = result["answer"]
        citation_text = result.get("citation", "")
        is_grounded = result.get("is_grounded", False)

        st.subheader("Answer")
        if answer_text == NOT_FOUND_RESPONSE:
            st.markdown(f'<div class="not-found-box">{NOT_FOUND_RESPONSE}</div>', unsafe_allow_html=True)
            st.caption(f"Reason: {result.get('reason', 'Document does not contain sufficient verified evidence.')}")
        else:
            st.markdown(f'<div class="answer-box">{answer_text}</div>', unsafe_allow_html=True)
            if citation_text:
                st.markdown(f'<div class="citation-tag">📄 {citation_text}</div>', unsafe_allow_html=True)

        # Inspection Details (Passages & Metadata)
        with st.expander("🔍 Inspect Retrieved Passages & Verification Metadata"):
            st.write(f"**Retrieved Chunks Count:** {len(retrieved_chunks)}")
            st.write(f"**Cited PDF Pages:** {result.get('cited_pages', [])}")
            st.write(f"**Grounding Status:** {'Verified Grounded' if is_grounded else 'Abstained / Not Found'}")
            st.markdown("---")
            for idx, chunk in enumerate(retrieved_chunks):
                st.markdown(f"**Passage {idx + 1}** — *PDF Page {chunk['page_number']}* | Chunk ID: `{chunk['chunk_id']}`")
                st.text(chunk["text"])
