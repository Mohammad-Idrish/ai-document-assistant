import streamlit as st
from utils.pdf_processor import extract_text_from_pdf, chunk_text
from utils.vector_store import build_vector_store, retrieve_relevant_chunks
from utils.groq_client import ask_question, summarize_text

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0d0d0d;
    color: #f0ede6;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #141414;
    border-right: 1px solid #2a2a2a;
}

/* Title */
.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: #f0ede6;
    letter-spacing: -1px;
    line-height: 1.1;
}

.main-title span {
    color: #c8f55a;
}

.subtitle {
    color: #6b6b6b;
    font-size: 1rem;
    margin-top: 4px;
    font-weight: 300;
}

/* Cards */
.info-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 12px 0;
}

.info-card h4 {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #c8f55a;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 0 0 8px 0;
}

.info-card p {
    color: #a0a0a0;
    font-size: 0.92rem;
    margin: 0;
    line-height: 1.5;
}

/* Answer box */
.answer-box {
    background: #111;
    border: 1px solid #c8f55a33;
    border-left: 3px solid #c8f55a;
    border-radius: 8px;
    padding: 20px 24px;
    margin-top: 12px;
    color: #f0ede6;
    font-size: 0.97rem;
    line-height: 1.7;
}

/* Summary box */
.summary-box {
    background: #111;
    border: 1px solid #5af5c833;
    border-left: 3px solid #5af5c8;
    border-radius: 8px;
    padding: 20px 24px;
    margin-top: 12px;
    color: #f0ede6;
    font-size: 0.97rem;
    line-height: 1.7;
}

/* Chunk preview */
.chunk-preview {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 6px 0;
    color: #8a8a8a;
    font-size: 0.85rem;
    line-height: 1.5;
}

/* Buttons */
.stButton > button {
    background: #c8f55a !important;
    color: #0d0d0d !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
    letter-spacing: 1px !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 10px 24px !important;
    text-transform: uppercase !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    background: #d4ff66 !important;
    transform: translateY(-1px) !important;
}

/* Input */
.stTextInput > div > div > input {
    background: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    color: #f0ede6 !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
}

.stTextInput > div > div > input:focus {
    border-color: #c8f55a !important;
    box-shadow: 0 0 0 1px #c8f55a40 !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #1a1a1a;
    border: 1px dashed #2a2a2a;
    border-radius: 12px;
    padding: 10px;
}

/* Divider */
hr {
    border-color: #1f1f1f !important;
    margin: 20px 0 !important;
}

/* Metric */
[data-testid="stMetric"] {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    padding: 14px 18px;
}

[data-testid="stMetricValue"] {
    color: #c8f55a !important;
    font-family: 'Space Mono', monospace !important;
}

/* Success/info messages */
.stSuccess {
    background: #1a2a1a !important;
    border-color: #c8f55a !important;
    color: #c8f55a !important;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #141414;
    border-radius: 8px;
    gap: 4px;
    padding: 4px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #6b6b6b;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 1px;
}

.stTabs [aria-selected="true"] {
    background: #c8f55a !important;
    color: #0d0d0d !important;
    border-radius: 6px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "full_text" not in st.session_state:
    st.session_state.full_text = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "doc_name" not in st.session_state:
    st.session_state.doc_name = ""

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 20px 0;'>
        <div style='font-family: Space Mono, monospace; font-size: 1.1rem; font-weight: 700; color: #f0ede6;'>
            📄 DOC<span style='color:#c8f55a;'>AI</span>
        </div>
        <div style='font-size: 0.75rem; color: #4a4a4a; margin-top: 2px;'>RAG-powered assistant</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        if st.button("⚡ Process Document", use_container_width=True):
            with st.spinner("Extracting text..."):
                raw_text = extract_text_from_pdf(uploaded_file)

            if not raw_text.strip():
                st.error("Could not extract text. Try a different PDF.")
            else:
                with st.spinner("Building vector index..."):
                    chunks = chunk_text(raw_text)
                    vector_store = build_vector_store(chunks)

                st.session_state.vector_store = vector_store
                st.session_state.chunks = chunks
                st.session_state.full_text = raw_text
                st.session_state.chat_history = []
                st.session_state.summary = ""
                st.session_state.doc_name = uploaded_file.name
                st.success("Document ready!")

    # Stats
    if st.session_state.vector_store:
        st.markdown("---")
        st.markdown("### Document Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chunks", len(st.session_state.chunks))
        with col2:
            words = len(st.session_state.full_text.split())
            st.metric("Words", f"{words:,}")

        st.markdown(f"""
        <div class='info-card'>
            <h4>Loaded File</h4>
            <p>{st.session_state.doc_name}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size: 0.75rem; color: #3a3a3a; line-height: 1.6;'>
        Built with LangChain · FAISS · Groq<br>
        Sentence-Transformers · Streamlit
    </div>
    """, unsafe_allow_html=True)

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-title'>AI Document <span>Assistant</span></div>
<div class='subtitle'>Upload a PDF → Ask questions → Get instant answers with RAG</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if not st.session_state.vector_store:
    # Landing state
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='info-card'>
            <h4>Step 01</h4>
            <p>Upload any PDF document using the sidebar panel on the left.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='info-card'>
            <h4>Step 02</h4>
            <p>Click Process — the doc is chunked, embedded and indexed with FAISS.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='info-card'>
            <h4>Step 03</h4>
            <p>Ask questions or generate a summary — powered by Groq Llama-3.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <br>
    <div style='text-align:center; color:#2a2a2a; font-family: Space Mono, monospace; font-size: 4rem; margin-top: 40px;'>
        ↑
    </div>
    <div style='text-align:center; color:#3a3a3a; font-size: 0.9rem; margin-top: 8px;'>
        Upload a PDF to get started
    </div>
    """, unsafe_allow_html=True)

else:
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["💬  Q&A CHAT", "📝  SUMMARIZE", "🔍  CHUNKS"])

    # ── Tab 1: Q&A ────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)

        # Chat history
        for entry in st.session_state.chat_history:
            st.markdown(f"""
            <div style='background:#1a1a1a; border:1px solid #2a2a2a; border-radius:8px;
                        padding:12px 18px; margin:6px 0; font-size:0.9rem; color:#c8f55a;'>
                🧑 {entry['question']}
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class='answer-box'>
                🤖 {entry['answer']}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        # Input
        with st.form("qa_form", clear_on_submit=True):
            question = st.text_input(
                "Ask a question",
                placeholder="What is this document about? Who are the key people mentioned?",
                label_visibility="collapsed"
            )
            submitted = st.form_submit_button("Ask →", use_container_width=False)

        if submitted and question.strip():
            with st.spinner("Searching document & generating answer..."):
                relevant_chunks = retrieve_relevant_chunks(
                    question,
                    st.session_state.vector_store,
                    st.session_state.chunks,
                    k=4
                )
                context = "\n\n".join(relevant_chunks)
                answer = ask_question(question, context)

            st.session_state.chat_history.append({
                "question": question,
                "answer": answer
            })
            st.rerun()

        if st.session_state.chat_history:
            if st.button("🗑 Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

    # ── Tab 2: Summary ────────────────────────────────────────────────────────
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])

        with col1:
            summary_type = st.radio(
                "Summary type",
                ["Brief (2-3 sentences)", "Detailed (key points)", "Executive (business format)"],
                label_visibility="visible"
            )

        with col2:
            if st.button("✨ Generate Summary", use_container_width=True):
                with st.spinner("Summarizing document..."):
                    # Use first ~3000 words to stay within token limits
                    text_snippet = " ".join(st.session_state.full_text.split()[:1500])
                    st.session_state.summary = summarize_text(text_snippet, summary_type)

        if st.session_state.summary:
            st.markdown(f"""
            <div class='summary-box'>
                {st.session_state.summary.replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)

    # ── Tab 3: Chunks ─────────────────────────────────────────────────────────
    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='info-card'>
            <h4>Vector Index Info</h4>
            <p>Document split into <strong style='color:#c8f55a'>{len(st.session_state.chunks)} chunks</strong>
            using 500-token windows with 50-token overlap. Embedded with
            <strong style='color:#c8f55a'>all-MiniLM-L6-v2</strong> and indexed with FAISS.</p>
        </div>
        """, unsafe_allow_html=True)

        show_n = st.slider("Preview first N chunks", 1, min(20, len(st.session_state.chunks)), 5)
        for i, chunk in enumerate(st.session_state.chunks[:show_n]):
            st.markdown(f"""
            <div class='chunk-preview'>
                <span style='color:#c8f55a; font-family: Space Mono, monospace; font-size:0.75rem;'>
                CHUNK {i+1:02d}</span><br><br>
                {chunk[:300]}{'...' if len(chunk) > 300 else ''}
            </div>
            """, unsafe_allow_html=True)
