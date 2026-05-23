# 📄 AI Document Assistant

> RAG-powered PDF Q&A and summarization app built with LangChain, FAISS, Groq, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=flat-square)
![Groq](https://img.shields.io/badge/LLM-Groq%20Llama3-orange?style=flat-square)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-green?style=flat-square)

---

## 🚀 Features

- **PDF Upload** — Upload any PDF up to 50MB
- **RAG Pipeline** — Retrieval-Augmented Generation for accurate, context-grounded answers
- **Vector Search** — FAISS-powered semantic similarity search with `all-MiniLM-L6-v2` embeddings
- **Q&A Chat** — Ask unlimited questions about your document
- **Smart Summarization** — Brief, Detailed, or Executive summary modes
- **Chunk Explorer** — Inspect how the document was split and indexed

---
🔗 Live Demo: https://ai-document-assistant-qzqmpgjhwmjwwmvxsdeh7s.streamlit.app
## 🏗️ Architecture

```
PDF Upload
    │
    ▼
Text Extraction (PyMuPDF)
    │
    ▼
Text Chunking (500 words, 50 overlap)
    │
    ▼
Embedding (sentence-transformers / all-MiniLM-L6-v2)
    │
    ▼
FAISS Vector Index
    │
    ▼
Query → Retrieve Top-4 Chunks → Groq Llama-3 → Answer
```

---

## 🛠️ Tech Stack

| Layer | Tool |
|-------|------|
| Frontend | Streamlit |
| LLM | Groq (Llama-3 8B) |
| Embeddings | Sentence-Transformers (all-MiniLM-L6-v2) |
| Vector DB | FAISS |
| PDF Parsing | PyMuPDF |

---

## ⚙️ Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/ai-document-assistant.git
cd ai-document-assistant

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Groq API key
# Create .streamlit/secrets.toml and add:
# GROQ_API_KEY = "your_key_here"

# 5. Run the app
streamlit run app.py
```

---

## 🔑 Get a Free Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up for free
3. Create an API key
4. Add it to `.streamlit/secrets.toml`

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `app.py` as the main file
5. Add `GROQ_API_KEY` in **Secrets** settings
6. Click Deploy!

---

## 📁 Project Structure

```
ai-document-assistant/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .streamlit/
│   ├── config.toml         # Theme configuration
│   └── secrets.toml        # API keys (DO NOT commit)
└── utils/
    ├── pdf_processor.py    # PDF extraction & chunking
    ├── vector_store.py     # FAISS embedding & retrieval
    └── groq_client.py      # Groq LLM API calls
```

---

## 👨‍💻 Author

**Mohammad Idrish** — [LinkedIn](https://linkedin.com) · [GitHub](https://github.com)

Machine Learning Intern @ DRDO | B.Tech IT @ Rajasthan Technical University
