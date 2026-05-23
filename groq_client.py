import os
from groq import Groq
import streamlit as st

def get_groq_client():
    """Initialize Groq client using API key from Streamlit secrets or env."""
    api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("❌ GROQ_API_KEY not found. Add it to Streamlit secrets or set as env variable.")
        st.stop()
    return Groq(api_key=api_key)


def ask_question(question: str, context: str) -> str:
    """
    Answer a user question using relevant context chunks via Groq.
    """
    client = get_groq_client()

    system_prompt = """You are an expert document analyst. 
Answer questions based ONLY on the provided document context.
Be accurate, concise and helpful. 
If the answer is not in the context, say "This information is not available in the document."
Format your answer clearly with bullet points or numbered lists when appropriate."""

    user_prompt = f"""Context from document:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    return response.choices[0].message.content.strip()


def summarize_text(text: str, summary_type: str) -> str:
    """
    Generate a summary of the document text.
    """
    client = get_groq_client()

    type_instructions = {
        "Brief (2-3 sentences)": "Write a brief 2-3 sentence summary covering the main topic.",
        "Detailed (key points)": "Write a detailed summary with bullet points covering all key topics, findings, and conclusions.",
        "Executive (business format)": "Write an executive summary with sections: Overview, Key Findings, and Recommendations.",
    }

    instruction = type_instructions.get(summary_type, type_instructions["Brief (2-3 sentences)"])

    user_prompt = f"""Summarize the following document text.

{instruction}

Document text:
{text}

Summary:"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are an expert document summarizer. Be clear, accurate and concise."},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=1024,
    )

    return response.choices[0].message.content.strip()
