import os
from groq import Groq
import streamlit as st

def get_groq_client():
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found.")
        st.stop()
    return Groq(api_key=api_key)


def ask_question(question: str, context: str) -> str:
    client = get_groq_client()
    system_prompt = "You are an expert document analyst. Answer questions based ONLY on the provided document context. If the answer is not in the context, say this information is not available in the document."
    user_prompt = "Context from document:\n" + context + "\n\nQuestion: " + question + "\n\nAnswer:"
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def summarize_text(text: str, summary_type: str) -> str:
    client = get_groq_client()
    type_instructions = {
        "Brief (2-3 sentences)": "Write a brief 2-3 sentence summary covering the main topic.",
        "Detailed (key points)": "Write a detailed summary with bullet points covering all key topics, findings, and conclusions.",
        "Executive (business format)": "Write an executive summary with sections: Overview, Key Findings, and Recommendations.",
    }
    instruction = type_instructions.get(summary_type, type_instructions["Brief (2-3 sentences)"])
    text = " ".join(text.split()[:800])
    user_prompt = "Summarize the following document text.\n\n" + instruction + "\n\nDocument text:\n" + text + "\n\nSummary:"
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are an expert document summarizer. Be clear, accurate and concise."},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()
