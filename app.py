import pickle
import random
import time
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from google import genai
from google.genai import errors, types
from sentence_transformers import SentenceTransformer

# =========================
# Fixed configuration (no tuning UI)
# =========================
BASE_DIR = Path(__file__).parent
INDEX_PATH = BASE_DIR / "coal.faiss"
META_PATH = BASE_DIR / "coal.pkl"

EMBED_MODEL = "all-MiniLM-L6-v2"  # must match the model used to build coal.faiss (dim 384)
TOP_K = 3
PRIMARY_MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODEL = "gemini-3.1-flash-lite"  # used if the primary stays rate-limited

MAX_RETRIES = 5
BASE_DELAY = 2.0   # seconds
MAX_DELAY = 30.0   # seconds
RETRYABLE_CODES = {429, 500, 503, 504}


# =========================
# Cached resources (loaded once, not rebuilt on every rerun)
# =========================
@st.cache_resource(show_spinner="Loading index and embedding model…")
def load_resources():
    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    embedder = SentenceTransformer(EMBED_MODEL)
    return index, meta["texts"], meta["sources"], embedder


@st.cache_resource
def get_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


# =========================
# Retrieval
# =========================
def retrieve(query: str, index, texts, sources, embedder, k: int = TOP_K):
    q = embedder.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(q, min(k, len(texts)))
    return [(texts[i], sources[i], float(d)) for d, i in zip(D[0], I[0]) if i >= 0]


# =========================
# Gemini call with exponential backoff
# =========================
def generate_with_backoff(client, model: str, system_prompt: str, query: str) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=query,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.2,
                    max_output_tokens=1024,
                ),
            )
            return resp.text or "I couldn't generate an answer."
        except errors.APIError as e:
            if e.code not in RETRYABLE_CODES or attempt == MAX_RETRIES - 1:
                raise
            delay = min(MAX_DELAY, BASE_DELAY * (2 ** attempt)) + random.uniform(0, 1)
            time.sleep(delay)  # 2s, 4s, 8s, 16s (+ jitter)


def ask_gemini(client, system_prompt: str, query: str) -> str:
    try:
        return generate_with_backoff(client, PRIMARY_MODEL, system_prompt, query)
    except errors.APIError as e:
        if e.code in RETRYABLE_CODES:  # still exhausted -> try the fallback model
            return generate_with_backoff(client, FALLBACK_MODEL, system_prompt, query)
        raise


# =========================
# UI
# =========================
st.set_page_config(page_title="Coal Mine Accidents Q&A", page_icon="⛏️", layout="wide")
st.title("⛏️ Coal Mine Accidents Q&A")
st.caption("RAG chatbot over DGMS SANKET – Key Evaluation of Trends in Coal Mine Accidents (Gemini + FAISS)")

if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ `GEMINI_API_KEY` not found in Streamlit secrets.")
    st.stop()

index, texts, sources, embedder = load_resources()
client = get_client()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask a question about coal mine accidents…"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):  # loading circle while the LLM generates
            hits = retrieve(query, index, texts, sources, embedder)
            context = "\n\n---\n\n".join(f"[{src}]\n{txt}" for txt, src, _ in hits)
            system_prompt = (
                "You are a helpful assistant answering ONLY from the provided context, "
                "which comes from a DGMS report on coal mine accidents in India. "
                "If the answer isn't in the context, say you don't know. "
                "Be concise and mention page numbers when useful.\n\n"
                f"Context:\n{context}"
            )
            try:
                answer = ask_gemini(client, system_prompt, query)
            except errors.APIError as e:
                answer = f"Sorry, the Gemini API request failed ({e.code}). Please try again in a moment."
            except Exception as e:
                answer = f"Sorry, something went wrong: {e}"

        st.markdown(answer)
        with st.expander("Sources"):
            for txt, src, dist in hits:
                st.markdown(f"**{src}** (L2 distance: `{dist:.3f}`)")

    st.session_state.messages.append({"role": "assistant", "content": answer})
