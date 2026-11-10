import json
import pickle
import random
import re
import time
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from google import genai
from google.genai import errors, types
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

# =========================
# Fixed configuration
# =========================
BASE_DIR = Path(__file__).parent
INDEX_PATH = BASE_DIR / "coal.faiss"
META_PATH = BASE_DIR / "coal.pkl"

EMBED_MODEL = "all-MiniLM-L6-v2"  # must match build_index.py
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DENSE_K = 20
BM25_K = 20
RRF_K = 60
FUSED_POOL = 20
FINAL_K = 5
MIN_RERANK_SCORE = -6.0

PRIMARY_MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODEL = "gemini-3.1-flash-lite"
JUDGE_MODEL = PRIMARY_MODEL  # ideally a different/stronger model than the generator

MAX_RETRIES = 5
BASE_DELAY = 2.0
MAX_DELAY = 30.0
RETRYABLE_CODES = {429, 500, 503, 504}

NO_ANSWER = "I don't know based on the provided report."

METRIC_LABELS = {
    "context_precision": "Context precision",
    "context_recall": "Context recall",
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer relevancy",
    "answer_correctness": "Answer correctness",
}


# =========================
# Cached resources
# =========================
def tokenize(text: str):
    return re.findall(r"\w+", text.lower())


@st.cache_resource(show_spinner="Loading index and models…")
def load_resources():
    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    texts, sources = meta["texts"], meta["sources"]
    embedder = SentenceTransformer(EMBED_MODEL)
    reranker = CrossEncoder(RERANK_MODEL, max_length=512)
    bm25 = BM25Okapi([tokenize(t) for t in texts])
    return index, texts, sources, embedder, reranker, bm25


@st.cache_resource
def get_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


# =========================
# Retrieval: dense + BM25 -> RRF -> cross-encoder rerank
# =========================
def reciprocal_rank_fusion(rankings, k: int = RRF_K):
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def retrieve(query, index, texts, sources, embedder, reranker, bm25):
    n = len(texts)

    q = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    _, I = index.search(q, min(DENSE_K, n))
    dense_ids = [int(i) for i in I[0] if i >= 0]

    bm25_scores = bm25.get_scores(tokenize(query))
    bm25_ids = [int(i) for i in np.argsort(bm25_scores)[::-1][:BM25_K] if bm25_scores[i] > 0]

    fused_ids = reciprocal_rank_fusion([dense_ids, bm25_ids])[:FUSED_POOL]

    pairs = [(query, texts[i]) for i in fused_ids]
    ce_scores = reranker.predict(pairs)
    order = np.argsort(ce_scores)[::-1][:FINAL_K]
    return [(texts[fused_ids[o]], sources[fused_ids[o]], float(ce_scores[o])) for o in order]


# =========================
# Prompt
# =========================
SYSTEM_TEMPLATE = """You are a precise analyst answering questions about the DGMS SANKET report \
"Key Evaluation of Trends in Coal Mine Accidents" (India).

RULES
1. Use ONLY the information inside <context>. Never use outside knowledge, and never guess.
2. If the context does not contain enough information to answer, reply exactly: "{no_answer}" \
You may add one short sentence on what is missing.
3. Numbers, years, percentages and names must be copied exactly as written in the context. \
Do not round, convert or extrapolate unless the user asks you to calculate; if you calculate, \
show the values used and the arithmetic.
4. For comparisons, trends or multi-part questions: first list the relevant facts from the context, \
then derive the conclusion from those facts only.
5. If excerpts conflict or are ambiguous (e.g., different years or units), say so explicitly.
6. Cite the source of every factual statement inline as [file - page N], using the labels on the excerpts.
7. Be concise. Start with the direct answer, then the supporting evidence. No preamble.

<context>
{context}
</context>"""


def build_system_prompt(hits):
    context = "\n\n".join(
        f'<excerpt source="{src}">\n{txt}\n</excerpt>' for txt, src, _ in hits
    )
    return SYSTEM_TEMPLATE.format(no_answer=NO_ANSWER, context=context)


# =========================
# Gemini call with exponential backoff
# =========================
def generate_with_backoff(client, model, system_prompt, contents, temperature=0.0,
                          max_tokens=1024, json_mode=False):
    for attempt in range(MAX_RETRIES):
        try:
            cfg = dict(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            if json_mode:
                cfg["response_mime_type"] = "application/json"
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**cfg),
            )
            return resp.text or ""
        except errors.APIError as e:
            if e.code not in RETRYABLE_CODES or attempt == MAX_RETRIES - 1:
                raise
            delay = min(MAX_DELAY, BASE_DELAY * (2 ** attempt)) + random.uniform(0, 1)
            time.sleep(delay)


def ask_gemini(client, system_prompt: str, query: str) -> str:
    contents = f"Question: {query}"
    try:
        out = generate_with_backoff(client, PRIMARY_MODEL, system_prompt, contents)
    except errors.APIError as e:
        if e.code in RETRYABLE_CODES:
            out = generate_with_backoff(client, FALLBACK_MODEL, system_prompt, contents)
        else:
            raise
    return out or "I couldn't generate an answer."


# =========================
# Evaluation metrics (LLM-as-judge, RAGAS-style)
# =========================
JUDGE_SCHEMA = """Return ONLY valid JSON (no markdown) in exactly this shape:
{
  "chunk_relevance": [1, 0, ...],
  "answer_claims": [{"claim": "...", "supported": 1}],
  "answer_relevancy": 0.0,
  "reference_statements": [{"statement": "...", "supported": 1}],
  "answer_correctness": 0.0
}
Use [] for reference_statements and null for answer_correctness when no reference answer is given."""


def build_judge_prompt(query, hits, answer, reference):
    chunks = "\n\n".join(f"[Chunk {i + 1}] ({src})\n{txt}" for i, (txt, src, _) in enumerate(hits))
    ref_block = f"\nREFERENCE (ground-truth) ANSWER:\n{reference}\n" if reference else ""
    ref_task = (
        "4. reference_statements: split the REFERENCE answer into atomic statements; supported = 1 only if "
        "the statement can be attributed to the retrieved chunks, else 0.\n"
        "5. answer_correctness: 0.0-1.0, how well the GENERATED answer agrees with the REFERENCE answer "
        "(facts and numbers must match).\n" if reference else ""
    )
    return (
        "You are a strict, impartial evaluator of a RAG system.\n\n"
        f"QUESTION:\n{query}\n\n"
        f"RETRIEVED CONTEXT CHUNKS (in retrieval order):\n{chunks}\n\n"
        f"GENERATED ANSWER:\n{answer}\n{ref_block}\n"
        "TASKS\n"
        f"1. chunk_relevance: for each of the {len(hits)} chunks, 1 if it contains information useful for "
        "answering the question, else 0. Exactly one item per chunk, in order.\n"
        "2. answer_claims: split the GENERATED answer into atomic factual claims; supported = 1 only if the "
        "claim can be directly inferred from the chunks, else 0.\n"
        "3. answer_relevancy: 0.0-1.0, how directly and completely the answer addresses the question "
        "(ignore whether it is correct).\n"
        f"{ref_task}\n{JUDGE_SCHEMA}"
    )


def parse_json(text: str):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        return json.loads(m.group(0)) if m else {}


def context_precision(relevance):
    """Rank-aware precision (RAGAS Context Precision@K): rewards relevant chunks ranked higher."""
    total_rel = sum(relevance)
    if total_rel == 0:
        return 0.0
    seen, score = 0, 0.0
    for k, v in enumerate(relevance, start=1):
        if v:
            seen += 1
            score += seen / k
    return score / total_rel


def ratio(items):
    vals = [int(bool(x.get("supported", 0))) for x in items if isinstance(x, dict)]
    return (sum(vals) / len(vals)) if vals else None


def clamp01(v):
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return None


def evaluate_rag(client, query, hits, answer, reference=""):
    """Returns a dict of metrics (None = not computable). Uses one judge LLM call."""
    refused = answer.startswith(NO_ANSWER)
    raw = generate_with_backoff(
        client, JUDGE_MODEL, None, build_judge_prompt(query, hits, answer, reference.strip()),
        temperature=0.0, max_tokens=2048, json_mode=True,
    )
    data = parse_json(raw)

    rel = [int(bool(x)) for x in data.get("chunk_relevance", [])][: len(hits)]
    rel += [0] * (len(hits) - len(rel))

    metrics = {
        "chunk_relevance": rel,
        "context_precision": context_precision(rel),
        "context_recall": ratio(data.get("reference_statements") or []) if reference.strip() else None,
        "faithfulness": None if refused else ratio(data.get("answer_claims") or []),
        "answer_relevancy": None if refused else clamp01(data.get("answer_relevancy")),
        "answer_correctness": clamp01(data.get("answer_correctness")) if reference.strip() else None,
    }
    return metrics


def fmt(v):
    return "n/a" if v is None else f"{v:.2f}"


def render_metrics(m):
    if not m:
        return
    with st.expander("📊 RAG metrics", expanded=False):
        cols = st.columns(5)
        for col, key in zip(cols, METRIC_LABELS):
            col.metric(METRIC_LABELS[key], fmt(m.get(key)))
        st.caption(
            f"Retrieval: {m.get('t_retrieval', 0):.2f}s · Generation: {m.get('t_generation', 0):.2f}s · "
            f"Top rerank score: {m.get('top_rerank', float('nan')):.2f}"
        )
        if m.get("error"):
            st.warning(f"Judge evaluation failed: {m['error']}")
        st.caption(
            "LLM-judged (RAGAS-style). Context recall & answer correctness need a reference answer "
            "(sidebar). Faithfulness/relevancy are n/a when the bot refuses to answer."
        )


# =========================
# UI
# =========================
st.set_page_config(page_title="Coal Mine Accidents Q&A", page_icon="⛏️", layout="wide")
st.title("⛏️ Coal Mine Accidents Q&A")
st.caption("RAG chatbot over DGMS SANKET – Key Evaluation of Trends in Coal Mine Accidents "
           "(Hybrid BM25 + FAISS · RRF · Cross-encoder rerank · Gemini)")

if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ `GEMINI_API_KEY` not found in Streamlit secrets.")
    st.stop()

with st.sidebar:
    st.header("Evaluation")
    eval_on = st.toggle("Compute RAG metrics (extra Gemini call)", value=False)
    reference = st.text_area(
        "Reference answer (optional)",
        placeholder="Paste the ground-truth answer for your NEXT question to also get "
                    "context recall and answer correctness.",
        height=140,
    )
    avg_box = st.container()

index, texts, sources, embedder, reranker, bm25 = load_resources()
client = get_client()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_metrics(msg.get("metrics"))

if query := st.chat_input("Ask a question about coal mine accidents…"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    metrics = None
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            t0 = time.perf_counter()
            hits = retrieve(query, index, texts, sources, embedder, reranker, bm25)
            t_retrieval = time.perf_counter() - t0

            t1 = time.perf_counter()
            if not hits or hits[0][2] < MIN_RERANK_SCORE:
                answer = f"{NO_ANSWER} I couldn't find a relevant passage."
            else:
                try:
                    answer = ask_gemini(client, build_system_prompt(hits), query)
                except errors.APIError as e:
                    answer = f"Sorry, the Gemini API request failed ({e.code}). Please try again in a moment."
                except Exception as e:
                    answer = f"Sorry, something went wrong: {e}"
            t_generation = time.perf_counter() - t1

        st.markdown(answer)

        if eval_on and hits:
            with st.spinner("Evaluating…"):
                try:
                    metrics = evaluate_rag(client, query, hits, answer, reference)
                except Exception as e:
                    metrics = {"error": str(e)}
            metrics.update(
                t_retrieval=t_retrieval,
                t_generation=t_generation,
                top_rerank=hits[0][2],
            )

        rel = (metrics or {}).get("chunk_relevance")
        with st.expander("Sources"):
            for i, (txt, src, score) in enumerate(hits):
                mark = "" if not rel else ("✅ " if rel[i] else "❌ ")
                st.markdown(f"{mark}**{src}** (rerank score: `{score:.2f}`)")
                st.caption(txt[:300] + ("…" if len(txt) > 300 else ""))
        render_metrics(metrics)

    st.session_state.messages.append({"role": "assistant", "content": answer, "metrics": metrics})

# Running averages across this session (sidebar)
scored = [m["metrics"] for m in st.session_state.messages if m.get("metrics") and not m["metrics"].get("error")]
if scored:
    with avg_box:
        st.subheader(f"Session averages ({len(scored)} answers)")
        for key, label in METRIC_LABELS.items():
            vals = [m[key] for m in scored if m.get(key) is not None]
            st.metric(label, f"{sum(vals) / len(vals):.2f}" if vals else "n/a")
