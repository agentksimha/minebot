"""
RAG agent for the Digital Mine Safety Officer.

Pipeline: dense FAISS + BM25 -> reciprocal-rank fusion -> cross-encoder rerank
          -> Gemini (google-genai) with exponential backoff and a fallback model.

Public API (all blocking - call from a worker thread):
    load_resources()                          warm the index/models (idempotent, thread-safe)
    ask(query)                                strict, cited Q&A over the report
    run_task(task_prompt, retrieval_query)    grounded free-form task (audit report, update analysis)

Gemini failures are RAISED, never returned as text, so callers can avoid caching them.
"""
import os
import pathlib
import pickle
import random
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

# =========================
# Configuration
# =========================
BASE_DIR = pathlib.Path(__file__).resolve().parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"


def _resolve_path(env_var: str, filename: str) -> pathlib.Path:
    """Env override -> vectorstore/<file> -> next to agent.py (first one that exists)."""
    if os.getenv(env_var):
        return pathlib.Path(os.environ[env_var])
    for folder in (VECTORSTORE_DIR, BASE_DIR):
        if (folder / filename).exists():
            return folder / filename
    return VECTORSTORE_DIR / filename  # not found: reported by load_resources()


INDEX_PATH = _resolve_path("FAISS_INDEX_PATH", "coal.faiss")
META_PATH = _resolve_path("FAISS_META_PATH", "coal.pkl")

EMBED_MODEL = "all-MiniLM-L6-v2"  # must match the model used to build the index
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DENSE_K = 30
BM25_K = 30
TAG_K = 30             # tag-matched accident cases injected as a third ranking (guaranteed recall)
RRF_K = 60
FUSED_POOL = 25
LIST_POOL = 40         # candidate pool for "list/some/all" questions
FINAL_K = 6            # excerpts for ordinary Q&A
LIST_K = 10            # excerpts for list / multi-case questions
TASK_K = 10            # excerpts for reports / analysis (broader context)
MIN_RERANK_SCORE = -6.0
TAG_BOOST = 2.0        # added to the reranker score when a chunk's tags match the question's topic
YEAR_BOOST = 1.0
NAME_BOOST = 1.5       # state / mine named in the question

# Free-tier friendly models. Override via env vars if you hit quota issues.
PRIMARY_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.1-flash-lite")

MAX_RETRIES = 5
BASE_DELAY = 2.0       # seconds: 2, 4, 8, 16 (+ jitter)
MAX_DELAY = 30.0
RETRYABLE_CODES = {429, 500, 503, 504}

QA_MAX_TOKENS = 1536
LIST_MAX_TOKENS = 3500
TASK_MAX_TOKENS = 2048

NO_ANSWER = "I don't know based on the provided report."


# =========================
# Lazy, cached resources
# =========================
@dataclass
class _Resources:
    index: Any
    texts: list
    sources: list
    embedder: Any
    reranker: Any
    bm25: Any
    tags: list = None      # per-chunk keyword tags (fire, explosion, ...) written by build_index.py
    kinds: list = None     # "case" | "text" | "table"
    meta: list = None      # per-chunk dict(year, mine, state, ...) for accident cases
    states: set = None
    mines: set = None


_resources = None
_load_lock = threading.Lock()


def tokenize(text: str):
    return re.findall(r"\w+", text.lower())


def _find_docs(obj):
    """Fallback for legacy index.pkl layouts: locate the list of document strings."""
    if isinstance(obj, list) and obj and all(isinstance(x, str) for x in obj):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            found = _find_docs(v)
            if found:
                return found
    if isinstance(obj, tuple):
        for v in obj:
            found = _find_docs(v)
            if found:
                return found
    return None


def _load_metadata(path: pathlib.Path):
    with open(path, "rb") as f:
        meta = pickle.load(f)

    extra = {}
    if isinstance(meta, dict) and "texts" in meta:
        texts = list(meta["texts"])
        sources = list(meta.get("sources") or [])
        extra = {k: meta.get(k) for k in ("tags", "kinds", "meta")}
    else:
        texts = _find_docs(meta)
        sources = []
        if texts is None:
            raise TypeError(f"❌ Could not locate any list of documents inside {path.name}. "
                            "Rebuild the vectorstore.")

    if len(sources) != len(texts):
        sources = [f"document {i + 1}" for i in range(len(texts))]
    n = len(texts)
    tags = extra.get("tags") if extra.get("tags") and len(extra["tags"]) == n else [[] for _ in range(n)]
    kinds = extra.get("kinds") if extra.get("kinds") and len(extra["kinds"]) == n else ["text"] * n
    meta_l = extra.get("meta") if extra.get("meta") and len(extra["meta"]) == n else [{} for _ in range(n)]
    return texts, sources, tags, kinds, meta_l


def load_resources() -> _Resources:
    """Load FAISS index, metadata, embedder, reranker and BM25 once, on first use."""
    global _resources
    if _resources is not None:
        return _resources

    with _load_lock:
        if _resources is not None:
            return _resources

        import faiss
        from rank_bm25 import BM25Okapi
        from sentence_transformers import CrossEncoder, SentenceTransformer

        for p in (INDEX_PATH, META_PATH):
            if not p.exists():
                raise FileNotFoundError(f"❌ {p} not found. Put coal.faiss and coal.pkl in "
                                        f"{VECTORSTORE_DIR} (or set FAISS_INDEX_PATH / FAISS_META_PATH).")

        print("🔄 Loading FAISS index and metadata...")
        index = faiss.read_index(str(INDEX_PATH))
        texts, sources, tags, kinds, meta_l = _load_metadata(META_PATH)
        if index.ntotal != len(texts):
            raise ValueError(f"❌ Index has {index.ntotal} vectors but metadata has {len(texts)} "
                             "documents - they were not built together.")

        print("🔄 Loading embedding + rerank models...")
        embedder = SentenceTransformer(EMBED_MODEL)
        reranker = CrossEncoder(RERANK_MODEL, max_length=512)

        print("🔄 Building BM25 index...")
        bm25 = BM25Okapi([tokenize(t) for t in texts])

        states = {m["state"].lower() for m in meta_l if m.get("state")}
        mines = {m["mine"].lower() for m in meta_l if m.get("mine") and len(m["mine"]) >= 4}
        _resources = _Resources(index, texts, sources, embedder, reranker, bm25,
                                tags, kinds, meta_l, states, mines)
        n_tagged = sum(1 for t in tags if t)
        if not any(k == "summary" for k in kinds):
            print(f"⚠️ {INDEX_PATH} has NO summary chunks - it is an old index. Re-run build_index.py "
                  "(it now writes to vectorstore/) and restart.")
        print(f"✅ RAG resources ready ({len(texts)} chunks, {n_tagged} tagged accident chunks).")
    return _resources


_client = None


def _get_client():
    global _client
    if _client is None:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("❌ GEMINI_API_KEY environment variable is missing!")
        _client = genai.Client(api_key=key)
    return _client


# =========================
# Query understanding
# =========================
# topic -> (regex on the question, words appended to the lexical/dense query)
TOPICS = {
    "fire": (r"\bfires?\b|\bburn\w*|\bblaze|\bignit\w*|spontaneous heating|\bflames?\b",
             "fire burn burns burnt injuries caught fire ignite ignition flames spontaneous heating firefighting"),
    "explosion": (r"explo\w+|\bblast\w*|misfire|fire-?damp",
                  "explosion blast exploded misfire explosive"),
    "electrical": (r"electr\w+|\bshock\b|\bvoltage\b",
                   "electric shock electrocuted electrical cable voltage"),
    "roof_fall": (r"\broof\b|side fall|\brib\b|highwall|fall of ground|\bsidewall",
                  "roof fall side fall rib highwall coal sandstone"),
    "transport_hemm": (r"tipper|dumper|dozer|\btruck|excavator|\bhemm\b|conveyor|haulage|winding|\bcage\b",
                       "tipper dumper dozer truck excavator HEMM conveyor"),
    "gas": (r"\bgas(?:es)?\b|suffocat|asphyx|noxious|methane",
            "gas suffocation asphyxiation noxious gases"),
    "water": (r"inundat|drown\w*|irruption|flood\w*",
              "water inundation drowned irruption flood"),
    "blasting": (r"blasting|shot[- ]?fir\w*|drilling",
                 "blasting shotfirer shots drilling explosive"),
}
TOPIC_RES = {k: re.compile(v[0], re.I) for k, v in TOPICS.items()}
EXTRA_EXPANSIONS = [
    (re.compile(r"precaution|prevent|avoid|safety measure|measures|could have|should have|might have", re.I),
     "could have been averted regulation required provisions Had"),
    (re.compile(r"injur\w*|hurt|victim|killed|died|death|fatal", re.I),
     "injuries injured killed persons succumbed hospital"),
    (re.compile(r"\bcause[sd]?\b|why|reason", re.I),
     "cause resulted while"),
    (RANK_RE := re.compile(r"\bsaf(?:e|er|est)\b|\bleast\b|\bfewest\b|\blowest\b|less(?:er)? number|\bminimum\b|"
                           r"\bbest\b|\brank\w*|\bworst\b|unsafe|dangerous", re.I),
     "company-wise subsidiary coalfields limited state wise fatal accidents fatalities summary"),
]
# a ranking question that is about places / organisations (not about one accident topic)
ENTITY_RE = re.compile(r"\bmines?\b|compan\w+|\bowners?\b|subsidiar\w+|coalfield\w*|\bstates?\b|\bwhere\b|\bsites?\b", re.I)
LIST_RE = re.compile(r"\b(some|few|several|list|all|every|examples?|cases|incidents|which mines|mines where|"
                     r"show me|give me|names? of|different|various|top|safer|safest|fewest|least)\b", re.I)
STAT_RE = re.compile(r"how many|number of|\bcount\b|\btotal\b|trend|percent\w*|statistic\w*|per year|"
                     r"year[- ]wise|compare|comparison|increase|decrease|highest|lowest|maximum|minimum", re.I)
YEAR_RE = re.compile(r"\b(20(?:1[6-9]|2[0-2]))\b")


def analyse_query(query: str):
    topics = [k for k, rx in TOPIC_RES.items() if rx.search(query)]
    extra = [TOPICS[k][1] for k in topics]
    extra += [words for rx, words in EXTRA_EXPANSIONS if rx.search(query)]
    expanded = (query + " " + " ".join(extra)).strip()
    is_list = bool(LIST_RE.search(query))
    return topics, expanded, is_list


# =========================
# Retrieval: dense + BM25 + tag list -> RRF -> cross-encoder rerank (+ metadata boosts)
# =========================
def reciprocal_rank_fusion(rankings, k: int = RRF_K):
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def _dense_ids(r, text, k):
    q = r.embedder.encode([text], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    _, I = r.index.search(q, min(k, len(r.texts)))
    return [int(i) for i in I[0] if i >= 0]


def retrieve(query: str, final_k: int = FINAL_K, is_list: bool = None):
    """Return [(text, source, score), ...] best-first. `score` includes metadata boosts."""
    r = load_resources()
    topics, expanded, auto_list = analyse_query(query)
    if is_list is None:
        is_list = auto_list
    if is_list:
        final_k = max(final_k, LIST_K)
    pool = LIST_POOL if is_list else FUSED_POOL

    rankings = [_dense_ids(r, query, DENSE_K)]
    if expanded != query:
        rankings.append(_dense_ids(r, expanded, DENSE_K))

    bm25_scores = r.bm25.get_scores(tokenize(expanded))
    rankings.append([int(i) for i in np.argsort(bm25_scores)[::-1][:BM25_K] if bm25_scores[i] > 0])

    # Guaranteed recall for topic questions ("fire accidents"): every accident chunk whose
    # keyword tags match the topic is a candidate, ordered by lexical relevance.
    # Statistics questions ("how many ... in 2018") should be answered from tables, so topic tags
    # are only used for case-style questions.
    topic_set = set(topics) if not STAT_RE.search(query) else set()
    tagged = [i for i, t in enumerate(r.tags or []) if r.kinds[i] == "case" and topic_set.intersection(t)] \
        if topic_set else []
    if tagged:
        tagged.sort(key=lambda i: bm25_scores[i], reverse=True)
        rankings.append(tagged[:TAG_K])

    fused_ids = reciprocal_rank_fusion(rankings)[:pool]
    if not fused_ids:
        return []

    ce = r.reranker.predict([(query, r.texts[i]) for i in fused_ids])
    years = {int(y) for y in YEAR_RE.findall(query)}
    q_low = query.lower()
    scored = []
    for i, s in zip(fused_ids, ce):
        s = float(s)
        if topic_set and r.kinds[i] == "case" and topic_set.intersection(r.tags[i]):
            s += TAG_BOOST
        m = r.meta[i] if r.meta else {}
        if m:
            if years and m.get("year") in years:
                s += YEAR_BOOST
            if (m.get("state") and m["state"].lower() in q_low) or (m.get("mine") and len(m["mine"]) >= 4 and m["mine"].lower() in q_low):
                s += NAME_BOOST
        scored.append((s, i))
    scored.sort(reverse=True)
    hits = [(r.texts[i], r.sources[i], s) for s, i in scored[:final_k]]

    # Ranking questions ("safer mines", "fewest accidents"): the reranker buries the computed summary
    # chunks, so put the relevant ones into the context directly instead of hoping they rank.
    if RANK_RE.search(query) and ENTITY_RE.search(query):
        wanted = ["owner", "state"] + (["mine"] if re.search(r"\bmines?\b|\bwhere\b|\bsites?\b", query, re.I) else [])
        picked, per_kind = [], {}
        for i, k in enumerate(r.kinds):
            if k != "summary":
                continue
            sk = (r.meta[i] or {}).get("summary")
            if sk in wanted and per_kind.get(sk, 0) < (2 if sk == "mine" else 2):
                per_kind[sk] = per_kind.get(sk, 0) + 1
                picked.append((r.texts[i], r.sources[i], 10.0))
        if picked:
            hits = picked + hits[:5]
    return hits


# =========================
# Prompts
# =========================
QA_TEMPLATE = """You are a precise analyst answering questions about the DGMS SANKET report \
"Key Evaluation of Trends in Coal Mine Accidents" (India, 2013-2022).

HOW THE REPORT IS ORGANISED (so you read the excerpts correctly)
- "ACCIDENT CASE" excerpts are DGMS fatal-accident write-ups (2016-2022). Each has: date, mine, owner, \
district, state, persons killed, a CAUSE category, "WHAT HAPPENED / CAUSE" (the narrative, which also \
describes injuries - including burn injuries and other persons hurt) and "PRECAUTIONS THAT COULD HAVE \
PREVENTED IT" (DGMS's "Had... this accident could have been averted" findings: the lapses, and the \
regulations/precautions that were NOT followed).
- "TABLE" and plain text excerpts hold statistics and analysis; "Major accident" rows list events with 4+ deaths.
- "SUMMARY" excerpts are counts computed from the case write-ups (fatal cases per owner company, state or mine, \
fewest first). Their source label reads "computed summary (...)".

RULES
1. Use ONLY the information inside <context>. Never use outside knowledge, and never guess.
2. Answer as much as the context supports. Give every case you can find that matches the question \
(do not stop at one). If only part of the question is covered, answer that part and simply leave the rest \
out. NEVER write "I don't know", "I do not know" or "{no_answer}" anywhere in an answer that contains any \
information - not at the end, not as a closing remark. Use that exact sentence ONLY as the entire reply when \
NOTHING in the context is relevant.
3. When asked for "precautions they might have taken / should have taken", answer from the PRECAUTIONS \
section and phrase it as what DGMS found should have been done. Do not claim what precautions a mine \
actually had in place unless an excerpt says so. If a case says the report lists no precaution findings, say so.
4. For "injured persons": give the persons killed as listed, and any injuries/other injured persons \
described in the narrative (e.g. burn injuries to others). Do not invent counts.
5. Numbers, years, percentages and names must be copied exactly as written in the context. Do not \
round or extrapolate unless asked to calculate; if you calculate, show the values and the arithmetic.
6. If excerpts conflict or are ambiguous (different years or units), say so explicitly.
7. Cite every factual statement inline as [file - page N] (copy the page label of the excerpt; for \
accident cases add the case number, e.g. [file - page 63 | case 29]).
9. For "safer / safest / fewest accidents" questions: the report only lists accidents, so a mine, company or \
state with zero accidents never appears and nothing can be called "safe". Use the SUMMARY excerpts: name the \
companies, states or mines with the FEWEST fatal cases, show their counts, and cite the summary label \
(e.g. [file - computed summary (owner companies)]). State in one sentence that this ranks reported fatal \
accidents only (2016-2022), not overall safety, and is not adjusted for mine size or production. \
Do NOT answer "{no_answer}" when SUMMARY excerpts are present.
8. For multi-case answers use one short block per case: Mine / owner / state / date; persons killed and \
injuries; cause (what happened); precautions that could have prevented it. Start with a one-line summary \
of how many matching cases you are showing. Be concise otherwise; no preamble.
10. Earlier turns of this conversation may precede the question. Use them only to understand follow-ups \
("it", "that mine", "the second case", "what about 2019?"); you may reuse facts you already stated in \
earlier answers, but any NEW fact must come from <context>.

<context>
{context}
</context>"""

TASK_TEMPLATE = """You are a mining safety analyst for the Digital Mine Safety Officer system.

The <context> below holds excerpts from the DGMS SANKET report "Key Evaluation of Trends in Coal \
Mine Accidents" (India). It is your only source of historical accident data.

RULES
1. Follow the task instructions and the required output format exactly.
2. Every statistic, count, year, percentage or name about historical accidents must come from \
<context> and be copied exactly. Never invent or estimate figures. If the context lacks the data \
for a point, write "Not available in the report." for that point instead of guessing.
3. Cite report facts inline as [file - page N] using the excerpt labels. Do not cite for your own \
classification or judgement of newly supplied text.

<context>
{context}
</context>"""


def _format_context(hits):
    if not hits:
        return "(no relevant excerpts found)"
    return "\n\n".join(f'<excerpt source="{src}">\n{txt}\n</excerpt>' for txt, src, _ in hits)


def build_qa_prompt(hits):
    return QA_TEMPLATE.format(no_answer=NO_ANSWER, context=_format_context(hits))


def build_task_prompt(hits):
    return TASK_TEMPLATE.format(context=_format_context(hits))


# =========================
# Gemini call with exponential backoff
# =========================
def generate_with_backoff(client, model, system_prompt, contents, temperature=0.0,
                          max_tokens=QA_MAX_TOKENS):
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            return resp.text or ""
        except errors.APIError as e:
            if e.code not in RETRYABLE_CODES or attempt == MAX_RETRIES - 1:
                raise
            delay = min(MAX_DELAY, BASE_DELAY * (2 ** attempt)) + random.uniform(0, 1)
            print(f"⏳ Gemini {e.code} on {model}, retrying in {delay:.1f}s "
                  f"(attempt {attempt + 1}/{MAX_RETRIES})...")
            time.sleep(delay)


def ask_gemini(system_prompt: str, contents, temperature=0.0, max_tokens=QA_MAX_TOKENS) -> str:
    """Primary model with backoff, then the fallback model. Raises on failure or empty output."""
    client = _get_client()
    try:
        out = generate_with_backoff(client, PRIMARY_MODEL, system_prompt, contents, temperature, max_tokens)
    except errors.APIError as e:
        if e.code not in RETRYABLE_CODES:
            raise
        print(f"⚠️ {PRIMARY_MODEL} still failing ({e.code}); trying {FALLBACK_MODEL}...")
        out = generate_with_backoff(client, FALLBACK_MODEL, system_prompt, contents, temperature, max_tokens)

    if not out.strip():
        raise RuntimeError("Gemini returned an empty response.")
    return out.strip()


# =========================
# Public pipelines
# =========================
# =========================
# Conversation memory + answer cleaning
# =========================
MAX_HISTORY_MESSAGES = 8       # last 4 exchanges are sent to the model
MAX_HISTORY_CHARS = 1500       # per earlier message (long answers are truncated)

FOLLOWUP_RE = re.compile(
    r"\b(it|its|that|those|these|this|they|them|their|he|she|his|her|there|same|above|previous|earlier|"
    r"former|latter|first|second|third|last|more|another|also|else|other|further)\b|^\s*(and|also|what about|how about|why)\b",
    re.I)

# a trailing "I don't know ..." sentence/line (optionally wrapped in markdown) at the very end of an answer
_IDK_TAIL = re.compile(r"[\s*_>\-]*I\s+(?:don['’]?t|do\s+not)\s+know\b[^\n]*$", re.I)


def clean_answer(text: str) -> str:
    """Remove a closing 'I don't know ...' from an answer that already contains real content."""
    out = text.strip()
    for _ in range(3):  # the model sometimes repeats it
        stripped = _IDK_TAIL.sub("", out).rstrip()
        if stripped == out:
            break
        out = stripped
    if len(out) < 40:   # nothing of substance besides the phrase: keep the genuine refusal
        return text.strip()
    return out


def _trim_history(history):
    """Keep the last N messages, starting with a user turn and strictly alternating roles."""
    msgs = []
    for m in (history or [])[-MAX_HISTORY_MESSAGES:]:
        role = "user" if m.get("role") == "user" else "model"
        content = str(m.get("content", "")).strip()
        if not content:
            continue
        if len(content) > MAX_HISTORY_CHARS:
            content = content[:MAX_HISTORY_CHARS] + " ..."
        if msgs and msgs[-1][0] == role:          # merge consecutive same-role turns
            msgs[-1] = (role, msgs[-1][1] + "\n" + content)
        else:
            msgs.append((role, content))
    while msgs and msgs[0][0] != "user":
        msgs.pop(0)
    return msgs


def _build_contents(history, query: str):
    """Gemini multi-turn contents: earlier turns + the new question."""
    contents = [types.Content(role=r, parts=[types.Part(text=t)]) for r, t in _trim_history(history)]
    if contents and contents[-1].role == "user":  # keep strict alternation before the new user turn
        contents.pop()
    contents.append(types.Content(role="user", parts=[types.Part(text=f"Question: {query}")]))
    return contents


def _standalone_query(query: str, history) -> str:
    """Rewrite a follow-up ('what precautions?') into a self-contained search query.
    One LLM call, single attempt; falls back to previous question + this one."""
    msgs = _trim_history(history)
    if not msgs:
        return query
    short = len(tokenize(query)) <= 10
    if not (short or FOLLOWUP_RE.search(query)):
        return query                                   # already self-contained: no extra API call
    last_user = next((t for r, t in reversed(msgs) if r == "user"), "")
    fallback = f"{last_user[:300]} {query}".strip()
    try:
        transcript = "\n".join(f"{'User' if r == 'user' else 'Assistant'}: {t[:400]}" for r, t in msgs[-4:])
        resp = _get_client().models.generate_content(
            model=PRIMARY_MODEL,
            contents=f"{transcript}\n\nLatest user message: {query}",
            config=types.GenerateContentConfig(
                system_instruction=("Rewrite the latest user message as ONE standalone search question about "
                                    "coal mine accidents in India, resolving pronouns and references (it, that "
                                    "mine, the second case...) from the conversation. Keep names, years and "
                                    "places exactly. Output only the question."),
                temperature=0.0, max_output_tokens=120),
        )
        rewritten = (resp.text or "").strip().strip('"').splitlines()[0] if resp.text else ""
        return rewritten[:400] if len(rewritten) >= 8 else fallback
    except Exception as e:
        print(f"⚠️ follow-up rewrite failed ({type(e).__name__}); using fallback query")
        return fallback


# =========================
# Public pipelines
# =========================
def ask(query: str, history=None) -> str:
    """Strict Q&A grounded in the report, with inline citations.
    `history` = [{"role": "user"|"assistant", "content": str}, ...] earlier turns of this chat (optional)."""
    search_query = _standalone_query(query, history) if history else query
    topics, _, is_list = analyse_query(search_query)
    hits = retrieve(search_query, LIST_K if is_list else FINAL_K, is_list)
    if not hits:
        return f"{NO_ANSWER} I couldn't find a relevant passage."
    # Only give up when nothing scored decently AND no topic-tagged case was retrieved.
    if hits[0][2] < MIN_RERANK_SCORE and not topics:
        return f"{NO_ANSWER} I couldn't find a relevant passage."
    tokens = LIST_MAX_TOKENS if is_list else QA_MAX_TOKENS
    contents = _build_contents(history, query) if history else f"Question: {query}"
    return clean_answer(ask_gemini(build_qa_prompt(hits), contents, max_tokens=tokens))


def run_task(task_prompt: str, retrieval_query: str) -> str:
    """
    Grounded free-form task (audit report, update analysis). Retrieval uses the short
    `retrieval_query` - not the long task prompt, which would swamp the retriever.
    """
    hits = retrieve(retrieval_query, TASK_K)
    return ask_gemini(build_task_prompt(hits), task_prompt, temperature=0.2, max_tokens=TASK_MAX_TOKENS)


# ---------------------------
# CLI testing
# ---------------------------
if __name__ == "__main__":
    while True:
        q = input("\nAsk me anything about mining (prefix with '?' to only show retrieval): ")
        if q.startswith("?"):
            for txt, srcl, sc in retrieve(q[1:].strip()):
                print(f"\n[{sc:.2f}] {srcl}\n{txt[:300]}...")
            continue
        print("\n🔍 Searching...")
        print("💬 Answer:", ask(q))
