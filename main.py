import asyncio
import hashlib
import hmac
import io
import os
import re
import secrets
import sqlite3
import textwrap
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta, timezone

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

import agent  # light import: heavy models/index load lazily inside agent.load_resources()
from rss_feed import fetch_dgms_updates

load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("❌ GEMINI_API_KEY environment variable is missing!")

# Only let one RAG/Gemini call run at a time from this process. Free-tier RPM
# caps (5-15 depending on model) are easy to blow through with concurrent
# requests - this turns "5 at once" into "5, one after another".
GEMINI_SEMAPHORE = asyncio.Semaphore(1)

MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "12"))  # messages loaded per chat turn

AUDIT_TIMEOUT = float(os.getenv("AUDIT_TIMEOUT", "60"))  # seconds (covers backoff + fallback)
AUTH_TOKEN_TTL_DAYS = int(os.getenv("AUTH_TOKEN_TTL_DAYS", "30"))
PASSWORD_ITERATIONS = 310_000

# Bump when the pipeline/prompt changes so stale answers from older
# architectures sitting in rag_cache.db are never served.
CACHE_VERSION = "case-v4"

# -------------------------------------------------------
# FastAPI Setup
# -------------------------------------------------------
app = FastAPI(title="Digital Mine Safety Officer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Trigger resource loading in background after server starts."""
    asyncio.create_task(_background_warmup())


async def _background_warmup():
    """Loads index + embedder + reranker + BM25 after a short delay, without blocking startup."""
    await asyncio.sleep(5)  # let server finish booting first
    print("🔄 Background warmup starting...")
    try:
        await asyncio.to_thread(agent.load_resources)
        print("✅ Warmup complete.")
    except Exception as e:  # first real request will retry and surface the error
        print(f"❌ Warmup failed: {e}")


print("🚀 FastAPI imported — server starting instantly...")


# -------------------------------------------------------
# In-Memory L1 Cache (true LRU)
# -------------------------------------------------------
L1_CACHE_SIZE = 100
lru_cache_store: "OrderedDict[str, str]" = OrderedDict()


def get_from_l1(key):
    if key in lru_cache_store:
        lru_cache_store.move_to_end(key)
        return lru_cache_store[key]
    return None


def set_to_l1(key, response):
    lru_cache_store[key] = response
    lru_cache_store.move_to_end(key)
    while len(lru_cache_store) > L1_CACHE_SIZE:
        lru_cache_store.popitem(last=False)


# -------------------------------------------------------
# SQLite L2 Cache
# -------------------------------------------------------
DB_PATH = os.getenv("DB_PATH", "rag_cache.db")
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS cache (
    query TEXT PRIMARY KEY,
    response TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()


# -------------------------------------------------------
# Authentication (accounts + revocable bearer sessions)
# -------------------------------------------------------
conn.execute("""
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.execute("""
CREATE TABLE IF NOT EXISTS auth_tokens (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
)
""")
conn.execute("CREATE INDEX IF NOT EXISTS idx_auth_tokens_user ON auth_tokens(user_id)")
conn.commit()


def _normalize_email(value: str) -> str:
    email = str(value or "").strip().lower()
    if len(email) > 254 or not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
        raise HTTPException(status_code=422, detail="Enter a valid email address.")
    return email


def _validate_password(value: str):
    password = str(value or "")
    if len(password) < 8:
        raise HTTPException(status_code=422, detail="Password must be at least 8 characters.")
    if len(password) > 256:
        raise HTTPException(status_code=422, detail="Password is too long.")
    return password


def _hash_password(password: str, salt: bytes = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS)
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt.hex()}${digest.hex()}"


def _verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations, salt_hex, expected_hex = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations)
        )
        return hmac.compare_digest(actual, bytes.fromhex(expected_hex))
    except (TypeError, ValueError):
        return False


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _public_user(row) -> dict:
    return {"id": row[0], "name": row[1], "email": row[2]}


def _issue_token(user_id: str) -> str:
    token = secrets.token_urlsafe(48)
    expires_at = datetime.now(timezone.utc) + timedelta(days=AUTH_TOKEN_TTL_DAYS)
    with conn:
        conn.execute("DELETE FROM auth_tokens WHERE expires_at <= ?", (datetime.now(timezone.utc).isoformat(),))
        conn.execute(
            "INSERT INTO auth_tokens (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
            (_token_hash(token), user_id, expires_at.isoformat()),
        )
    return token


def _bearer_token(authorization: str = Header(default="")) -> str:
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


def current_user(token: str = Depends(_bearer_token)) -> dict:
    row = conn.execute(
        """SELECT u.id, u.name, u.email, t.expires_at
           FROM auth_tokens t JOIN users u ON u.id = t.user_id
           WHERE t.token_hash = ?""",
        (_token_hash(token),),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=401, detail="Your session is invalid. Please sign in again.")
    try:
        expires_at = datetime.fromisoformat(row[3])
    except ValueError:
        expires_at = datetime.min.replace(tzinfo=timezone.utc)
    if expires_at <= datetime.now(timezone.utc):
        with conn:
            conn.execute("DELETE FROM auth_tokens WHERE token_hash = ?", (_token_hash(token),))
        raise HTTPException(status_code=401, detail="Your session has expired. Please sign in again.")
    return {"id": row[0], "name": row[1], "email": row[2]}


def get_from_l2(key):
    cursor.execute("SELECT response FROM cache WHERE query = ?", (key,))
    row = cursor.fetchone()
    return row[0] if row else None


def set_to_l2(key, response):
    cursor.execute(
        "INSERT OR REPLACE INTO cache (query, response, timestamp) VALUES (?, ?, CURRENT_TIMESTAMP)",
        (key, response),
    )
    conn.commit()


# -------------------------------------------------------
# Chat history (SQLite) - sessions + messages
# -------------------------------------------------------
conn.execute("""
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT 'default',
    title TEXT NOT NULL DEFAULT 'New chat',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.execute("""
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id, updated_at)")
conn.commit()


def _make_title(text: str) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    return (t[:57] + "...") if len(t) > 60 else (t or "New chat")


def db_create_session(user_id: str, title: str = "New chat") -> str:
    sid = uuid.uuid4().hex
    with conn:
        conn.execute("INSERT INTO sessions (id, user_id, title) VALUES (?, ?, ?)", (sid, user_id, title))
    return sid


def db_get_session(session_id, user_id: str):
    """Returns the session row only if it exists AND belongs to this user_id."""
    if not session_id:
        return None
    row = conn.execute(
        "SELECT id, user_id, title, created_at, updated_at FROM sessions WHERE id = ? AND user_id = ?",
        (str(session_id), user_id),
    ).fetchone()
    return dict(zip(("id", "user_id", "title", "created_at", "updated_at"), row)) if row else None


def db_history(session_id: str, limit: int = MAX_HISTORY_MESSAGES):
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]


def db_add_exchange(session_id: str, user_text: str, assistant_text: str):
    with conn:
        conn.execute("INSERT INTO messages (session_id, role, content) VALUES (?, 'user', ?)", (session_id, user_text))
        conn.execute("INSERT INTO messages (session_id, role, content) VALUES (?, 'assistant', ?)",
                     (session_id, assistant_text))
        conn.execute("UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (session_id,))


def db_list_sessions(user_id: str, limit: int = 100):
    rows = conn.execute(
        """SELECT s.id, s.title, s.created_at, s.updated_at,
                  (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id)
           FROM sessions s WHERE s.user_id = ? ORDER BY s.updated_at DESC, s.rowid DESC LIMIT ?""",
        (user_id, limit),
    ).fetchall()
    return [{"id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3], "message_count": r[4]} for r in rows]


def db_messages(session_id: str):
    rows = conn.execute(
        "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id", (session_id,)
    ).fetchall()
    return [{"role": r, "content": c, "created_at": t} for r, c, t in rows]


def db_delete_session(session_id: str):
    with conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


def db_rename_session(session_id: str, title: str):
    with conn:
        conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (_make_title(title), session_id))


# -------------------------------------------------------
# Cached RAG calls
# -------------------------------------------------------
def _cache_key(kind: str, text: str) -> str:
    return f"{CACHE_VERSION}|{kind}|{text.strip()}"


async def _cached(key: str, fn, *args) -> str:
    result = get_from_l1(key)
    if result:
        return result

    result = get_from_l2(key)
    if result:
        set_to_l1(key, result)
        return result

    # Compute - serialized against free-tier RPM limits. Errors propagate
    # (and are therefore never cached).
    async with GEMINI_SEMAPHORE:
        result = await asyncio.to_thread(fn, *args)
    result = str(result).strip()

    if result:
        set_to_l1(key, result)
        set_to_l2(key, result)
    return result


async def cached_ask(query: str, history=None) -> str:
    """Strict, cited Q&A over the report.
    Follow-up turns depend on the conversation, so they bypass the shared cache (never read or written)."""
    if history:
        async with GEMINI_SEMAPHORE:
            result = await asyncio.to_thread(agent.ask, query, history)
        return str(result).strip()
    return await _cached(_cache_key("qa", query), agent.ask, query)


async def cached_task(task_prompt: str, retrieval_query: str) -> str:
    """Grounded free-form task (report / update analysis); retrieval uses a short query."""
    return await _cached(_cache_key("task", task_prompt), agent.run_task, task_prompt, retrieval_query)


# -------------------------------------------------------
# Root + QUERY ENDPOINTS
# -------------------------------------------------------
@app.post("/auth/register", status_code=201)
async def register(request: Request):
    data = await request.json()
    name = re.sub(r"\s+", " ", str(data.get("name") or "")).strip()
    if len(name) < 2 or len(name) > 80:
        raise HTTPException(status_code=422, detail="Name must be between 2 and 80 characters.")
    email = _normalize_email(data.get("email"))
    password = _validate_password(data.get("password"))
    user_id = uuid.uuid4().hex
    try:
        with conn:
            conn.execute(
                "INSERT INTO users (id, name, email, password_hash) VALUES (?, ?, ?, ?)",
                (user_id, name, email, _hash_password(password)),
            )
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="An account with this email already exists.")
    return {"token": _issue_token(user_id), "user": {"id": user_id, "name": name, "email": email}}


@app.post("/auth/login")
async def login(request: Request):
    data = await request.json()
    email = _normalize_email(data.get("email"))
    password = str(data.get("password") or "")
    row = conn.execute(
        "SELECT id, name, email, password_hash FROM users WHERE email = ?", (email,)
    ).fetchone()
    if not row or not _verify_password(password, row[3]):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")
    return {"token": _issue_token(row[0]), "user": _public_user(row)}


@app.get("/auth/me")
async def auth_me(user: dict = Depends(current_user)):
    return {"user": user}


@app.post("/auth/logout")
async def logout(token: str = Depends(_bearer_token)):
    with conn:
        conn.execute("DELETE FROM auth_tokens WHERE token_hash = ?", (_token_hash(token),))
    return {"ok": True}


@app.get("/")
async def root():
    return {"message": "🦺 Digital Mine Safety Officer API is running!"}


@app.post("/query")
async def query_agent(request: Request, user: dict = Depends(current_user)):
    """Body: {"query": str, "session_id": str (optional)}.
    Without a session_id a new chat is created and its id is returned - send it back on the next
    message so the assistant remembers the conversation."""
    data = await request.json()
    query = str(data.get("query") or "").strip()
    user_id = user["id"]
    session_id = data.get("session_id")
    if not query:
        return {"response": "⚠️ Query is empty.", "session_id": session_id}

    session = db_get_session(session_id, user_id)       # unknown / foreign id -> start a fresh chat
    history = db_history(session["id"]) if session else []

    try:
        response = await cached_ask(query, history)
    except Exception as e:
        print(f"❌ /query failed: {type(e).__name__}: {e}")
        return JSONResponse(
            status_code=503,
            content={"response": "⚠️ The assistant is temporarily unavailable "
                                 "(API rate limit or error). Please try again in a moment.",
                     "session_id": session["id"] if session else None},
        )

    # Persist only successful exchanges, so failed calls never leave half a conversation behind.
    sid = session["id"] if session else db_create_session(user_id, _make_title(query))
    if session and session["title"] == "New chat":
        db_rename_session(sid, query)
    db_add_exchange(sid, query, response)
    return {"response": response, "session_id": sid}


# -------------------------------------------------------
# Chat history endpoints
# -------------------------------------------------------
def _not_found():
    return JSONResponse(status_code=404, content={"error": "Chat session not found."})


@app.get("/sessions")
async def list_sessions(user: dict = Depends(current_user)):
    return {"sessions": db_list_sessions(user["id"])}


@app.post("/sessions")
async def new_session(request: Request, user: dict = Depends(current_user)):
    try:
        data = await request.json()
    except Exception:
        data = {}
    title = _make_title(str((data or {}).get("title") or "New chat"))
    return {"session_id": db_create_session(user["id"], title), "title": title}


@app.get("/sessions/{session_id}")
async def get_session(session_id: str, user: dict = Depends(current_user)):
    session = db_get_session(session_id, user["id"])
    if not session:
        return _not_found()
    return {"id": session["id"], "title": session["title"], "updated_at": session["updated_at"],
            "messages": db_messages(session["id"])}


@app.patch("/sessions/{session_id}")
async def rename_session(session_id: str, request: Request, user: dict = Depends(current_user)):
    data = await request.json()
    session = db_get_session(session_id, user["id"])
    if not session:
        return _not_found()
    db_rename_session(session["id"], str(data.get("title") or "New chat"))
    return {"ok": True}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, user: dict = Depends(current_user)):
    session = db_get_session(session_id, user["id"])
    if not session:
        return _not_found()
    db_delete_session(session["id"])
    return {"ok": True}


# -------------------------------------------------------
# DGMS Updates Endpoint
# -------------------------------------------------------
@app.get("/updates")
async def get_dgms_updates(_user: dict = Depends(current_user)):
    updates = await asyncio.to_thread(fetch_dgms_updates, limit=5)

    async def analyze_update(item):
        title = item.get("title", "")
        link = item.get("link", "")
        published = item.get("published", "")

        try:
            response = await asyncio.to_thread(requests.get, link, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = [p.get_text() for p in soup.find_all("p")]
            content = " ".join(paragraphs[:5]) if paragraphs else "(No text found.)"
        except Exception:
            content = "(Could not fetch full article text.)"

        prompt = (
            f"You are a mining safety officer. Analyze the following DGMS update "
            f"and classify the risk level (High, Medium, Low, or None), and describe "
            f"the hazard type.\n\n"
            f"Title: {title}\nPublished: {published}\nLink: {link}\n"
            f"Content: {content}"
        )
        # Short retrieval query: the report is used as background on similar hazards.
        retrieval_query = f"{title} {content[:200]}".strip()

        try:
            output = await cached_task(prompt, retrieval_query)
        except Exception as e:
            output = f"⚠️ Error: {e}"

        return {
            "title": title,
            "link": link,
            "published": published,
            "danger_analysis": output,
        }

    analyzed_updates = await asyncio.gather(*[analyze_update(u) for u in updates])
    return {"updates": analyzed_updates}


# -------------------------------------------------------
# PDF Audit Report Endpoint
# -------------------------------------------------------
def _pdf_safe(text: str) -> str:
    """Helvetica only covers cp1252; replace anything else (emoji etc.) instead of drawing junk."""
    return text.encode("cp1252", "replace").decode("cp1252")


def _scope_terms(*values) -> str:
    return " ".join(str(v) for v in values if v and not str(v).lower().startswith("all "))


@app.post("/audit_report_pdf")
async def generate_audit_report_pdf(request: Request, _user: dict = Depends(current_user)):
    data = await request.json()
    state = str(data.get("state", "All States"))
    year = str(data.get("year", "All Years"))
    hazard_type = str(data.get("hazard_type", "All Hazards"))

    prompt = f"""
You are a mining safety audit assistant.

Generate a COMPLETE and WELL-FORMATTED report.

State: {state}
Year: {year}
Hazard Type: {hazard_type}

STRICT FORMAT:
1. Summary
2. Incident Count
3. Categories
4. Severity Distribution
5. Root Causes
6. Safety Recommendations
7. Trends

Rules:
- Do NOT stop mid-sentence
- Do NOT repeat text
- Keep each section under 120 words
- Ensure ALL 7 sections are present
- Output clean readable text (no markdown, no JSON)
"""
    retrieval_query = f"coal mine accidents {_scope_terms(hazard_type, state, year)}".strip()

    # Timeout protection (prevents hanging). Failures are shown in the PDF, never cached.
    try:
        report_text = await asyncio.wait_for(cached_task(prompt, retrieval_query), timeout=AUDIT_TIMEOUT)
    except asyncio.TimeoutError:
        report_text = "Report generation timed out. Please try again."
    except Exception as e:
        print(f"❌ /audit_report_pdf failed: {type(e).__name__}: {e}")
        report_text = "Report generation failed. Please retry."

    if not report_text or len(report_text.strip()) < 50:
        report_text = "Report generation failed. Please retry."

    # Fix incomplete sentences (common LLM issue)
    if report_text[-1] not in [".", "!", "?"]:
        report_text += "..."

    # -------------------------------------------------------
    # PDF creation (in memory - nothing written to disk)
    # -------------------------------------------------------
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Mining Safety Audit Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, _pdf_safe(f"State: {state}")[:90])
    c.drawString(50, height - 100, _pdf_safe(f"Year: {year}")[:90])
    c.drawString(50, height - 120, _pdf_safe(f"Hazard Type: {hazard_type}")[:90])

    y = height - 160
    c.setFont("Helvetica", 11)

    for paragraph in _pdf_safe(report_text).split("\n"):
        if not paragraph.strip():
            y -= 8  # paragraph spacing
            continue
        for line in textwrap.wrap(paragraph, width=90):
            c.drawString(60, y, line)
            y -= 15

            if y < 50:
                c.showPage()
                c.setFont("Helvetica", 11)
                y = height - 50

    c.save()

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"Audit_Report_{state}_{year}")[:80]
    return Response(
        content=pdf_buffer.getvalue(),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}.pdf"'},
    )
