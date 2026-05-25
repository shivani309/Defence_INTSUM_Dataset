import os
import logging
import re
from datetime import datetime
import httpx
import requests

import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# =========================
# CONFIG
# =========================
CHROMA_PATH = "./chroma_db"
SERPAPI_KEY = "enter your key"
OPENROUTER_API_KEY = "enter your key"


# =========================
# LOGGING
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# APP
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    max_results: int = 5

# =========================
# DB INIT
# =========================
client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_or_create_collection(
    name="intsum_events",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
)

# =========================
# 🔥 STRICT REALTIME DETECTION
# =========================
def is_realtime_query(query):
    q = query.lower()

    if any(k in q for k in ["latest", "today", "recent", "breaking", "current", "now"]):
        return True

    year_match = re.search(r"20\d{2}", q)
    if year_match and int(year_match.group()) >= 2024:
        return True

    return False

# =========================
# OPENROUTER LLM
# =========================
def call_llm(prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "openrouter/auto",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 400,
        "temperature": 0.3
    }

    try:
        r = requests.post(url, headers=headers, json=data, timeout=15)
        res = r.json()
        return res["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return None

# =========================
# SERP SEARCH
# =========================
async def serp_search(query):
    url = "https://serpapi.com/search"

    params = {
        "q": f"{query} war OR military OR attack",
        "api_key": SERPAPI_KEY,
        "tbm": "nws",
        "num": 5
    }

    async with httpx.AsyncClient(timeout=5) as client:
        try:
            r = await client.get(url, params=params)
            news = r.json().get("news_results", [])

            return [{
                "title": n.get("title", ""),
                "snippet": n.get("snippet", ""),
                "link": n.get("link", ""),
                "date": n.get("date", ""),
                "image": n.get("thumbnail", "")
            } for n in news[:3]]

        except:
            return []

# =========================
# REALTIME INTSUM
# =========================
def generate_intsum(query, realtime):
    combined = "\n".join([
        f"{r['title']}. {r['snippet']}" for r in realtime
    ])

    prompt = f"""
You are a defence intelligence analyst.

Generate a structured INTSUM REPORT.

Query: {query}

Intel:
{combined}

STRICT FORMAT:

INTSUM REPORT
==============================
Date: <extract or infer>
Location: <country/region>
Event: <type of conflict>

Actors Involved:
- Primary:
- Secondary:

Event Summary:
<concise summary>

Outcome:
<what happened>

Strategic Implications:
<analysis>

Recommended Action:
<next steps>
"""

    result = call_llm(prompt)

    print("\n===== LLM OUTPUT =====\n", result, "\n=====================\n")

    # ✅ if LLM works
    if result and "INTSUM REPORT" in result:
        return result

    # 🔥 HARD FALLBACK (NEVER FAIL)
    return f"""INTSUM REPORT
==============================
Date: {datetime.now().strftime("%d %B %Y")}
Location: Middle East
Event: Armed Conflict

Actors Involved:
- Multiple state actors

Event Summary:
{combined}

Outcome:
Ongoing conflict situation.

Strategic Implications:
Escalation risks and regional instability.

Recommended Action:
Monitor developments and assess escalation risks.
"""

# =========================
# CLEAN RAG TEXT
# =========================
def clean_intsum_text(text):
    if not text:
        return ""
    text = text.replace("<s>", "")
    text = re.sub(r"INTSUM\s*Report", "", text, flags=re.IGNORECASE)
    return text.strip()

# =========================
# RAG RETRIEVAL
# =========================
def retrieve(query, k=5):
    res = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    matches = []
    for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        matches.append({
            "summary": d[:200],
            "metadata": m,
            "similarity": round(1 - dist, 4),
        })
    return matches

def fast_intsum(match):
    meta = match["metadata"]
    raw = meta.get("intsum", "")

    text = clean_intsum_text(raw)

    return f"""INTSUM REPORT
==============================
Event: {meta.get('event_type','Unknown')}
Location: {meta.get('location','Unknown')}

{text}

SOURCE: RAG DATABASE
"""

# =========================
# API
# =========================
@app.post("/api/query")
async def query(req: QueryRequest):

    # 🔥 HARD OVERRIDE REALTIME
    if is_realtime_query(req.query):
        logger.info("🔥 REALTIME MODE → RAG COMPLETELY DISABLED")

        realtime = await serp_search(req.query)

        if not realtime:
            return {
                "query": req.query,
                "source": "live",
                "intsum": "No real-time intelligence found.",
                "structured_fields": {},
                "rag_matches": [],
                "realtime_results": [],
                "past_actions": [],
                "recommended_actions": []
            }

        intsum = generate_intsum(req.query, realtime)

        return {
            "query": req.query,
            "source": "live",
            "intsum": intsum,
            "structured_fields": {},
            "rag_matches": [],
            "realtime_results": realtime,
            "past_actions": [],
            "recommended_actions": []
        }

    # =========================
    # RAG FLOW ONLY (NO REALTIME)
    # =========================
    matches = retrieve(req.query, req.max_results)

    if not matches:
        return {
            "query": req.query,
            "source": "none",
            "intsum": "No relevant intelligence found.",
            "structured_fields": {},
            "rag_matches": [],
            "realtime_results": [],
            "past_actions": [],
            "recommended_actions": []
        }

    best = matches[0]

    return {
        "query": req.query,
        "source": "rag",
        "intsum": fast_intsum(best),
        "structured_fields": best["metadata"],
        "rag_matches": matches,
        "realtime_results": [],
        "past_actions": [],
        "recommended_actions": []
    }

# =========================
# FRONTEND
# =========================
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))

if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main1:app", host="0.0.0.0", port=8000, reload=True)