"""
Defence INTSUM Intelligence System - FastAPI Backend
Stack: FastAPI + ChromaDB (RAG) + SerpAPI (real-time) + Flan-T5 (generation)
"""

import os
import json
import re
import asyncio
import httpx
from datetime import datetime
from typing import Optional

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import pipeline
import torch

# ──────────────────────────────────────────────
# CONFIG  — set your API keys here or via .env
# ──────────────────────────────────────────────
SERPAPI_KEY   = "56a54208a69a61eccd4ca4d1edd9f15af05db1482d7e0f45fe0face9f98d3b06"
DATA_CSV      = os.path.join(os.path.dirname(__file__), "../data/synthetic_intsum_event_specific.csv")
DATA_JSONL    = os.path.join(os.path.dirname(__file__), "../data/intsum_extraction_finetune_with_geo__1_.jsonl")
CHROMA_PATH   = os.path.join(os.path.dirname(__file__), "../data/chroma_db")

# ──────────────────────────────────────────────
# APP INIT
# ──────────────────────────────────────────────
app = FastAPI(title="Defence INTSUM System", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# MODELS
# ──────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    max_results: Optional[int] = 5

class INTSUMResponse(BaseModel):
    query: str
    source: str            # "rag", "realtime", "hybrid"
    intsum: str
    structured_fields: dict
    past_actions: list
    recommended_actions: list
    rag_matches: list
    realtime_results: list
    generated_at: str

# ──────────────────────────────────────────────
# CHROMADB SETUP
# ──────────────────────────────────────────────
print("🔧 Initialising ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create / load collections
event_collection = chroma_client.get_or_create_collection(
    name="intsum_events",
    embedding_function=embed_fn,
    metadata={"hnsw:space": "cosine"}
)

def load_data_into_chroma():
    """Ingest CSV + JSONL into ChromaDB if empty."""
    if event_collection.count() > 0:
        print(f"✅ ChromaDB already has {event_collection.count()} documents. Skipping ingest.")
        return

    print("📥 Ingesting data into ChromaDB...")
    docs, metas, ids = [], [], []

    # --- CSV (synthetic events with INTSUM) ---
    df = pd.read_csv(DATA_CSV)
    for i, row in df.iterrows():
        note = str(row.get("notes", ""))
        if not note or note == "nan":
            continue
        intsum_text = str(row.get("intsum", "")) if pd.notna(row.get("intsum")) else ""
        # strip <s> token
        intsum_text = intsum_text.replace("<s>", "").strip()

        meta = {
            "source":       "synthetic_csv",
            "event_date":   str(row.get("event_date", "")),
            "location":     str(row.get("location", "")),
            "event_type":   str(row.get("event_type", "")),
            "sub_event_type": str(row.get("sub_event_type", "")),
            "actor1":       str(row.get("actor1", "")),
            "actor2":       str(row.get("actor2", "")) if pd.notna(row.get("actor2")) else "",
            "fatalities":   str(row.get("fatalities", "0")),
            "intsum":       intsum_text[:2000],
        }
        docs.append(note)
        metas.append(meta)
        ids.append(f"csv_{i}")

    # --- JSONL (extraction fine-tune data) ---
    with open(DATA_JSONL) as f:
        for j, line in enumerate(f):
            record = json.loads(line)
            instruction = record.get("instruction", "")
            # extract field note text from instruction
            note_match = re.search(r"Field Note:\n(.+)", instruction, re.DOTALL)
            note = note_match.group(1).strip() if note_match else instruction
            output = record.get("output", {})
            meta = {
                "source":     "jsonl_finetune",
                "event_date": "",
                "location":   str(output.get("location", "")),
                "event_type": str(output.get("event_type", "")),
                "sub_event_type": "",
                "actor1":     str(output.get("actor1", "")),
                "actor2":     str(output.get("actor2", "")) if output.get("actor2") else "",
                "fatalities": "0",
                "latitude":   str(output.get("latitude", "")),
                "longitude":  str(output.get("longitude", "")),
                "intsum":     "",
            }
            docs.append(note)
            metas.append(meta)
            ids.append(f"jsonl_{j}")

    # Batch upsert
    batch_size = 50
    for start in range(0, len(docs), batch_size):
        event_collection.upsert(
            documents=docs[start:start+batch_size],
            metadatas=metas[start:start+batch_size],
            ids=ids[start:start+batch_size]
        )

    print(f"✅ ChromaDB ingested {event_collection.count()} documents")

load_data_into_chroma()

# ──────────────────────────────────────────────
# LLM — Flan-T5-Large (free, CPU-friendly)
# ──────────────────────────────────────────────
print("🤖 Loading TinyLlama (text-generation)...")
device = 0 if torch.cuda.is_available() else -1
generator = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device=device,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
print("✅ TinyLlama loaded")

# ──────────────────────────────────────────────
# RAG RETRIEVAL
# ──────────────────────────────────────────────
def retrieve_from_rag(query: str, n_results: int = 5) -> list:
    results = event_collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    matches = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        matches.append({
            "note":          doc,
            "metadata":      meta,
            "similarity":    round(1 - dist, 4),
            "has_intsum":    bool(meta.get("intsum", "").strip())
        })
    return matches

# ──────────────────────────────────────────────
# SERPAPI REAL-TIME SEARCH
# ──────────────────────────────────────────────
async def search_realtime(query: str, num: int = 5) -> list:
    if SERPAPI_KEY == "YOUR_SERPAPI_KEY_HERE":
        # Return mock data if key not set
        return [{
            "title":   f"[Mock] Security incident: {query}",
            "snippet": "Real-time search disabled. Set SERPAPI_KEY environment variable.",
            "link":    "#",
            "date":    datetime.now().strftime("%Y-%m-%d")
        }]
    url = "https://serpapi.com/search"
    params = {
        "q":       f"{query} India security incident defence",
        "api_key": SERPAPI_KEY,
        "num":     num,
        "hl":      "en",
        "gl":      "in",
        "tbm":     "nws"   # news search
    }
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(url, params=params)
            data = resp.json()
            news = data.get("news_results", data.get("organic_results", []))
            return [{
                "title":   item.get("title", ""),
                "snippet": item.get("snippet", item.get("description", "")),
                "link":    item.get("link", ""),
                "date":    item.get("date", "")
            } for item in news[:num]]
        except Exception as e:
            return [{"title": "Search error", "snippet": str(e), "link": "#", "date": ""}]

# ──────────────────────────────────────────────
# INTSUM GENERATION
# ──────────────────────────────────────────────
def extract_structured_fields(query: str, rag_matches: list, realtime: list) -> dict:
    """Use LLM to extract structured fields from query + context."""
    context = ""
    if rag_matches:
        context = rag_matches[0]["note"]
    elif realtime:
        context = realtime[0].get("snippet", "")

    prompt = (
        f"Extract intelligence fields from this event description. "
        f"Return actor1, actor2, event_type, location, date, fatalities.\n\n"
        f"Event: {query}\nContext: {context}\n\n"
        f"actor1:"
    )
    try:
        result = generator(prompt, max_new_tokens=100, return_full_text=False)[0]["generated_text"]
        # Parse key-value pairs
        fields = {}
        for key in ["actor1", "actor2", "event_type", "location", "date", "fatalities"]:
            match = re.search(rf"{key}[:\s]+([^\n,]+)", result, re.IGNORECASE)
            fields[key] = match.group(1).strip() if match else "Unknown"

        # Try to get lat/lon from best RAG match
        meta = rag_matches[0]["metadata"] if rag_matches else {}
        fields["latitude"]  = meta.get("latitude", "")
        fields["longitude"] = meta.get("longitude", "")
        if not fields["latitude"] and rag_matches:
            # Try from JSONL records
            note_meta = rag_matches[0]["metadata"]
            fields["latitude"]  = note_meta.get("latitude", "N/A")
            fields["longitude"] = note_meta.get("longitude", "N/A")
        return fields
    except Exception:
        return {"actor1": "Unknown", "actor2": "Unknown", "event_type": "Unknown",
                "location": "Unknown", "date": "Unknown", "fatalities": "0",
                "latitude": "N/A", "longitude": "N/A"}


def extract_past_actions(rag_matches: list) -> list:
    """Pull past authority actions from INTSUM text in RAG matches."""
    actions = []
    for match in rag_matches:
        intsum_text = match["metadata"].get("intsum", "")
        if not intsum_text:
            continue
        # Extract Recommended Action / Response sections
        for pattern in [
            r"Recommended Action[s]?:(.*?)(?:\n\n|\Z)",
            r"Response[s]?:(.*?)(?:\n\n|\Z)",
            r"Actions? [Tt]aken:(.*?)(?:\n\n|\Z)",
        ]:
            found = re.search(pattern, intsum_text, re.DOTALL | re.IGNORECASE)
            if found:
                action_text = found.group(1).strip()
                lines = [l.strip("- •").strip() for l in action_text.split("\n") if l.strip()]
                for line in lines[:3]:
                    if len(line) > 15:
                        actions.append({
                            "action":   line,
                            "event":    match["metadata"].get("event_type", ""),
                            "location": match["metadata"].get("location", ""),
                            "date":     match["metadata"].get("event_date", ""),
                            "source":   "Historical INTSUM"
                        })
    return actions[:6]


def generate_intsum_text(query: str, fields: dict, rag_matches: list,
                          realtime: list, is_historical: bool) -> tuple:
    """Generate full INTSUM report and action recommendations."""

    # Build context from RAG + realtime
    rag_context = ""
    if rag_matches and rag_matches[0]["has_intsum"]:
        rag_context = rag_matches[0]["metadata"]["intsum"][:800]

    rt_context = ""
    if realtime:
        rt_context = " | ".join([r.get("snippet", "") for r in realtime[:2]])

    date_str = datetime.now().strftime("%d %B %Y")
    loc      = fields.get("location", "Unknown")
    evt_type = fields.get("event_type", "Security Incident")
    actor1   = fields.get("actor1", "Unknown actors")
    actor2   = fields.get("actor2", "")
    lat      = fields.get("latitude", "N/A")
    lon      = fields.get("longitude", "N/A")

    # Generate INTSUM narrative
    intsum_prompt = (
        f"Write a military intelligence summary (INTSUM) for the following event.\n\n"
        f"Event: {query}\n"
        f"Location: {loc} | Event Type: {evt_type} | Actors: {actor1}, {actor2}\n"
        f"Context: {rag_context[:400]} {rt_context[:200]}\n\n"
        f"Write a concise INTSUM covering: situation summary, strategic implications, analyst remarks."
    )
    try:
        intsum_narrative = generator(intsum_prompt, max_new_tokens=300, return_full_text=False)[0]["generated_text"]
    except Exception:
        intsum_narrative = f"Intelligence summary for {evt_type} event at {loc} involving {actor1}."

    # Generate recommended actions
    if is_historical:
        rec_prompt = (
            f"Based on past {evt_type} incidents in India involving {actor1}, "
            f"what security measures and recommended actions were taken? "
            f"List 4 specific action items for military and police authorities."
        )
    else:
        rec_prompt = (
            f"A {evt_type} incident has been reported at {loc} involving {actor1}. "
            f"What immediate action items should Indian military and police authorities take? "
            f"List 4 specific, actionable recommendations."
        )

    try:
        rec_raw = generator(rec_prompt, max_new_tokens=200, return_full_text=False)[0]["generated_text"]
        rec_lines = [l.strip("0123456789.-) ").strip()
                     for l in rec_raw.split("\n") if len(l.strip()) > 20]
        recommended = rec_lines[:5] if rec_lines else [
            "Deploy rapid response units to the affected area",
            "Establish a cordon and conduct area search operations",
            "Coordinate with local intelligence agencies for threat assessment",
            "Alert nearest security force battalion for reinforcement"
        ]
    except Exception:
        recommended = [
            "Deploy rapid response units to the affected area",
            "Establish a cordon and conduct area search operations",
            "Coordinate with local intelligence agencies",
            "Alert nearest security force battalion"
        ]

    # Build full formatted INTSUM
    full_intsum = f"""INTSUM REPORT
{'='*50}
Date/Time : {date_str} | {datetime.now().strftime('%H:%M')} HRS IST
Location  : {loc}
Coordinates: {lat}, {lon}
Event Type: {evt_type}
{'='*50}

ACTORS INVOLVED
Primary  : {actor1}
Secondary: {actor2 if actor2 else 'Not identified'}

SITUATION SUMMARY
{intsum_narrative}

STRATEGIC IMPLICATIONS
{"Based on historical precedent, similar incidents have escalated regional tensions." if is_historical else "Immediate assessment required. Situation may escalate without prompt intervention."}

SOURCE: {"Historical RAG Database + Real-time Intelligence" if rag_matches else "Real-time Intelligence Feed"}
CLASSIFICATION: RESTRICTED — FOR OFFICIAL USE ONLY
"""

    return full_intsum, recommended


# ──────────────────────────────────────────────
# API ENDPOINTS
# ──────────────────────────────────────────────

@app.post("/api/query", response_model=INTSUMResponse)
async def process_query(req: QueryRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # 1. RAG retrieval
    rag_matches = retrieve_from_rag(query, n_results=5)

    # 2. Real-time search
    realtime_results = await search_realtime(query, num=5)

    # 3. Determine if event is historical (strong RAG match) or recent
    best_similarity = rag_matches[0]["similarity"] if rag_matches else 0
    is_historical = best_similarity > 0.6

    # 4. Determine source
    if is_historical and realtime_results:
        source = "hybrid"
    elif is_historical:
        source = "rag"
    else:
        source = "realtime"

    # 5. Extract structured fields
    structured_fields = extract_structured_fields(query, rag_matches, realtime_results)

    # 6. Get past actions from historical INTSUM
    past_actions = extract_past_actions(rag_matches) if is_historical else []

    # 7. Generate INTSUM + recommendations
    intsum_text, recommended_actions = generate_intsum_text(
        query, structured_fields, rag_matches, realtime_results, is_historical
    )

    return INTSUMResponse(
        query=query,
        source=source,
        intsum=intsum_text,
        structured_fields=structured_fields,
        past_actions=past_actions,
        recommended_actions=recommended_actions,
        rag_matches=[{
            "note":       m["note"][:300],
            "event_type": m["metadata"].get("event_type", ""),
            "location":   m["metadata"].get("location", ""),
            "date":       m["metadata"].get("event_date", ""),
            "actor1":     m["metadata"].get("actor1", ""),
            "similarity": m["similarity"],
            "has_intsum": m["has_intsum"]
        } for m in rag_matches[:3]],
        realtime_results=realtime_results[:5],
        generated_at=datetime.now().isoformat()
    )


@app.get("/api/health")
def health():
    return {
        "status": "online",
        "rag_documents": event_collection.count(),
        "llm": "tinyllama-1.1b",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/event-types")
def get_event_types():
    return {
        "event_types": [
            "Battles", "Explosions/Remote violence", "Riots",
            "Violence against civilians", "Protests", "Strategic developments"
        ],
        "sub_event_types": [
            "Armed clash", "Remote explosive/landmine/IED", "Mob violence",
            "Abduction/forced disappearance", "Peaceful protest", "Attack",
            "Disrupted weapons use", "Grenade", "Arrests"
        ]
    }


# Serve frontend
frontend_path = os.path.join(os.path.dirname(__file__), "../frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
