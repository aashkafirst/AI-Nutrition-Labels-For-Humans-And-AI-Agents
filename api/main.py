"""
REST API over the AI Nutrition Label registry (SQLite).

Run:
    uvicorn api.main:app --reload --port 8000

Then browse interactive docs at http://127.0.0.1:8000/docs

Endpoints:
    GET  /labels                        -> list all labels (summary view)
    GET  /labels/{label_id}             -> full label document
    GET  /labels/search                 -> filter by modality, manufacturer, open_weights,
                                            min_performance_value, min_safety_value, min_bias_value,
                                            carbon_footprint_grade, min_energy_rating_stars,
                                            privacy_seal, min_context_window_tokens
    GET  /schema                        -> the JSON Schema itself
"""
import json
import sqlite3
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query

DB_PATH = Path(__file__).parent.parent / "registry" / "registry.db"
SCHEMA_PATH = Path(__file__).parent.parent / "schema" / "ai_nutrition_label.schema.json"

app = FastAPI(
    title="AI Nutrition Label Registry API",
    description="Agent-readable registry of AI model nutrition labels (schema v0.2-draft).",
    version="0.2-draft",
)


def get_conn():
    if not DB_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="registry.db not found. Run `python registry/build_seed_data.py` then `python registry/init_db.py` first.",
        )
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@app.get("/schema")
def get_schema():
    return json.loads(SCHEMA_PATH.read_text())


@app.get("/labels")
def list_labels():
    conn = get_conn()
    rows = conn.execute(
        "SELECT label_id, name, manufacturer, modality, open_weights, performance_value, "
        "safety_value, bias_value, carbon_footprint_grade, energy_rating_stars, "
        "water_footprint_level, green_energy_seal_pct, ai_energy_score_on_leaderboard, "
        "privacy_seal, context_window_tokens, child_safety_score_pct FROM labels"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/labels/search")
def search_labels(
    modality: Optional[str] = Query(None, description="text, image, audio, translation, multimodal, tabular"),
    manufacturer: Optional[str] = Query(None),
    open_weights: Optional[bool] = Query(None),
    min_performance_value: Optional[float] = Query(None, ge=0, le=100),
    min_safety_value: Optional[float] = Query(None, ge=0, le=100),
    min_bias_value: Optional[float] = Query(None, ge=0, le=1),
    carbon_footprint_grade: Optional[str] = Query(None, description="A+, A, B, C, D -- exact match"),
    min_energy_rating_stars: Optional[int] = Query(None, ge=1, le=5),
    privacy_seal: Optional[str] = Query(None, description="Gold, Silver, Bronze, None"),
    min_context_window_tokens: Optional[int] = Query(None),
    min_child_safety_score_pct: Optional[float] = Query(
        None, description="Filters on extensions.child_safety.overall_safety_score_pct (KORA benchmark); only present for conversational LLM labels"
    ),
):
    conn = get_conn()
    clauses, params = [], []

    if modality:
        clauses.append("modality = ?")
        params.append(modality)
    if manufacturer:
        clauses.append("manufacturer LIKE ?")
        params.append(f"%{manufacturer}%")
    if open_weights is not None:
        clauses.append("open_weights = ?")
        params.append(int(open_weights))
    if min_performance_value is not None:
        clauses.append("performance_value >= ?")
        params.append(min_performance_value)
    if min_safety_value is not None:
        clauses.append("safety_value >= ?")
        params.append(min_safety_value)
    if min_bias_value is not None:
        clauses.append("bias_value >= ?")
        params.append(min_bias_value)
    if carbon_footprint_grade:
        clauses.append("carbon_footprint_grade = ?")
        params.append(carbon_footprint_grade)
    if min_energy_rating_stars is not None:
        clauses.append("energy_rating_stars >= ?")
        params.append(min_energy_rating_stars)
    if privacy_seal:
        clauses.append("privacy_seal = ?")
        params.append(privacy_seal)
    if min_context_window_tokens is not None:
        clauses.append("context_window_tokens >= ?")
        params.append(min_context_window_tokens)
    if min_child_safety_score_pct is not None:
        clauses.append("child_safety_score_pct >= ?")
        params.append(min_child_safety_score_pct)

    query = "SELECT raw_json FROM labels"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [json.loads(r["raw_json"]) for r in rows]


@app.get("/labels/{label_id}")
def get_label(label_id: str):
    conn = get_conn()
    row = conn.execute("SELECT raw_json FROM labels WHERE label_id = ?", (label_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail=f"No label with id '{label_id}'")
    return json.loads(row["raw_json"])
