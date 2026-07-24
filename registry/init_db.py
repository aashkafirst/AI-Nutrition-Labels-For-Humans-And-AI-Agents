"""
Initializes registry.db (SQLite) from seed_data.json.

Run:
    python registry/build_seed_data.py   # computes PV/SV/BV/environmental grades
    python registry/init_db.py           # loads them into SQLite

Storage choice: SQLite. It's a single file, needs no server process,
ships with Python, and is easily queryable both via SQL and via the
REST API below. Each label is stored as a full JSON blob (matching
the schema exactly) plus a handful of indexed columns pulled out for
fast filtering (modality, performance/safety/bias values, environmental
grade, cost, etc.).
"""
import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "registry.db"
SEED_PATH = Path(__file__).parent / "seed_data.json"

SCHEMA = """
CREATE TABLE IF NOT EXISTS labels (
    label_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    manufacturer TEXT NOT NULL,
    modality TEXT NOT NULL,
    open_weights INTEGER,
    performance_value REAL,
    safety_value REAL,
    bias_value REAL,
    carbon_footprint_grade TEXT,
    energy_rating_stars INTEGER,
    water_footprint_level TEXT,
    green_energy_seal_pct REAL,
    ai_energy_score_on_leaderboard INTEGER,
    privacy_seal TEXT,
    context_window_tokens INTEGER,
    child_safety_score_pct REAL,     -- null unless extensions.child_safety is present
    raw_json TEXT NOT NULL           -- full label document, schema v0.3-draft
);
"""


def build_db():
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)

    labels = json.loads(SEED_PATH.read_text())
    for label in labels:
        child_safety = label.get("extensions", {}).get("child_safety")
        conn.execute(
            """INSERT INTO labels
               (label_id, name, manufacturer, modality, open_weights,
                performance_value, safety_value, bias_value,
                carbon_footprint_grade, energy_rating_stars, water_footprint_level,
                green_energy_seal_pct, ai_energy_score_on_leaderboard, privacy_seal,
                context_window_tokens, child_safety_score_pct, raw_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                label["label_id"],
                label["model_identity"]["name"],
                label["model_identity"]["manufacturer"],
                label["model_identity"]["modality"],
                int(label["model_identity"]["open_weights"]),
                label["performance"]["performance_value"],
                label["safety_and_bias"]["safety_value"],
                label["safety_and_bias"]["bias_value"],
                label["environmental_impact"]["carbon_footprint_grade"],
                label["environmental_impact"]["energy_rating_stars"],
                label["environmental_impact"]["water_footprint_level"],
                label["environmental_impact"]["green_energy_seal_pct"],
                int(label["environmental_impact"]["ai_energy_score"]["on_official_leaderboard"]),
                label["privacy"]["privacy_seal"],
                label["functional_capabilities"]["context_window_tokens"],
                child_safety["overall_safety_score_pct"] if child_safety else None,
                json.dumps(label),
            ),
        )
    conn.commit()
    conn.close()
    print(f"Loaded {len(labels)} labels into {DB_PATH}")


if __name__ == "__main__":
    build_db()
