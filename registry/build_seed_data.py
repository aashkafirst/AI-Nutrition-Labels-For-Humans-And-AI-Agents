"""
Builds registry/seed_data.json (fully schema-conformant, with computed
composite scores) from registry/seed_source.json (raw component data only).

Run:
    python registry/build_seed_data.py

This keeps PV/SV/BV and environmental grades reproducible from their inputs --
they are computed here via agent/scoring.py's formulae rather than hand-typed,
so changing a component score automatically recomputes the composite.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.scoring import (
    compute_performance_value,
    compute_safety_value,
    compute_bias_value,
    carbon_grade_from_estimate,
    water_level_from_estimate,
    energy_stars_from_estimate,
)

SOURCE_PATH = Path(__file__).parent / "seed_source.json"
OUTPUT_PATH = Path(__file__).parent / "seed_data.json"


def build_label(raw: dict) -> dict:
    perf = raw["performance"]
    pv = compute_performance_value(perf["components"])

    sb = raw["safety_and_bias"]
    safety_applicable = sb.get("safety_applicable", True)
    sv = compute_safety_value(sb["safety_components"]) if safety_applicable else None
    bv = compute_bias_value(sb["bias_benchmark_scores"])

    env_raw = raw["environmental_impact"]
    carbon_grade = carbon_grade_from_estimate(env_raw["estimated_co2e_grams_per_1k_queries"])
    water_level = water_level_from_estimate(env_raw["estimated_water_liters_per_1k_queries"])
    energy_stars = energy_stars_from_estimate(env_raw["estimated_energy_wh_per_query"])

    label = {
        "label_id": raw["label_id"],
        "schema_version": "0.3-draft",
        "estimated_fields": raw["estimated_fields"],
        "model_identity": raw["model_identity"],
        "functional_capabilities": raw["functional_capabilities"],
        "performance": {
            "components": perf["components"],
            "performance_value": pv,
            "formula": "PV = (GR + C + M + CSR) / 4",
            "benchmark_sources": perf["benchmark_sources"],
            "processing_speed": perf["processing_speed"],
        },
        "safety_and_bias": {
            "safety_applicable": safety_applicable,
            "safety_components": sb["safety_components"],
            "safety_value": sv,
            "safety_formula": "SV = (Rtoxic + (100 - Rnontoxic) + (100 - IR)) / 3",
            "bias_benchmark_scores": sb["bias_benchmark_scores"],
            "bias_value": bv,
            "bias_formula": "BV = weighted average of bias benchmark values (0-1 scale)",
            "red_team_summary": sb["red_team_summary"],
        },
        "environmental_impact": {
            "carbon_footprint_grade": carbon_grade,
            "energy_rating_stars": energy_stars,
            "green_energy_seal_pct": env_raw["green_energy_seal_pct"],
            "water_footprint_level": water_level,
            "estimated_co2e_grams_per_1k_queries": env_raw["estimated_co2e_grams_per_1k_queries"],
            "estimated_water_liters_per_1k_queries": env_raw["estimated_water_liters_per_1k_queries"],
            "estimated_energy_wh_per_query": env_raw["estimated_energy_wh_per_query"],
            "methodology_note": env_raw["methodology_note"],
            "ai_energy_score": env_raw["ai_energy_score"],
        },
        "privacy": raw["privacy"],
        "limitations": raw["limitations"],
        "provenance_metadata": raw["provenance_metadata"],
    }
    if "extensions" in raw:
        label["extensions"] = raw["extensions"]
    return label


def build():
    raw_labels = json.loads(SOURCE_PATH.read_text())
    labels = [build_label(r) for r in raw_labels]
    OUTPUT_PATH.write_text(json.dumps(labels, indent=2))
    print(f"Computed {len(labels)} labels -> {OUTPUT_PATH}")
    for l in labels:
        pv = l["performance"]["performance_value"]
        sv = l["safety_and_bias"]["safety_value"]
        bv = l["safety_and_bias"]["bias_value"]
        env = l["environmental_impact"]
        print(
            f"  {l['model_identity']['name']:<28} PV={pv:<6} SV={sv if sv is not None else 'N/A':<6} "
            f"BV={bv if bv is not None else 'N/A':<6} Carbon={env['carbon_footprint_grade']:<3} "
            f"Energy={'*' * env['energy_rating_stars']:<5} Water={env['water_footprint_level']}"
        )


if __name__ == "__main__":
    build()
