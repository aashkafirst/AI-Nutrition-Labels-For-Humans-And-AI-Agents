"""
Domain AI Agent: model procurement for a maternal-health app system in rural India.

Unlike the earlier version of this demo, the constraints below are expressed
entirely in terms of the *standardized, domain-agnostic* label fields (PV,
SV, BV, environmental grades, open_weights, supported_languages,
not_recommended_for) -- nothing healthcare-specific was added to the schema
itself. This is deliberate: it shows that meaningful domain-aware procurement
reasoning is possible even from a fully generic label, by combining generic
signals in a domain-aware way. A future healthcare-specific consumer could
still attach extra fields under a label's optional `extensions` object
without changing the core schema.

Key reasoning move: none of these models have a literal "offline_capable" or
"rural_validated" field (that would be domain-specific). Instead the agent
uses `open_weights` as the honest proxy for "can this run fully on-device
with no data leaving the clinic" -- which is exactly what matters for
rural/low-connectivity deployment, and is a real, structural property of the
model rather than an invented one.

Run standalone (API must already be running on :8000):
    python agent/procurement_agent.py
"""
from dataclasses import dataclass, field
from typing import Optional

import requests

API_BASE = "http://127.0.0.1:8000"

# Carbon letter grade -> normalized 0-1 score, used as a scoring tiebreaker.
_CARBON_GRADE_SCORE = {"A+": 1.0, "A": 0.85, "B": 0.65, "C": 0.45, "D": 0.2}


@dataclass
class TaskConstraints:
    modality: str
    require_open_weights: bool = False      # proxy for "must run fully on-device / offline"
    required_languages: list = field(default_factory=list)  # hard gate
    min_performance_value: float = 0
    min_safety_value: Optional[float] = None   # None = skip (or model has no safety concept)
    domain_caution_terms: list = field(default_factory=list)  # informational, not a hard reject


def fetch_all_in_modality(modality: str):
    resp = requests.get(f"{API_BASE}/labels/search", params={"modality": modality}, timeout=5)
    resp.raise_for_status()
    return resp.json()


def explain_rejection(label: dict, constraints: TaskConstraints) -> Optional[str]:
    mi, fc, perf, sb = (
        label["model_identity"], label["functional_capabilities"],
        label["performance"], label["safety_and_bias"],
    )

    if constraints.require_open_weights and not mi["open_weights"]:
        return (
            "not open-weight -- cannot be self-hosted on-device, so it can't run "
            "in a rural clinic with unreliable connectivity (this deployment's hard requirement)"
        )

    if perf["performance_value"] < constraints.min_performance_value:
        return f"performance_value {perf['performance_value']} < required {constraints.min_performance_value}"

    if constraints.min_safety_value is not None:
        if sb["safety_value"] is None:
            return "safety_value not applicable/available for this modality, but a safety floor was required"
        if sb["safety_value"] < constraints.min_safety_value:
            return f"safety_value {sb['safety_value']} < required {constraints.min_safety_value}"

    if constraints.required_languages:
        supported = set(fc.get("supported_languages", []))
        missing = set(constraints.required_languages) - supported
        if missing:
            return f"documented supported_languages does not list required language(s): {sorted(missing)}"

    return None


def score_candidate(label: dict, constraints: TaskConstraints) -> float:
    perf = label["performance"]["performance_value"] / 100
    safety = label["safety_and_bias"]["safety_value"]
    safety_norm = (safety / 100) if safety is not None else 0.7  # neutral prior when N/A
    bias = label["safety_and_bias"]["bias_value"]
    bias_norm = bias if bias is not None else 0.7  # neutral prior when no bias eval exists
    env = _CARBON_GRADE_SCORE.get(label["environmental_impact"]["carbon_footprint_grade"], 0.5)

    supported = set(label["functional_capabilities"].get("supported_languages", []))
    required = set(constraints.required_languages)
    lang_bonus = len(required & supported) / len(required) if required else 1.0

    score = 0.35 * perf + 0.25 * safety_norm + 0.15 * bias_norm + 0.15 * lang_bonus + 0.10 * env
    return round(score, 4)


def select_model(constraints: TaskConstraints, verbose: bool = True):
    all_candidates = fetch_all_in_modality(constraints.modality)
    passing, rejected = [], []

    for label in all_candidates:
        reason = explain_rejection(label, constraints)
        (rejected if reason else passing).append((label, reason))

    if verbose:
        print(f"\n=== Selecting {constraints.modality.upper()} model ===")
        for label, reason in rejected:
            print(f"  REJECTED  {label['model_identity']['name']:<28} -> {reason}")
        for label, _ in passing:
            score = score_candidate(label, constraints)
            print(f"  ELIGIBLE  {label['model_identity']['name']:<28} -> score={score}")

    if not passing:
        if verbose:
            print("  -> No model meets constraints. Escalating to human reviewer.")
        return None

    winner = max((l for l, _ in passing), key=lambda l: score_candidate(l, constraints))

    if verbose:
        mi, sb, env = winner["model_identity"], winner["safety_and_bias"], winner["environmental_impact"]
        caution = [
            t for t in winner["limitations"]["not_recommended_for"]
            if any(term.lower() in t.lower() for term in constraints.domain_caution_terms)
        ]
        print(f"  SELECTED  {mi['name']} (score={score_candidate(winner, constraints)})")
        print(
            f"  Reasoning: performance_value={winner['performance']['performance_value']}, "
            f"safety_value={sb['safety_value']}, bias_value={sb['bias_value']}, "
            f"open_weights={mi['open_weights']}, carbon_grade={env['carbon_footprint_grade']}."
        )
        if caution:
            print(f"  Caution ({mi['name']} provider disclaims): {caution} -- deploy with mandatory human review for these uses.")

    return winner


def run_procurement():
    """
    Scenario: procuring components for a maternal health app for rural India
    (Hindi/Marathi-speaking regions, unreliable clinic connectivity).

    Different components get different hard constraints because they differ in
    where patient data physically goes:
      - Conversational LLM: connectivity is sometimes available via a health
        worker's phone, so a cloud model is acceptable -- but it must clear a
        safety floor and cover Hindi.
      - Vision triage model (ultrasound/visual anemia screening) and the
        translation model: these process identifiable patient data at the
        point of care and must run fully on-device (open-weight, self-hosted)
        with no connectivity requirement at all.
    """
    llm_constraints = TaskConstraints(
        modality="multimodal",
        require_open_weights=False,
        required_languages=["Hindi"],
        min_safety_value=60,
        domain_caution_terms=["medical", "diagnosis"],
    )
    vision_constraints = TaskConstraints(
        modality="image",
        require_open_weights=True,
        domain_caution_terms=["medical"],
    )
    translation_constraints = TaskConstraints(
        modality="translation",
        require_open_weights=True,
        required_languages=["Hindi", "Marathi"],
    )

    results = {
        "language_model": select_model(llm_constraints),
        "vision_model": select_model(vision_constraints),
        "translation_model": select_model(translation_constraints),
    }

    print("\n=== Final Procurement Decision ===")
    for role, label in results.items():
        name = label["model_identity"]["name"] if label else "NONE SELECTED (escalate to human)"
        print(f"  {role}: {name}")

    return results


if __name__ == "__main__":
    run_procurement()
