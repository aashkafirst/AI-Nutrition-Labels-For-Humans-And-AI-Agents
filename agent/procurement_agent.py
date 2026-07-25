"""
Domain AI Agent: model procurement for a maternal-health app system in rural India.

The core label fields used here are the *standardized, domain-agnostic* ones
(PV, SV, BV, environmental grade, open_weights, supported_languages,
not_recommended_for) -- nothing healthcare-specific was added to the core
schema itself. Where a genuinely domain-relevant benchmark exists (HealthBench,
for the LLM candidates), it's attached under each label's optional
`extensions.healthcare` object instead -- same pattern as the KORA child-safety
data used in the classroom-teacher use case elsewhere in this repo.

Key reasoning move #1: none of these models have a literal "offline_capable"
or "rural_validated" field (that would be domain-specific). Instead the agent
uses `open_weights` as the honest proxy for "can this run fully on-device
with no data leaving the clinic."

Key reasoning move #2: there is no composite score. A single weighted formula
hides *why* a model won behind an opaque number, and forces an arbitrary
judgment call about how much safety should count against performance. Instead,
candidates that already clear the hard requirements (see explain_rejection)
are compared one metric at a time, in a priority order that reflects what
actually matters for THIS deployment. A metric only breaks a tie when the gap
between candidates exceeds a tolerance -- important because many component
scores in this registry are estimated placeholders (see each label's
`estimated_fields`), and a small difference between two estimates isn't a
real signal worth acting on. Every elimination is printed with the metric and
values that caused it, so the decision is fully traceable.

Key reasoning move #3: not every metric deserves equal billing. All of PV, SV,
environmental grade, etc. are still examined for every candidate (nothing is
hidden), but for the LLM role in a health-adjacent deployment, HealthBench --
which directly measures realistic healthcare-conversation quality against
physician-written rubrics -- is a far more relevant signal than the generic,
domain-agnostic PV/SV numbers, so it's checked first. A different use case
(e.g. the classroom-safety agent elsewhere in this repo) would reasonably
lead with a different metric instead.

Run standalone (API must already be running on :8000):
    python agent/procurement_agent.py
"""
from dataclasses import dataclass, field
from typing import Optional

import requests

API_BASE = "http://127.0.0.1:8000"

_GRADE_RANK = {"A+": 5, "A": 4, "B": 3, "C": 2, "D": 1}


def _get_metric(label: dict, metric: str):
    """Pull a comparable numeric value for `metric` out of the full label. None if not applicable."""
    if metric == "performance_value":
        return label["performance"]["performance_value"]
    if metric == "safety_value":
        return label["safety_and_bias"]["safety_value"]
    if metric == "bias_value":
        return label["safety_and_bias"]["bias_value"]
    if metric == "carbon_footprint_grade":
        return _GRADE_RANK.get(label["environmental_impact"]["carbon_footprint_grade"])
    if metric == "energy_rating_stars":
        return label["environmental_impact"]["energy_rating_stars"]
    if metric == "healthbench_score":
        # Domain-specific overlay (extensions.healthcare), not part of the core
        # standardized schema. Absent entirely for non-LLM/non-healthcare-scored
        # labels -- handled the same as any other "not available" metric.
        return label.get("extensions", {}).get("healthcare", {}).get("healthbench_score_pct")
    raise ValueError(f"Unknown metric: {metric}")


def _display_metric(label: dict, metric: str):
    """Human-readable value for print statements -- e.g. the letter grade, not its rank number."""
    if metric == "carbon_footprint_grade":
        return label["environmental_impact"]["carbon_footprint_grade"]
    if metric == "healthbench_score":
        value = _get_metric(label, metric)
        healthcare = label.get("extensions", {}).get("healthcare", {})
        confidence = "real" if healthcare.get("estimated") is False else "estimated"
        return f"{value} ({confidence})" if value is not None else None
    return _get_metric(label, metric)


@dataclass
class TaskConstraints:
    modality: str
    require_open_weights: bool = False       # proxy for "must run fully on-device / offline"
    required_languages: list = field(default_factory=list)  # hard gate
    min_performance_value: float = 0
    min_safety_value: Optional[float] = None   # None = skip this hard gate
    domain_caution_terms: list = field(default_factory=list)  # informational, not a hard reject
    # Ordered list of (metric_name, tolerance): the priority in which this
    # task's candidates get compared. A numeric metric only eliminates a
    # candidate if it trails the best value by more than `tolerance`;
    # candidates within tolerance are a practical tie and comparison moves to
    # the next metric. carbon_footprint_grade is grade-exact (tolerance
    # ignored) -- only an identical letter grade counts as tied.
    comparison_priority: list = field(default_factory=lambda: [
        ("safety_value", 2.0), ("carbon_footprint_grade", 0), ("performance_value", 2.0),
    ])


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


def compare_candidates(candidates: list, comparison_priority: list, verbose: bool = True):
    """
    Narrow `candidates` down one metric at a time, in priority order, printing
    the reasoning trace. Returns the single winner (first remaining candidate
    if a full tie survives every metric).
    """
    remaining = list(candidates)

    for metric, tolerance in comparison_priority:
        if len(remaining) <= 1:
            break

        scored = [(l, _get_metric(l, metric)) for l in remaining]
        scored = [(l, v) for l, v in scored if v is not None]
        if not scored:
            if verbose:
                print(f"    {metric}: not available for any remaining candidate -- skipping")
            continue

        best_value = max(v for _, v in scored)
        if metric == "carbon_footprint_grade":
            tied = [l for l, v in scored if v == best_value]  # exact grade match only
        else:
            tied = [l for l, v in scored if best_value - v <= tolerance]

        display = {l["model_identity"]["name"]: _display_metric(l, metric) for l, _ in scored}
        if verbose:
            print(f"    Compare on {metric}: {display}")

        if len(tied) < len(remaining):
            eliminated = [l["model_identity"]["name"] for l in remaining if l not in tied]
            if verbose:
                print(f"      -> eliminates {eliminated}; {[l['model_identity']['name'] for l in tied]} remain")
            remaining = tied
        elif verbose:
            print("      -> all remaining candidates within tolerance here; no elimination")

    return remaining[0]


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
            print(f"  ELIGIBLE  {label['model_identity']['name']:<28}")

    candidates = [l for l, _ in passing]
    if not candidates:
        if verbose:
            print("  -> No model meets constraints. Escalating to human reviewer.")
        return None

    if len(candidates) == 1:
        winner = candidates[0]
        if verbose:
            print(f"  Only one eligible candidate -- no comparison needed: {winner['model_identity']['name']}")
    else:
        if verbose:
            print("  Comparing eligible candidates metric-by-metric (no composite score):")
        winner = compare_candidates(candidates, constraints.comparison_priority, verbose)

    if verbose:
        mi = winner["model_identity"]
        print(f"  SELECTED  {mi['name']}")
        caution = [
            t for t in winner["limitations"]["not_recommended_for"]
            if any(term.lower() in t.lower() for term in constraints.domain_caution_terms)
        ]
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
        safety floor and cover Hindi. All of PV, SV, environmental grade, etc.
        are still examined (see the printed trace for every metric checked),
        but for THIS use case, HealthBench -- a domain benchmark that
        specifically measures realistic healthcare-conversation quality
        against physician-written rubrics (extensions.healthcare, since it's
        domain-specific, not part of the core standardized schema) -- is the
        single most relevant signal: it's a direct read on both capability
        (PV-like) and safety-in-context (SV-like) for exactly this job, so
        it's compared first. Generic safety_value, environmental grade, and
        generic performance_value are still checked afterward as
        corroborating signals / tiebreakers if HealthBench doesn't
        distinguish the candidates.
      - Vision triage model (ultrasound/visual anemia screening) and the
        translation model: these process identifiable patient data at the
        point of care and must run fully on-device (open-weight, self-hosted)
        with no connectivity requirement at all. Neither has a domain
        benchmark equivalent to HealthBench in this registry, so they compare
        on the most directly relevant generic metrics instead: translation
        quality (performance_value) first for the translation slot, since
        it's the direct measure of fitness for that specific job.
    """
    llm_constraints = TaskConstraints(
        modality="multimodal",
        require_open_weights=False,
        required_languages=["Hindi"],
        min_safety_value=60,
        domain_caution_terms=["medical", "diagnosis"],
        comparison_priority=[
            ("healthbench_score", 3.0),   # most relevant to THIS use case: real health-conversation quality + safety
            ("safety_value", 2.0),        # generic safety, as a corroborating check
            ("carbon_footprint_grade", 0),
            ("performance_value", 2.0),
        ],
    )
    vision_constraints = TaskConstraints(
        modality="image",
        require_open_weights=True,
        domain_caution_terms=["medical"],
        comparison_priority=[("carbon_footprint_grade", 0), ("performance_value", 2.0)],
    )
    translation_constraints = TaskConstraints(
        modality="translation",
        require_open_weights=True,
        required_languages=["Hindi", "Marathi"],
        comparison_priority=[("performance_value", 2.0), ("carbon_footprint_grade", 0)],
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
