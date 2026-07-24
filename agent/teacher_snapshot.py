"""
Use case: a teacher in Nairobi deciding whether an AI model is safe to let her
students use. She doesn't need -- and shouldn't have to wade through -- the
full nutrition label (performance benchmarks, context window size, GPU energy
figures, carbon grades, provenance metadata...). She needs a handful of
answers: is this safe for kids, what age is it appropriate for, does it keep
my students' data private, and what should I watch out for.

This module produces a deliberately SHORT snapshot from the full label,
keeping only what's relevant to that decision and dropping everything else --
the opposite move from the procurement agent elsewhere in this repo, which
consumes the full label. Child-safety data comes from the real KORA benchmark
(https://korabench.ai/), attached under each label's optional `extensions`
object (see registry/seed_source.json) -- exactly the kind of domain-specific
overlay that object was reserved for.

Run standalone (API must already be running on :8000):
    python agent/teacher_snapshot.py
"""
import requests

API_BASE = "http://127.0.0.1:8000"


def _classroom_tier(score_pct: float) -> str:
    """
    Thresholds are our own, not KORA's -- KORA publishes a raw percentage and
    explicitly declines to certify any score as 'safe'. We still need a
    plain-language tier for a non-technical reader, so we define one here and
    say so, rather than implying KORA itself hands out these labels.
    """
    if score_pct >= 70:
        return "Usable with standard classroom supervision"
    if score_pct >= 50:
        return "Use with caution -- active teacher supervision recommended"
    return "Not recommended for direct student use without heavy adult oversight"


def generate_teacher_snapshot(label: dict) -> dict:
    """Filter a full nutrition label down to what a teacher needs to decide."""
    mi = label["model_identity"]
    priv = label["privacy"]
    child_safety = label.get("extensions", {}).get("child_safety")

    snapshot = {
        "model_name": mi["name"],
        "provider": mi["manufacturer"],
    }

    if child_safety is None:
        snapshot["child_safety"] = {
            "available": False,
            "note": "This model has no KORA child-safety evaluation attached. Treat as unassessed, not as safe.",
        }
    else:
        score = child_safety["overall_safety_score_pct"]
        snapshot["child_safety"] = {
            "available": True,
            "kora_score_pct": score,
            "classroom_guidance": _classroom_tier(score),
            "is_estimated": child_safety.get("estimated", False),
            "confidence_note": (
                "This score is our own estimate, not a number read directly off KORA's leaderboard -- see score_basis for how it was derived."
                if child_safety.get("estimated")
                else "Directly sourced from KORA's published leaderboard."
            ),
            "score_basis": child_safety.get("score_basis"),
            "kora_caution": "KORA itself cautions this is not a safety guarantee -- even the best-scoring models fail or are merely 'adequate' on a meaningful share of tested scenarios.",
        }

    snapshot["privacy"] = {
        "seal": priv["privacy_seal"],
        "student_conversations_used_for_training": priv["data_used_for_training"],
        "plain_summary": (
            "Student conversations are NOT used to train the model."
            if not priv["data_used_for_training"]
            else "Student conversations MAY be used to improve the model unless the school opts out -- check the retention policy before classroom rollout."
        ),
    }

    # known_limitations/not_recommended_for are already teacher-relevant as-is
    # (they're written for a general audience, not developers) -- just narrow
    # to the ones most likely to matter in a classroom with minors.
    relevant_keywords = ["hallucin", "bias", "medical", "legal", "financial", "child", "minor", "content"]
    snapshot["watch_out_for"] = [
        item for item in label["limitations"]["known_limitations"]
        if any(k in item.lower() for k in relevant_keywords)
    ] or label["limitations"]["known_limitations"][:2]

    return snapshot


def fetch_conversational_models():
    resp = requests.get(f"{API_BASE}/labels/search", params={"modality": "multimodal"}, timeout=5)
    resp.raise_for_status()
    return resp.json()


def recommend_for_classroom(min_kora_score: float = 50):
    """
    Rank the conversational models by KORA child-safety score (models with no
    KORA data are excluded, not defaulted to a neutral score -- for a
    classroom decision, "unassessed" should never quietly look like "safe").
    """
    candidates = fetch_conversational_models()
    scored = []
    for label in candidates:
        cs = label.get("extensions", {}).get("child_safety")
        if cs is None:
            continue
        scored.append((label, cs["overall_safety_score_pct"]))

    scored.sort(key=lambda pair: pair[1], reverse=True)

    print("=== Classroom AI model recommendation (Nairobi teacher use case) ===\n")
    for label, score in scored:
        tier = _classroom_tier(score)
        flag = "" if score >= min_kora_score else "  (BELOW your minimum threshold)"
        print(f"  {label['model_identity']['name']:<20} KORA score={score}%  -> {tier}{flag}")

    passing = [(l, s) for l, s in scored if s >= min_kora_score]
    if not passing:
        print(f"\n  No model meets the {min_kora_score}% threshold -- recommend NOT deploying any of these for unsupervised student use.")
        return None

    winner = passing[0][0]
    print(f"\n  RECOMMENDED: {winner['model_identity']['name']}\n")
    return generate_teacher_snapshot(winner)


def print_snapshot(snapshot: dict):
    print(f"AI Nutrition Label -- Classroom Snapshot: {snapshot['model_name']} ({snapshot['provider']})")
    print("-" * 60)
    cs = snapshot["child_safety"]
    if cs["available"]:
        print(f"Child safety (KORA benchmark): {cs['kora_score_pct']}%  -- {cs['classroom_guidance']}")
        print(f"  {cs['confidence_note']}")
        print(f"  KORA's own caveat: {cs['kora_caution']}")
    else:
        print(f"Child safety: NOT ASSESSED -- {cs['note']}")
    priv = snapshot["privacy"]
    print(f"\nPrivacy seal: {priv['seal']}  -- {priv['plain_summary']}")
    print("\nWatch out for:")
    for item in snapshot["watch_out_for"]:
        print(f"  - {item}")


if __name__ == "__main__":
    snapshot = recommend_for_classroom(min_kora_score=50)
    if snapshot:
        print()
        print_snapshot(snapshot)
