"""
Implements the scoring formulae from the AI Nutrition Labels framework
(https://medium.com/@aashkafirst/ai-nutrition-labels-for-everyone...).

Labels in the registry store raw *component* scores (some real, sourced from
published model/system cards; some estimated placeholders where no public
data exists -- see each label's `estimated_fields` list). This module derives
the composite label values from those components using the framework's
formulae, so the composite numbers are always reproducible from their inputs
rather than being opaque hardcoded figures.

Composite values implemented:
    Performance Value (PV)     = (GR + C + M + CSR) / 4
    Safety Value (SV)          = (Rtoxic + (100 - Rnontoxic) + (100 - IR)) / 3
    Bias Value (BV)            = weighted average of bias benchmark scores (0-1 scale)
    Environmental grades        = threshold lookups over raw estimated resource use

Note on SV: the source article gives "SV = (Rtoxic + (100-Rnontoxic) + (100-IR)) / 4"
but lists exactly three terms; dividing by 4 looks like a documentation typo; we
divide by 3 (the number of terms actually summed) and note this explicitly.
"""
from dataclasses import dataclass
from typing import Optional


def compute_performance_value(components: dict) -> float:
    """PV = (general_reasoning + coding + math + common_sense_reasoning) / 4, each on 0-100 scale."""
    keys = ["general_reasoning", "coding", "math", "common_sense_reasoning"]
    return round(sum(components[k] for k in keys) / 4, 2)


def compute_safety_value(components: dict) -> float:
    """
    SV = (Rtoxic + (100 - Rnontoxic) + (100 - IR)) / 3
      Rtoxic  = refusal rate on toxic/harmful prompts (higher is better)
      Rnontoxic = compliance rate on benign prompts (higher is better, so
                  we invert it: (100 - Rnontoxic) rewards NOT over-blocking)
      IR      = inappropriate refusal rate / over-refusal on benign prompts
                (lower is better, so we invert it too)
    Result is 0-100, higher = safer.
    """
    r_toxic = components["toxic_prompt_refusal_rate_pct"]
    r_nontoxic = components["nontoxic_prompt_compliance_rate_pct"]
    ir = components["inappropriate_refusal_rate_pct"]
    return round((r_toxic + (100 - r_nontoxic) + (100 - ir)) / 3, 2)


def compute_bias_value(bias_benchmark_scores: dict, weights: Optional[dict] = None) -> float:
    """
    BV = weighted average of bias benchmark values (0-1 scale, higher = LESS biased).
    Defaults to an equal-weighted average across whichever cultural/demographic
    benchmark contexts are present (e.g. BBQ_english, BBQ_korean, BBQ_indian).
    """
    if not bias_benchmark_scores:
        return None
    if weights is None:
        weights = {k: 1.0 for k in bias_benchmark_scores}
    total_weight = sum(weights.get(k, 1.0) for k in bias_benchmark_scores)
    weighted_sum = sum(v * weights.get(k, 1.0) for k, v in bias_benchmark_scores.items())
    return round(weighted_sum / total_weight, 3)


# --- Environmental grade thresholds -----------------------------------------
# The source article uses a visual scale (A+ to D carbon grade, 1-5 star
# energy rating, water footprint tiers) but does not publish numeric cutoffs.
# These thresholds are calibrated against REAL published reference points
# (not arbitrary), so grades stay meaningful once real-world numbers are
# plugged in instead of illustrative placeholders:
#   - Google: median Gemini Apps text prompt = 0.24 Wh, 0.03 gCO2e, 0.26 mL
#     water (Google Cloud sustainability report, Aug 2025)
#   - OpenAI: ~0.34 Wh and ~0.322 mL water per average ChatGPT query
#     (Sam Altman, public statement, June 2025 -- not independently audited)
#   - Luccioni, Jernite & Strubell, "Power Hungry Processing: Watts Driving
#     the Cost of AI Deployment?" (FAccT 2024, arXiv:2311.16863): image
#     generation averaged ~2.9 Wh/image across tested diffusion models;
#     stable-diffusion-xl-base-1.0 measured at 1,594 gCO2eq per 1,000 images.
# These real data points span ~30 g CO2e/1k queries (efficient text) to
# ~1,600 g CO2e/1k queries (image generation) -- roughly a 50x gap, matching
# the paper's own finding that image/generative tasks are orders of
# magnitude more carbon-intensive than text. Thresholds below are set to
# span that real range instead of the much smaller illustrative range used
# in the first draft of this POC.

_CARBON_THRESHOLDS_G_CO2E_PER_1K = [
    (20, "A+"), (50, "A"), (150, "B"), (500, "C"),
]  # anything above the last bound -> "D"

_WATER_THRESHOLDS_LITERS_PER_1K = [
    (2, "Water Saver"), (10, "Low"), (30, "Moderate"), (80, "High"),
]  # anything above the last bound -> "Very High"

# Fixed grid-carbon-intensity constant used ONLY to derive CO2e from an energy
# figure when a provider publishes energy but not CO2e directly (e.g. OpenAI's
# 0.34 Wh/query claim came with a water figure but no CO2e figure). ~400 g
# CO2e/kWh approximates a blended global/US-average grid intensity (IEA global
# average ~429 gCO2/kWh 2023; US EPA eGRID national average ~369 gCO2/kWh
# 2022) -- a documented approximation, not a precise per-provider figure.
GENERIC_GRID_INTENSITY_G_CO2_PER_KWH = 400


def co2e_from_energy(wh_per_query: float) -> float:
    """Derive grams CO2e per 1,000 queries from Wh/query using the generic grid constant."""
    return round(wh_per_query * GENERIC_GRID_INTENSITY_G_CO2_PER_KWH, 1)


def carbon_grade_from_estimate(grams_co2e_per_1k_queries: float) -> str:
    for bound, grade in _CARBON_THRESHOLDS_G_CO2E_PER_1K:
        if grams_co2e_per_1k_queries <= bound:
            return grade
    return "D"


def water_level_from_estimate(liters_per_1k_queries: float) -> str:
    for bound, level in _WATER_THRESHOLDS_LITERS_PER_1K:
        if liters_per_1k_queries <= bound:
            return level
    return "Very High"


def energy_stars_from_estimate(wh_per_query: float) -> int:
    """
    5 stars = extremely efficient (<=0.3 Wh/query) down to 1 star (>3.0 Wh/query).
    Mirrors the AI Energy Score leaderboard's 5-tier design (it splits the GPU
    energy range for a task into five equal 20% bands), but our bands are fixed
    thresholds rather than a live percentile split over a same-task cohort,
    because none of this registry's models are on that leaderboard (see
    AI_ENERGY_SCORE_REFERENCE_MODELS below for why: it only benchmarks
    self-hostable models on standardized GPU hardware, not closed APIs).
    """
    if wh_per_query <= 0.3:
        return 5
    if wh_per_query <= 0.8:
        return 4
    if wh_per_query <= 1.5:
        return 3
    if wh_per_query <= 3.0:
        return 2
    return 1


# --- AI Energy Score (Hugging Face / Salesforce / Carnegie Mellon) ---------
# https://huggingface.co/blog/sasha/ai-energy-score-v2
# https://huggingface.github.io/AIEnergyScore/
# https://huggingface.co/spaces/AIEnergyScore/Leaderboard
#
# Real methodology: 10 tasks (text generation, reasoning, summarization,
# extractive QA, binary text classification, semantic sentence similarity,
# image classification, object detection, speech-to-text, image generation,
# image captioning), each model run on standardized GPU hardware (NVIDIA
# H100) via a Docker-based harness (CodeCarbon + the open-source
# ai-energy-benchmarks package), reporting GPU energy in Wh per 1,000
# queries and a 1-5 star rating (bottom 20% of measured GPU energy for that
# task = 1 star, top 20% = 5 stars). Text-generation models are further
# split into Class A (single consumer GPU, <=20B params), Class B (single
# cloud GPU, <=66B params), Class C (multiple cloud GPUs, >66B params).
#
# CRITICAL LIMITATION for this registry: the leaderboard can only measure
# models that can be *run* on its standardized hardware -- i.e. open-weight,
# self-hostable models. None of Claude, GPT-4o, Gemini, or DALL-E 3 (closed
# proprietary APIs) appear on it, and Stable Diffusion 3.5 Large / NLLB-200
# specifically are not in the current snapshot either (only smaller/older
# Stable Diffusion variants and no translation-task category exist). Every
# label in this registry is therefore "not on the official leaderboard" --
# each one's ai_energy_score.disclosure field says so explicitly, and where
# available cites the closest real reference points instead.
#
# A few real rows pulled directly from the leaderboard's published CSVs
# (huggingface.co/spaces/AIEnergyScore/Leaderboard, data/energy/*.csv). Their
# `total_gpu_energy` column is in kWh per 1,000 queries, which is numerically
# identical to Wh per query (1 kWh / 1000 queries = 1000 Wh / 1000 queries),
# so the raw CSV values are used directly below. Used as comparable_real_models
# context in the registry, not as scores for our (unmeasured) models.
AI_ENERGY_SCORE_REFERENCE_MODELS = {
    "text_generation": [
        {"name": "meta-llama/Meta-Llama-3-70B", "gpu_energy_wh_per_query": 1.7197, "stars": 5, "class": "C"},
        {"name": "mistralai/Mixtral-8x7B-v0.1", "gpu_energy_wh_per_query": 0.6154, "stars": 1, "class": "B"},
        {"name": "mistralai/Mistral-7B-v0.1", "gpu_energy_wh_per_query": 0.0191, "stars": 2, "class": "A"},
    ],
    "image_generation": [
        {"name": "stabilityai/stable-diffusion-xl-base-1.0", "gpu_energy_wh_per_query": 1.640, "stars": 1},
        {"name": "stabilityai/sdxl-turbo", "gpu_energy_wh_per_query": 0.386, "stars": 5},
        {"name": "stabilityai/sd-turbo", "gpu_energy_wh_per_query": 0.190, "stars": 5},
    ],
}
AI_ENERGY_SCORE_SOURCE_URL = "https://huggingface.co/spaces/AIEnergyScore/Leaderboard"
AI_ENERGY_SCORE_VERSION = "v2 (Dec 2025 refresh)"
