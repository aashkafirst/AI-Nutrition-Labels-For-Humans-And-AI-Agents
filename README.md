# AI Nutrition Labels for AI Agents — POC (v0.3-draft)

A technical proof-of-concept that turns the ["AI Nutrition Labels for Everyone"](https://medium.com/@aashkafirst/ai-nutrition-labels-for-everyone-simplifying-model-cards-from-geek-to-street-7dec406b56f3)
framework into something an **AI agent can read and act on**: a standardized,
domain-agnostic JSON Schema, a registry of **real models** labeled from their
actual public model/system cards (plus real third-party energy/child-safety
benchmarks), a REST API, and two domain agents that each use the registry to
make an auditable, use-case-appropriate decision.

## Project layout

```
schema/
  ai_nutrition_label.schema.json   JSON Schema (v0.3-draft), standardized/global
registry/
  seed_source.json                 raw component data for 7 real models
  build_seed_data.py                computes PV/SV/BV + environmental grades -> seed_data.json
  seed_data.json                    full schema-conformant labels (generated, don't hand-edit)
  init_db.py                        builds registry.db (SQLite) from seed_data.json
agent/
  scoring.py                        the framework's formulae + AI Energy Score data, as code
  procurement_agent.py               use case 1: rural-India maternal health procurement
  teacher_snapshot.py                use case 2: Nairobi classroom safety snapshot
api/
  main.py                           FastAPI REST layer over the registry
notebook/
  demo.ipynb                        runnable, narrated end-to-end walkthrough (both use cases)
requirements.txt
```

## The label schema, in brief (mirrors your original categories)

| Category | Fields | Formula / source |
|---|---|---|
| **Model Identity** | name, manufacturer, release date, knowledge cutoff, parameter count, open-weights | from model/system card, or "undisclosed" if the provider doesn't publish it |
| **Functional Capabilities** | core capabilities, modality support, context window, supported languages | from model/system card |
| **Performance Value (PV)** | general reasoning, coding, math, common-sense reasoning (0-100 each) | `PV = (GR + C + M + CSR) / 4` |
| **Safety Value (SV)** | toxic-prompt refusal rate, non-toxic compliance rate, inappropriate (over-)refusal rate | `SV = (Rtoxic + (100-Rnontoxic) + (100-IR)) / 3` — the source article's formula lists 3 terms but divides by 4; we divide by 3 and note this explicitly as a correction |
| **Bias Value (BV)** | per-context bias benchmark scores (e.g. BBQ-style, 0-1 scale) | `BV = weighted average of bias benchmark values`; `null` when no such evaluation exists for a model (true for most models today) |
| **Environmental Impact** | Carbon Footprint grade (A+ to D), Energy Rating (1-5 stars), Green Energy Seal (%), Water Footprint level, plus `ai_energy_score` | grades derived from per-1k-query CO2e/water/energy figures (real where a provider published one, otherwise estimated -- see table below); thresholds calibrated against real reference points in `agent/scoring.py`; `ai_energy_score` explicitly ties each figure to the real AI Energy Score benchmark and discloses whether it's on the leaderboard |
| **Privacy** | Privacy Seal (Gold/Silver/Bronze/None), data used for training, retention policy, certifications | from provider's stated policy |
| **Limitations** | known limitations, not-recommended-for uses | from model/system card |
| **Provenance metadata** | who issued the label, when, source document URLs | — |
| **`estimated_fields`** | dot-path list of which fields above are placeholders, not sourced facts | lets an agent discount low-confidence fields instead of silently trusting them |
| **`extensions`** (optional) | reserved, empty by default; populated with `child_safety` (real KORA data) for the 3 conversational models in this registry | where a domain-specific consumer can attach extra fields without changing the core schema -- exactly what use case 2 below does |

## The 7 real models in the registry

| Model | Modality | Open weights? | Energy figure | Child safety (KORA) |
|---|---|---|---|---|
| Claude Sonnet 4.6 (Anthropic) | multimodal | No | 2.8 Wh/query -- third-party estimate, not Anthropic-published | 72% -- estimated, anchored to Claude Haiku 4.5's real reported 75.6% and KORA's "4 of top 5 are Claude family" statement |
| GPT-4o (OpenAI) | multimodal | No | 0.34 Wh/query -- **real**, Sam Altman's June 2025 public statement | 37% -- derived from KORA's own reported 38-point delta to ChatGPT 5.2's real 75% score |
| Gemini 1.5 Pro (Google DeepMind) | multimodal | No | 0.24 Wh/query -- **real**, Google's Aug 2025 environmental report (0.03 gCO2e, 0.26 mL water too) | 45% -- entirely estimated/dummy; no KORA figure exists for this specific model version |
| DALL-E 3 (OpenAI) | image | No | 2.9 Wh/query -- estimated, grounded in the real Luccioni et al. diffusion-model energy study | not applicable -- not a conversational model |
| Stable Diffusion 3.5 Large (Stability AI) | image | **Yes** | 3.5 Wh/query -- estimated, extrapolated from the real SDXL-base AI Energy Score measurement | not applicable |
| NLLB-200 (Meta AI) | translation | **Yes** | 0.7 Wh/query -- estimated (translation isn't one of AI Energy Score's 10 tasks at all) | not applicable |
| **Sarvam-Translate (Sarvam AI, India)** | translation | **Yes** | 0.011 Wh/query -- estimated, scaled from the real Mistral-7B AI Energy Score measurement (4B vs. NLLB's 54.5B MoE) | not applicable |

**Sarvam-Translate** is a real, India-built translation model (fine-tuned from Gemma3-4B-IT by Sarvam AI in partnership with AI4Bharat, open-weight under GPL-3.0, released June 2025) covering all 22 official Indian languages including Hindi and Marathi. Sarvam AI's own published human evaluation found it rated significantly better than much larger models (Gemma3-27B-IT, Llama4 Scout, Llama-3.1-405B) specifically for Indian-language translation quality -- a genuine domain-specialization advantage baked into this label, not an invented one. Combined with its much smaller footprint (4B vs. NLLB-200's 54.5B parameters, giving it an A+ carbon grade vs. NLLB's C), it out-scores NLLB-200 in the procurement demo below.

Every real vs. estimated distinction is also encoded machine-readably in each
label's `estimated_fields` array and in `environmental_impact.ai_energy_score.disclosure` /
`extensions.child_safety.score_basis` -- check the notebook's Section 2 to see
it printed directly from the data.

- **Real energy benchmark integration.** Environmental figures are tied
  to the real [AI Energy Score](https://huggingface.co/spaces/AIEnergyScore/Leaderboard)
  (Hugging Face / Salesforce / Carnegie Mellon) benchmark and methodology,
  plus other real published data: Google's own Gemini energy/water/CO2e
  report, OpenAI's stated ChatGPT energy figure, and the Luccioni et al.
  "Power Hungry Processing" image-generation energy study. See
  `environmental_impact.ai_energy_score` on every label.
- **KORA Child Safety benchmark integration.**: a teacher in Nairobi deciding which
  conversational AI is safe for her students, using real
  [KORA](https://korabench.ai/) child-safety benchmark data, attached under
  each label's `extensions.child_safety` -- and a **filtered snapshot view**
  (`agent/teacher_snapshot.py`) that shows her only what she needs, not the
  full label.
- Environmental grade thresholds were recalibrated against these real data
  points (see `agent/scoring.py`) -- the first draft's thresholds were
  calibrated on illustrative placeholder numbers that turned out to be far
  smaller than real published figures.

## How to run it (minimum tools: Python 3.10+, pip)

### 1. Set up the environment
```bash
cd "AI Nutrition Labels For AI Agents POC"
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Compute the labels and build the registry
```bash
python registry/build_seed_data.py   # computes PV/SV/BV + environmental grades -> seed_data.json
python registry/init_db.py           # loads seed_data.json into registry.db (SQLite)
```

### 3. Start the REST API
```bash
uvicorn api.main:app --reload --port 8000
```
Leave this running in its own terminal. Open http://127.0.0.1:8000/docs for
interactive Swagger docs. Quick checks:
```bash
curl http://127.0.0.1:8000/labels
curl "http://127.0.0.1:8000/labels/search?open_weights=true&carbon_footprint_grade=A"
curl "http://127.0.0.1:8000/labels/search?min_child_safety_score_pct=50"
curl http://127.0.0.1:8000/labels/gpt-4o-2024-05-13
```

### 4. Run the two domain agent demos
Either standalone (with the API running from step 3):
```bash
python agent/procurement_agent.py     # use case 1: rural-India maternal health
python agent/teacher_snapshot.py      # use case 2: Nairobi classroom safety
```
or, for the fully narrated version with both use cases, open the notebook:
```bash
jupyter notebook notebook/demo.ipynb
```
The notebook builds the registry and starts the API for you in a background
process if it isn't already running.

## Use case 1: maternal health app procurement for rural India

An agent is procuring the model stack for a **maternal health assistant for
rural clinics in India** (Hindi/Marathi-speaking regions). It needs three
components, and applies **different constraints per component** based on
where patient data physically goes:

- **Conversational LLM** — a health worker's phone sometimes has
  connectivity, so a cloud model is acceptable, but it must clear a safety
  floor (`safety_value >= 60`) and document Hindi support.
- **Vision triage model** (ultrasound/visual anemia screening) and the
  **translation model** — these touch identifiable patient data at the point
  of care and must run fully **on-device**, so `open_weights` is a hard
  requirement.

No fabricated `offline_capable` flag exists anywhere in the schema — the
agent uses `open_weights`, a real structural property of each model's actual
license, as the honest proxy for on-device deployability. **DALL-E 3 is
rejected** for the vision role purely for being closed-weight, and **Stable
Diffusion 3.5 Large** is selected instead; for the LLM slot, **Gemini 1.5
Pro's real, Google-published energy efficiency data** tips the composite
score in its favor over Claude Sonnet 4.6 -- a genuine, data-driven
tiebreaker rather than an arbitrary one.

For translation, both NLLB-200 and **Sarvam-Translate** are open-weight and
support Hindi/Marathi, so this slot comes down to score rather than a hard
reject -- and **Sarvam-Translate wins** (0.677 vs. NLLB-200's 0.616): it's a
much smaller, India-built model specifically fine-tuned for the 22 official
Indian languages, with a real published human-evaluation finding that it
outperforms much larger general-purpose models on Indian-language
translation quality, and a real efficiency advantage (4B params vs.
NLLB-200's 54.5B) that earns it an A+ carbon grade vs. NLLB's C. It's a
clean illustration of the framework's point: bigger and more general isn't
automatically the better registry pick once domain fit and efficiency are
weighed in.

## Use case 2: a teacher in Nairobi choosing a classroom-safe AI model

A completely different-shaped problem gets a completely different-shaped
answer. `agent/teacher_snapshot.py` ranks the three conversational models by
their real (or clearly-flagged-estimated) KORA child-safety score, then
renders a **short snapshot** for the winner -- child-safety tier, privacy
seal in plain language, and a filtered "watch out for" list -- deliberately
omitting performance benchmarks, context window, GPU energy figures, and
provenance metadata, none of which help a non-technical teacher decide.
In this registry, Claude Sonnet 4.6 (72%, estimated) is recommended over
Gemini 1.5 Pro (45%, dummy) and GPT-4o (37%, derived from KORA's own
reported improvement-over-time statistic) -- both alternates fall below the
default 50% classroom threshold and are explicitly flagged, not silently
hidden.

This is exactly what the schema's optional `extensions` object was reserved
for in the previous round: KORA data lives entirely under
`extensions.child_safety` on 3 labels, with zero changes to the core
standardized schema.

## Extending this POC

- **Add more models**: append a raw entry to `registry/seed_source.json`
  (component scores + sources only), then re-run
  `python registry/build_seed_data.py && python registry/init_db.py`. Never
  hand-edit `seed_data.json` directly — it's generated.
- **Add another domain-specific overlay**: attach extra fields under a
  label's `extensions` object (e.g. `extensions.finance = {...}`) without
  changing the core schema.
- **Tighten the environmental thresholds**: the letter-grade/star cutoffs in
  `agent/scoring.py` are calibrated against the real reference points cited
  there (Google's Gemini report, Luccioni et al.) — adjust them there if you
  have a better reference point.
- **Re-check the leaderboards periodically**: AI Energy Score updates every
  6-9 months and KORA is an actively maintained leaderboard -- if a model in
  this registry gets added to either, replace its estimated figure with the
  real one and remove it from `estimated_fields`.
- **Write endpoint**: currently the API is read-only; a `POST /labels` for
  registering new labels (validated via `jsonschema`) would be the natural
  next addition.
