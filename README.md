# 🔍 Citation Detective — Forensic Peer Reviewer Environment

An OpenEnv-compatible RL environment where agents act as **forensic peer reviewers**, detecting hallucinated, misattributed, and misrepresented citations in scientific manuscripts.

## Overview

Scientific papers increasingly contain fabricated or manipulated citations — references that look legitimate but are actually ghost papers, identity thefts, contradictions, or statistical fabrications. This environment trains agents to catch these manipulations through a multi-step investigation process.

### Why This Matters

- **Ghost papers** make up an estimated 2-5% of citations in predatory journals
- **Misquoted statistics** and **causality reversals** are common in science journalism
- An RL-trained forensic reviewer could serve as a scalable defense against citation fraud

## Task Design

| Task | Difficulty | Type | Description |
|------|-----------|------|-------------|
| 1 | Easy | Ghost Paper | A citation that is completely fabricated — the paper does not exist |
| 2 | Medium | Identity Theft | Real paper title but wrong authors and year |
| 3 | Hard | Contradiction | Correct citation but manuscript claim contradicts the paper |
| 4 | Medium | Misquoted Statistic | Real paper but manuscript fabricates the numerical result |
| 5 | Hard | Causality Reversal | Paper shows correlation only, manuscript claims causation |
| 6 | Hard | Selective Omission | Cherry-picks one minor positive finding, ignores main negative conclusion |
| 7 | Expert | Temporal Fabrication | Cites a future paper that doesn't exist, contradicting real research |

## Reward Design

The grader uses a **three-component composite score** that produces values strictly in (0, 1):

```
score = BASE(0.05) + IDENTIFICATION(0..0.45) + REASON_QUALITY(0..0.40)
```

| Component | Values | Signal |
|-----------|--------|--------|
| **Base** | 0.05 | Minimum for any attempt — ensures non-zero gradient |
| **Identification** | 0.45 correct, 0.15 wrong flag, 0.00 approve | Rewards finding the right citation |
| **Reason Quality** | 0.40 excellent, 0.25 good, 0.15 partial, 0.05 minimal | Rewards understanding *why* it's wrong |

This produces a **smooth, continuous reward signal** with natural range [0.05, 0.90] — no artificial clamping needed.

### Score Examples
- ✅ Correct citation + excellent reason: **0.90** (strong positive)
- ✅ Correct citation + good reason: **0.75** (good signal)
- ⚠️ Wrong citation flagged: **0.25** (mild negative)
- ❌ Approved a hallucination: **0.05** (strong negative)

## Action Space

| Action | Description |
|--------|-------------|
| `search(query)` | Query the citation database — returns matching papers with full metadata |
| `flag_hallucination(citation_id, reason)` | Terminal — flag a specific citation as problematic with explanation |
| `approve()` | Terminal — approve the manuscript (always wrong in this environment) |

Agents can perform multiple searches before making a terminal decision, enabling a **plan → investigate → conclude** workflow.

## Docker (Recommended)

```bash
cd citation_detective
docker build -t citation-detective:latest .
docker run --rm -p 7860:7860 citation-detective:latest
curl http://localhost:7860/health
```

## Without Docker

```bash
cd citation_detective
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Run Inference

```bash
export HF_TOKEN="your_huggingface_token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
python inference.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Start new episode with a task |
| `/step` | POST | Execute an action (search/flag/approve) |
| `/state` | GET | Current environment state |
| `/schema` | GET | Action/observation JSON schemas |
| `/tasks` | GET | List all available tasks |
| `/grader` | POST | Grade an action for a specific task |
| `/baseline` | POST | Run baseline agent on a task |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes (runtime) | HuggingFace API token for inference |
| `API_BASE_URL` | Yes (runtime) | OpenAI-compatible API endpoint |
| `MODEL_NAME` | Yes (runtime) | Model identifier |

## Architecture

```
citation_detective/
├── Dockerfile               # Container configuration
├── requirements.txt         # Python dependencies
├── scenario_config.json     # Task configuration and verifiers
├── openenv.yaml             # OpenEnv specification
├── models.py                # Pydantic models (Action, Observation)
├── inference.py             # Agent inference script
├── baseline.py              # Baseline agent for testing
├── server/
│   ├── app.py               # FastAPI application
│   ├── graders.py           # Task scenarios + grader functions
│   └── citation_detective_environment.py  # Environment implementation
└── README.md
```
