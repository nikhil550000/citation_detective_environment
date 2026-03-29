---
title: Citation Detective
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
tags:
  - openenv
license: mit
---

# Citation Detective — Forensic Peer Reviewer 🔬🔍

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment that trains AI agents to detect **hallucinated, misattributed, and contradicting citations** in scientific manuscripts.

## Real-World Problem

Scientific citation fraud is a growing problem. LLMs often generate plausible-sounding but entirely fabricated references. Human peer reviewers struggle to verify every citation in a manuscript. This environment trains agents to perform **forensic citation review** — the critical task of cross-referencing cited papers against a database to catch:

- **Ghost Papers** — citations that don't exist at all
- **Identity Theft** — real paper titles with wrong authors/year
- **Contradictions** — correct citations whose conclusions are misrepresented

## Environment Overview

| Feature | Detail |
|---------|--------|
| **Tasks** | 3 (easy → hard) |
| **Max Steps** | 5 per episode |
| **Actions** | `search`, `flag_hallucination`, `approve` |
| **Reward Range** | -1.0 to +1.0 |
| **External APIs** | None (fully deterministic, mock database) |
| **Spec Compliance** | Full OpenEnv Gymnasium API |

## Tasks

### Task 1 — The Ghost Paper (Easy)
A citation references a paper that **does not exist** in the database. The agent should search the database, notice the paper is missing, and flag it.

### Task 2 — The Identity Theft (Medium)
A citation uses a **real paper's title** but lists **wrong authors and year**. The agent must notice the metadata mismatch by searching the database.

### Task 3 — The Contradiction (Hard)
A citation is **correctly attributed** (right title, authors, year), but the manuscript **claims the opposite** of what the cited paper actually found. The agent must read the abstract carefully.

## Action Space

```python
ForensicAction(
    task_id="task_1",           # Required — identifies the task
    action_type="search",       # "search" | "flag_hallucination" | "approve"
    query="Neural Pathways",    # Search query (for search actions)
    citation_id=2,              # Citation to flag (for flag actions)
    reason="Paper not found",   # Explanation (for flag actions)
    search_history="...",       # Accumulated search results from prior steps
)
```

## Observation Space

```python
ForensicObservation(
    manuscript_excerpt="...",   # Scientific text with [1], [2] citations
    citations_list=[...],       # List of {id, title, authors, year} dicts
    search_results="...",       # Database search output (after search action)
    step_count=1,               # Steps taken so far
    task_id="task_1",           # Current task
    done=False,                 # Episode finished?
    reward=0.1,                 # Current step reward
)
```

## Reward Shaping

| Action | Reward |
|--------|--------|
| Good search (results found) | +0.1 |
| Bad search (no results) | -0.1 |
| Correct flag + good reason | +1.0 |
| Correct flag, weak reason | +0.5 to +0.7 |
| Wrong citation flagged | -0.5 |
| Approve (missed hallucination, Tasks 1-2) | -1.0 |
| Approve (missed contradiction, Task 3) | -0.5 |

## Quick Start

### 1. Install dependencies
```bash
git clone https://github.com/nikhil550000/citation_detective_environment.git
cd citation_detective_environment
uv sync
```

### 2. Start the server
```bash
uv run server
# Or manually:
# uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 3. Reset and explore
```bash
# Get available tasks
curl http://localhost:8000/tasks

# Reset to a task
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_1"}'

# Search the database
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"task_id": "task_1", "action_type": "search", "query": "Neural Pathways"}}'

# Flag a citation
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"task_id": "task_1", "action_type": "flag_hallucination", "citation_id": 2, "reason": "Paper not found in database"}}'
```

### 4. Run baseline agent
```bash
# Create .env with your API key
echo 'GEMINI_API_KEY=your-key-here' > .env

# Run baseline
python baseline.py --url http://localhost:8000
```

### 5. Use the standalone grader
```bash
curl -X POST http://localhost:8000/grader \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_1", "action": {"citation_id": 2, "reason": "not found"}}'
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/schema` | Action/Observation JSON schemas |
| `GET` | `/tasks` | List available tasks |
| `POST` | `/reset` | Start new episode |
| `POST` | `/step` | Execute action |
| `GET` | `/state` | Get current state |
| `POST` | `/grader` | Run grader on an action |
| `POST` | `/baseline` | Run baseline LLM agent |
| `WS` | `/ws` | WebSocket for persistent sessions |

## Project Structure

```
citation_detective/
├── models.py                              # ForensicAction & ForensicObservation
├── client.py                              # EnvClient subclass
├── baseline.py                            # Standalone baseline inference
├── openenv.yaml                           # OpenEnv configuration
├── pyproject.toml                         # Python project + dependencies
├── server/
│   ├── app.py                             # FastAPI app + hackathon endpoints
│   ├── citation_detective_environment.py  # Environment logic
│   ├── graders.py                         # Mock database + grading functions
│   └── Dockerfile                         # Container build
└── README.md                              # This file
```

## Deployment to Hugging Face Spaces

```bash
# Build Docker image
cd citation_detective
docker build -t citation-detective:latest -f server/Dockerfile .

# Or deploy directly to HF Spaces
# (push this directory to a HF Space with the `openenv` tag)
```

## Built For

**Meta PyTorch OpenEnv Hackathon × SST (India AI Hackathon 2026)** — organized by [Scaler School of Technology](https://www.scaler.com/school-of-technology/).

Built as a complete OpenEnv environment submission for Round 1.

## License

MIT License — see [LICENSE](LICENSE) file.
