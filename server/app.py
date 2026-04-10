# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Citation Detective Environment.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions
    - GET /tasks: List available tasks (hackathon)
    - POST /grader: Run grader on an action (hackathon)
    - POST /baseline: Run baseline LLM agent (hackathon)

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

import os
from pathlib import Path

# Auto-load .env file if present (for GEMINI_API_KEY etc.)
_env_file = Path(__file__).resolve().parent.parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import ForensicAction, ForensicObservation
    from .citation_detective_environment import CitationDetectiveEnvironment
except (ImportError, ModuleNotFoundError):
    from models import ForensicAction, ForensicObservation
    from server.citation_detective_environment import CitationDetectiveEnvironment


# Create the app with web interface and README integration
app = create_app(
    CitationDetectiveEnvironment,
    ForensicAction,
    ForensicObservation,
    env_name="citation_detective",
    max_concurrent_envs=1,
)


# Root endpoint for HF Spaces
@app.get("/", include_in_schema=False)
async def root():
    from fastapi.responses import HTMLResponse
    return HTMLResponse("""<!DOCTYPE html>
<html><head><title>Citation Detective</title>
<style>
  body { font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0;
         display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; }
  .card { background: #1e293b; border-radius: 16px; padding: 48px; max-width: 600px; text-align: center;
          box-shadow: 0 25px 50px rgba(0,0,0,0.5); }
  h1 { font-size: 2rem; margin-bottom: 8px; }
  .emoji { font-size: 3rem; }
  p { color: #94a3b8; line-height: 1.6; }
  .links { display: flex; flex-wrap: wrap; gap: 12px; justify-content: center; margin-top: 24px; }
  a { background: #3b82f6; color: white; padding: 10px 20px; border-radius: 8px;
      text-decoration: none; font-weight: 500; transition: background 0.2s; }
  a:hover { background: #2563eb; }
  .badge { display: inline-block; background: #22c55e; color: #0f172a; padding: 4px 12px;
           border-radius: 99px; font-size: 0.8rem; font-weight: 600; margin-bottom: 16px; }
</style></head>
<body><div class="card">
  <div class="emoji">🔬🔍</div>
  <h1>Citation Detective</h1>
  <div class="badge">● Running</div>
  <p>Forensic Peer Reviewer — an OpenEnv environment that trains AI agents to detect
     hallucinated, misattributed, and contradicting citations in scientific manuscripts.</p>
  <div class="links">
    <a href="/docs">API Docs</a>
    <a href="/health">Health</a>
    <a href="/tasks">Tasks</a>
    <a href="/schema">Schema</a>
  </div>
</div></body></html>""")


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(host="0.0.0.0", port=args.port)


# Also satisfy openenv validate which checks for bare main()
# main()



# ---------------------------------------------------------------------------
# Required hackathon endpoints: /tasks, /grader, /baseline
# ---------------------------------------------------------------------------
from fastapi import Request
from fastapi.responses import JSONResponse

try:
    from .graders import GRADERS, SCENARIOS
except ImportError:
    from graders import GRADERS, SCENARIOS


@app.get("/tasks")
async def list_tasks():
    """List all available tasks with descriptions and schemas."""
    tasks = []
    for task_id, scenario in SCENARIOS.items():
        tasks.append({
            "task_id": task_id,
            "difficulty": scenario["difficulty"],
            "description": scenario["description"],
            "issue_type": scenario["ground_truth"]["issue_type"],
            "action_schema": {
                "task_id": f"string — set to '{task_id}'",
                "action_type": "string — 'search', 'flag_hallucination', or 'approve'",
                "query": "string — search query (for action_type='search')",
                "citation_id": "int — citation ID to flag (for action_type='flag_hallucination')",
                "reason": "string — explanation for flagging",
                "search_history": "string — accumulated search results from prior steps",
            },
            "observation_fields": [
                "manuscript_excerpt", "citations_list", "search_results",
                "step_count", "task_id", "done", "reward", "metadata",
            ],
            "scoring": {
                "model": "composite_three_component",
                "formula": "BASE(0.05) + IDENTIFICATION(0..0.45) + REASON_QUALITY(0..0.40)",
                "score_range": "(0.05, 0.90) + efficiency bonus up to +0.04",
                "good_search": +0.1,
                "bad_search": -0.1,
                "max_steps": 5,
            },
        })
    return JSONResponse(content={"tasks": tasks, "total": len(tasks)})


@app.post("/grader")
async def run_grader(request: Request):
    """Run grader on an action for a specific task."""
    try:
        body = await request.json()
        task_id = body.get("task_id", "task_1")
        action = body.get("action", {})

        if task_id not in GRADERS:
            return JSONResponse(
                status_code=400,
                content={"error": f"Unknown task_id '{task_id}'. Valid: {list(GRADERS.keys())}"},
            )

        action_dict = {
            "action_type": action.get("action_type", "flag_hallucination"),
            "citation_id": action.get("citation_id", -1),
            "reason": action.get("reason", ""),
        }

        score = GRADERS[task_id](action_dict)
        scenario = SCENARIOS[task_id]

        return JSONResponse(content={
            "task_id": task_id,
            "difficulty": scenario["difficulty"],
            "score": score,
            "max_score": 1.0,
            "action_received": action_dict,
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/baseline")
async def run_baseline(request: Request):
    """Run baseline LLM agent against all tasks using a proper RL loop.

    Flow per task:
      1. env.reset(task_id)          — get manuscript + citations
      2. env.step(search) x N        — search each citation in DB
      3. LLM analyses search results — decides action
      4. env.step(flag/approve)      — terminal action, graded
    """
    import os
    import re
    import json as _json

    results = {}

    try:
        # Provider selection: Gemini → OpenAI → Azure
        gemini_key = os.environ.get("GEMINI_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")
        azure_key = os.environ.get("AZURE_OPENAI_API_KEY")

        provider = None
        model = None

        if gemini_key:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
            gemini_model = genai.GenerativeModel(model)
            provider = "gemini"
        elif openai_key:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            model = os.environ.get("OPENAI_MODEL", "gpt-4o")
            provider = "openai"
        elif azure_key:
            from openai import AzureOpenAI
            client = AzureOpenAI(
                api_key=azure_key,
                azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            )
            model = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
            provider = "azure"
        else:
            return JSONResponse(
                status_code=500,
                content={"error": "No API key found. Set GEMINI_API_KEY, OPENAI_API_KEY, or AZURE_OPENAI_API_KEY."},
            )

        def call_llm(prompt: str) -> str:
            if provider == "gemini":
                resp = gemini_model.generate_content(
                    prompt,
                    generation_config={"temperature": 0, "max_output_tokens": 1024,
                                       "response_mime_type": "application/json"},
                )
                return resp.text.strip()
            else:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0, max_tokens=512,
                )
                return resp.choices[0].message.content.strip()

        def parse_action(raw: str) -> dict:
            """Parse LLM JSON response with fallback strategies."""
            raw = raw.strip()
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1][4:] if len(parts) >= 2 else raw
            try:
                return _json.loads(raw)
            except Exception:
                pass
            m = re.search(r'\{[^{}]*"action_type"[^{}]*\}', raw, re.DOTALL)
            if m:
                try:
                    return _json.loads(m.group())
                except Exception:
                    pass
            return {
                "action_type": (re.search(r'"action_type"\s*:\s*"([^"]+)"', raw) or type('', (), {'group': lambda s, x: 'approve'})()).group(1),
                "citation_id": int(m.group(1)) if (m := re.search(r'"citation_id"\s*:\s*(\d+)', raw)) else -1,
                "reason": (re.search(r'"reason"\s*:\s*"([^"]*)"', raw) or type('', (), {'group': lambda s, x: raw[:200]})()).group(1),
            }

        # RL loop: one CitationDetectiveEnvironment instance per task
        for task_id in SCENARIOS:
            env = CitationDetectiveEnvironment()

            # Step 1: Reset — get manuscript and citations
            obs = env.reset(task_id=task_id)
            citations = obs.citations_list
            manuscript = obs.manuscript_excerpt
            search_history = ""
            step_count = 0
            cumulative_reward = 0.0

            # Step 2: Search each citation via env.step()
            for citation in citations:
                search_action = ForensicAction(
                    task_id=task_id,
                    action_type="search",
                    query=citation["title"],
                    search_history=search_history,
                    step_count=step_count,
                    cumulative_reward=cumulative_reward,
                )
                obs = env.step(search_action)
                search_history += f"\n--- Search: '{citation['title']}' ---\n{obs.search_results}\n"
                step_count = obs.step_count
                cumulative_reward += obs.reward

            # Step 3: LLM analyses all search results and decides
            citations_text = "\n".join(
                f"  [{c['id']}] {c['title']} — {', '.join(c['authors'])} ({c['year']})"
                for c in citations
            )
            prompt = f"""You are a forensic peer reviewer. Analyze the manuscript for hallucinated citations.

MANUSCRIPT:
{manuscript}

CITATIONS:
{citations_text}

DATABASE SEARCH RESULTS:
{search_history}

Based on the search results, determine if any citation is:
- A ghost paper (not found in the database)
- Misattributed (wrong authors or year compared to DB entry)
- Contradicting (manuscript claim contradicts what the cited paper says)
- Misquoted statistic (manuscript reports a different number than the cited paper)
- Causality reversal (paper shows correlation only, manuscript claims proven causation)
- Selective omission (manuscript cherry-picks one minor result, ignores the main conclusion)
- Temporal fabrication (paper with a future date that doesn't exist in the database)

Compare the manuscript claims against the actual database entries carefully.
Look for discrepancies in numbers, conclusions, authorship, and existence.

Respond with JSON only:
{{"action_type": "flag_hallucination", "citation_id": <int>, "reason": "<detailed 1-2 sentence explanation citing specific evidence>"}}
Or if all citations are correct:
{{"action_type": "approve", "citation_id": -1, "reason": "All citations verified"}}"""

            raw = call_llm(prompt)
            action_dict = parse_action(raw)

            # Step 4: Submit terminal action via env.step()
            terminal_action = ForensicAction(
                task_id=task_id,
                action_type=action_dict.get("action_type", "approve"),
                citation_id=action_dict.get("citation_id", -1),
                reason=action_dict.get("reason", ""),
                search_history=search_history,
                step_count=step_count,
                cumulative_reward=cumulative_reward,
            )
            obs = env.step(terminal_action)

            results[task_id] = {
                "difficulty": SCENARIOS[task_id]["difficulty"],
                "reward": obs.reward,
                "action_type": terminal_action.action_type,
                "citation_id": terminal_action.citation_id,
                "reason": terminal_action.reason,
                "steps_taken": obs.step_count,
            }

    except Exception as e:
        import traceback
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": traceback.format_exc(), "partial_results": results})

    overall = sum(r["reward"] for r in results.values()) / max(len(results), 1)

    return JSONResponse(content={
        "model": model,
        "provider": provider,
        "overall_reward": round(overall, 4),
        "task_results": results,
    })
