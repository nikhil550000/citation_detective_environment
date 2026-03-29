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
                "step_count", "task_id", "done", "reward",
            ],
            "scoring": {
                "correct_flag": "up to 1.0 (depends on reason quality)",
                "wrong_flag": -0.5,
                "approve_task_1_2": -1.0,
                "approve_task_3": -0.5,
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
    """Run baseline LLM agent against all tasks."""
    import os
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

        # Run each task
        for task_id, scenario in SCENARIOS.items():
            citations_text = ""
            for c in scenario["citations_list"]:
                authors = ", ".join(c["authors"])
                citations_text += f"  [{c['id']}] {c['title']} — {authors} ({c['year']})\n"

            prompt = f"""You are a forensic peer reviewer analysing scientific manuscripts for hallucinated citations.

MANUSCRIPT EXCERPT:
{scenario['manuscript_excerpt']}

CITATIONS:
{citations_text}

INSTRUCTIONS:
1. Search the citation database mentally for each cited paper.
2. Check if each citation exists, has correct authors/year, and if the manuscript's claims match the cited paper.
3. If you find a problem, flag the specific citation.

Respond with a JSON object:
{{
  "action_type": "flag_hallucination",
  "citation_id": <int — the ID of the problematic citation>,
  "reason": "<1-2 sentence explanation>"
}}

If all citations are correct, respond:
{{
  "action_type": "approve",
  "citation_id": -1,
  "reason": "All citations verified"
}}

Respond with JSON only. No other text."""

            # Model inference
            if provider == "gemini":
                response = gemini_model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0,
                        "max_output_tokens": 1024,
                        "response_mime_type": "application/json",
                    },
                )
                raw = response.text.strip()

            elif provider in ["openai", "azure"]:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=512,
                )
                raw = response.choices[0].message.content.strip()

            # Clean markdown code fences
            if raw.startswith("```"):
                parts = raw.split("```")
                if len(parts) >= 2:
                    raw = parts[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
            raw = raw.strip()

            # Parse JSON — try multiple strategies
            action_dict = None
            import re
            try:
                action_dict = _json.loads(raw)
            except Exception:
                # Try to find complete JSON object
                json_match = re.search(r'\{[^{}]*"action_type"[^{}]*\}', raw, re.DOTALL)
                if json_match:
                    try:
                        action_dict = _json.loads(json_match.group())
                    except Exception:
                        pass

            # Handle truncated JSON — extract fields with regex
            if action_dict is None:
                action_type_m = re.search(r'"action_type"\s*:\s*"([^"]+)"', raw)
                citation_id_m = re.search(r'"citation_id"\s*:\s*(\d+)', raw)
                reason_m = re.search(r'"reason"\s*:\s*"([^"]*)"', raw)
                action_dict = {
                    "action_type": action_type_m.group(1) if action_type_m else "approve",
                    "citation_id": int(citation_id_m.group(1)) if citation_id_m else -1,
                    "reason": reason_m.group(1) if reason_m else raw[:200],
                }

            score = GRADERS[task_id](action_dict)

            results[task_id] = {
                "difficulty": scenario["difficulty"],
                "score": score,
                "action_type": action_dict.get("action_type", ""),
                "citation_id": action_dict.get("citation_id", -1),
                "reason": action_dict.get("reason", ""),
                "raw_response": raw,
            }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "partial_results": results})

    overall = sum(r["score"] for r in results.values()) / max(len(results), 1)

    return JSONResponse(content={
        "model": model,
        "provider": provider,
        "overall_score": round(overall, 4),
        "task_results": results,
    })
