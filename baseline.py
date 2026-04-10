#!/usr/bin/env python3
"""
Baseline inference script for the Citation Detective environment.

Runs an LLM agent against all 7 tasks via the environment's HTTP API.
Uses OpenAI-compatible API (HF Router, OpenAI, Azure).

Usage:
    # Against a deployed HF Space
    HF_TOKEN=<key> python baseline.py --url https://nikhilsai55000-citation-detective.hf.space

    # Against a local dev server
    HF_TOKEN=<key> python baseline.py --url http://localhost:7860

Environment variables:
    HF_TOKEN         — HuggingFace API token (preferred)
    API_BASE_URL     — OpenAI-compatible API endpoint
    MODEL_NAME       — Model identifier
"""

import argparse
import json
import os
import re
import sys
from typing import Any, Dict

import requests
from openai import OpenAI


def get_client() -> tuple:
    """Initialize the OpenAI client from env vars."""
    api_key = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("API_BASE_URL") or "https://router.huggingface.co/v1"
    model = os.environ.get("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

    if not api_key:
        raise RuntimeError("No API key found. Set HF_TOKEN, API_KEY, or OPENAI_API_KEY.")

    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, model


def get_llm_response(client: OpenAI, model: str, prompt: str) -> str:
    """Get JSON response from the LLM."""
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": (
                    "You are a forensic peer reviewer analyzing scientific manuscripts "
                    "for citation fraud. Detect ghost papers, misattributions, contradictions, "
                    "fabricated statistics, causality reversals, selective omissions, and "
                    "temporal fabrications. Cite specific evidence. Respond with JSON only."
                )},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=512,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"  LLM error: {e}")
        return '{"action_type": "approve", "citation_id": -1, "reason": "LLM error"}'


def parse_action(raw: str) -> Dict[str, Any]:
    """Parse LLM JSON response with fallback strategies."""
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) >= 2:
            raw = parts[1]
            if raw.startswith("json"):
                raw = raw[4:]
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    json_match = re.search(r'\{[^{}]*"action_type"[^{}]*\}', raw, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    action_type_m = re.search(r'"action_type"\s*:\s*"([^"]+)"', raw)
    citation_id_m = re.search(r'"citation_id"\s*:\s*(\d+)', raw)
    reason_m = re.search(r'"reason"\s*:\s*"([^"]*)"', raw)
    return {
        "action_type": action_type_m.group(1) if action_type_m else "approve",
        "citation_id": int(citation_id_m.group(1)) if citation_id_m else -1,
        "reason": reason_m.group(1) if reason_m else raw[:200],
    }


def run_task(base_url: str, task_id: str, client: OpenAI, model: str) -> Dict[str, Any]:
    """Run the full search → flag flow for a single task."""
    # Step 1: Reset to get the scenario
    reset_resp = requests.post(
        f"{base_url}/reset",
        json={"task_id": task_id},
        timeout=30,
    ).json()

    obs = reset_resp["observation"]
    manuscript = obs["manuscript_excerpt"]
    citations = obs["citations_list"]
    search_history = ""
    step_count = 0
    cumulative_reward = 0.0

    # Step 2: Search for each citation
    for citation in citations:
        query = citation["title"]
        step_resp = requests.post(
            f"{base_url}/step",
            json={
                "action": {
                    "task_id": task_id,
                    "action_type": "search",
                    "query": query,
                    "search_history": search_history,
                    "step_count": step_count,
                    "cumulative_reward": cumulative_reward,
                }
            },
            timeout=30,
        ).json()

        step_obs = step_resp.get("observation", {})
        search_result = step_obs.get("search_results", "")
        step_reward = step_resp.get("reward", 0.0)
        search_history += f"\n--- Search for '{query}' ---\n{search_result}\n"
        step_count = step_obs.get("step_count", step_count + 1)
        cumulative_reward += step_reward

    # Step 3: Ask LLM to analyze
    citations_text = "\n".join(
        f"  [{c['id']}] {c['title']} — {', '.join(c['authors'])} ({c['year']})"
        for c in citations
    )

    prompt = f"""Analyze the following manuscript for citation issues.

MANUSCRIPT:
{manuscript}

CITATIONS:
{citations_text}

DATABASE SEARCH RESULTS:
{search_history}

Compare manuscript claims against database entries. Check for:
1. Ghost paper: citation not found in database
2. Identity theft: wrong authors or year
3. Contradiction: manuscript claim contradicts the cited paper
4. Misquoted statistic: manuscript reports a different number
5. Causality reversal: paper shows correlation only, manuscript claims causation
6. Selective omission: manuscript cherry-picks one finding, ignores main conclusion
7. Temporal fabrication: citation with a future year that doesn't exist

In your reason, cite SPECIFIC evidence from the database.

Respond with JSON only:
{{"action_type": "flag_hallucination", "citation_id": <int>, "reason": "<detailed explanation>"}}
Or if all citations are correct:
{{"action_type": "approve", "citation_id": -1, "reason": "All citations verified"}}"""

    raw = get_llm_response(client, model, prompt)
    action_dict = parse_action(raw)

    # Step 4: Submit the terminal action
    step_resp = requests.post(
        f"{base_url}/step",
        json={
            "action": {
                "task_id": task_id,
                "action_type": action_dict.get("action_type", "approve"),
                "citation_id": action_dict.get("citation_id", -1),
                "reason": action_dict.get("reason", ""),
                "search_history": search_history,
                "step_count": step_count,
                "cumulative_reward": cumulative_reward,
            }
        },
        timeout=30,
    ).json()

    return {
        "task_id": task_id,
        "done": step_resp.get("done", False),
        "reward": step_resp.get("reward", 0.0),
        "action_submitted": action_dict,
        "metadata": step_resp.get("observation", {}).get("metadata", {}),
    }


def main():
    parser = argparse.ArgumentParser(description="Citation Detective baseline agent")
    parser.add_argument(
        "--url",
        default="https://nikhilsai55000-citation-detective.hf.space",
        help="Base URL of the environment server",
    )
    args = parser.parse_args()

    base_url = args.url.rstrip("/")

    # Verify server is running
    try:
        health = requests.get(f"{base_url}/health", timeout=10).json()
        print(f"Server health: {health['status']}")
    except Exception as e:
        print(f"Cannot reach server at {base_url}: {e}")
        sys.exit(1)

    # Initialize LLM client
    client, model = get_client()
    print(f"Model: {model}\n")

    # Get tasks
    tasks_resp = requests.get(f"{base_url}/tasks", timeout=10).json()
    task_ids = [t["task_id"] for t in tasks_resp["tasks"]]
    print(f"Tasks: {task_ids} ({len(task_ids)} total)\n")

    # Run each task
    total_reward = 0.0
    for task_id in task_ids:
        print(f"--- {task_id} ---")
        result = run_task(base_url, task_id, client, model)
        reward = result["reward"]
        action = result["action_submitted"]
        print(f"  Score:  {reward:.4f}")
        print(f"  Action: {action.get('action_type', '?')} citation_id={action.get('citation_id', '?')}")
        print(f"  Reason: {action.get('reason', '?')[:120]}")
        total_reward += reward
        print()

    avg_reward = total_reward / max(len(task_ids), 1)
    print(f"{'='*50}")
    print(f"Total score:   {total_reward:.4f}")
    print(f"Average score: {avg_reward:.4f}")
    print(f"All in (0,1):  {'YES' if all(0 < r['reward'] < 1 for r in [result]) else 'CHECK'}")


if __name__ == "__main__":
    main()
