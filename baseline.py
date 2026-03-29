#!/usr/bin/env python3
"""
Baseline inference script for the Citation Detective environment.

Runs an LLM agent against all 3 tasks via the environment's HTTP API.
Supports Gemini (default), OpenAI, and Azure OpenAI.

Usage:
    # Against a local dev server
    GEMINI_API_KEY=<key> python baseline.py --url http://localhost:8000

    # Against a deployed HF Space
    GEMINI_API_KEY=<key> python baseline.py --url https://nikhilsai55000-citation-detective.hf.space

Environment variables:
    GEMINI_API_KEY       — Google Gemini API key (preferred)
    OPENAI_API_KEY       — OpenAI API key (fallback)
    AZURE_OPENAI_API_KEY — Azure OpenAI API key (fallback)
"""

import argparse
import json
import os
import sys
from typing import Any, Dict

import requests


def get_llm_response(prompt: str) -> str:
    """Get a response from the configured LLM provider."""
    gemini_key = os.environ.get("GEMINI_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    azure_key = os.environ.get("AZURE_OPENAI_API_KEY")

    if gemini_key:
        import google.generativeai as genai
        genai.configure(api_key=gemini_key)
        model = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
        gemini_model = genai.GenerativeModel(model)
        response = gemini_model.generate_content(
            prompt,
            generation_config={
                "temperature": 0,
                "max_output_tokens": 1024,
                "response_mime_type": "application/json",
            },
        )
        return response.text.strip()

    elif openai_key:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        model = os.environ.get("OPENAI_MODEL", "gpt-4o")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()

    elif azure_key:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=azure_key,
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        )
        model = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()

    else:
        raise RuntimeError(
            "No API key found. Set GEMINI_API_KEY, OPENAI_API_KEY, or AZURE_OPENAI_API_KEY."
        )


def clean_json_response(raw: str) -> Dict[str, Any]:
    """Parse LLM response, stripping markdown code fences if present."""
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

    # Try to find JSON object in the response via regex
    import re
    json_match = re.search(r'\{[^{}]*"action_type"[^{}]*\}', raw, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Handle truncated JSON — extract individual fields with regex
    action_type_m = re.search(r'"action_type"\s*:\s*"([^"]+)"', raw)
    citation_id_m = re.search(r'"citation_id"\s*:\s*(\d+)', raw)
    reason_m = re.search(r'"reason"\s*:\s*"([^"]*)"', raw)
    return {
        "action_type": action_type_m.group(1) if action_type_m else "approve",
        "citation_id": int(citation_id_m.group(1)) if citation_id_m else -1,
        "reason": reason_m.group(1) if reason_m else raw[:200],
    }


def run_task(base_url: str, task_id: str) -> Dict[str, Any]:
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
                }
            },
            timeout=30,
        ).json()

        search_result = step_resp["observation"].get("search_results", "")
        search_history += f"\n--- Search for '{query}' ---\n{search_result}\n"

    # Step 3: Ask LLM to analyze
    citations_text = ""
    for c in citations:
        authors = ", ".join(c["authors"])
        citations_text += f"  [{c['id']}] {c['title']} — {authors} ({c['year']})\n"

    prompt = f"""You are a forensic peer reviewer. Analyze the manuscript for hallucinated citations.

MANUSCRIPT:
{manuscript}

CITATIONS:
{citations_text}

DATABASE SEARCH RESULTS:
{search_history}

Based on the search results, determine if any citation is:
- A ghost paper (doesn't exist in the database)
- Misattributed (wrong authors or year)
- Contradicting (manuscript claim contradicts the cited paper)

Respond with JSON:
{{
  "action_type": "flag_hallucination",
  "citation_id": <int>,
  "reason": "<explanation>"
}}

Or if all citations check out:
{{
  "action_type": "approve",
  "citation_id": -1,
  "reason": "All citations verified"
}}

JSON only, no other text."""

    raw = get_llm_response(prompt)
    action_dict = clean_json_response(raw)

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
            }
        },
        timeout=30,
    ).json()

    return {
        "task_id": task_id,
        "done": step_resp.get("done", False),
        "reward": step_resp.get("reward", 0.0),
        "action_submitted": action_dict,
        "raw_llm_response": raw,
    }


def main():
    parser = argparse.ArgumentParser(description="Citation Detective baseline agent")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
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

    # Get tasks
    tasks_resp = requests.get(f"{base_url}/tasks", timeout=10).json()
    task_ids = [t["task_id"] for t in tasks_resp["tasks"]]
    print(f"Tasks: {task_ids}\n")

    # Run each task
    total_reward = 0.0
    for task_id in task_ids:
        print(f"--- Running {task_id} ---")
        result = run_task(base_url, task_id)
        print(f"  Reward: {result['reward']}")
        print(f"  Action: {result['action_submitted'].get('action_type', '?')} "
              f"citation_id={result['action_submitted'].get('citation_id', '?')}")
        print(f"  Reason: {result['action_submitted'].get('reason', '?')[:100]}")
        total_reward += result["reward"]
        print()

    avg_reward = total_reward / max(len(task_ids), 1)
    print(f"=== Overall ===")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average reward: {avg_reward:.4f}")


if __name__ == "__main__":
    main()
