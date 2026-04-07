#!/usr/bin/env python3
"""
Inference Script — Citation Detective (Forensic Peer Reviewer)
================================================================
MANDATORY ENV VARS (provided by the validator):
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration from environment variables (set by the validator)
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

ENV_URL = os.getenv("ENV_URL") or "https://nikhilsai55000-citation-detective.hf.space"
BENCHMARK = "citation_detective"
MAX_STEPS = 6  # up to 5 searches + 1 terminal action


# ---------------------------------------------------------------------------
# Structured logging helpers (MANDATORY format)
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM call using OpenAI Client (MANDATORY: must use OpenAI Client)
# ---------------------------------------------------------------------------
def get_llm_response(client: OpenAI, prompt: str) -> str:
    """Call the LLM via OpenAI-compatible API."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a forensic peer reviewer analyzing scientific manuscripts for citation fraud. Respond with JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=512,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else '{"action_type": "approve", "citation_id": -1, "reason": "No response"}'
    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", flush=True)
        return '{"action_type": "approve", "citation_id": -1, "reason": "LLM error"}'


# ---------------------------------------------------------------------------
# JSON parsing with fallback
# ---------------------------------------------------------------------------
def parse_action(raw: str) -> Dict[str, Any]:
    """Parse LLM JSON response with multiple fallback strategies."""
    raw = raw.strip()
    # Strip markdown code fences
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) >= 2:
            inner = parts[1]
            if inner.startswith("json"):
                inner = inner[4:]
            raw = inner.strip()

    # Try direct JSON parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try regex for complete JSON object
    m = re.search(r'\{[^{}]*"action_type"[^{}]*\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    # Extract fields individually
    action_type_m = re.search(r'"action_type"\s*:\s*"([^"]+)"', raw)
    citation_id_m = re.search(r'"citation_id"\s*:\s*(\d+)', raw)
    reason_m = re.search(r'"reason"\s*:\s*"([^"]*)"', raw)
    return {
        "action_type": action_type_m.group(1) if action_type_m else "approve",
        "citation_id": int(citation_id_m.group(1)) if citation_id_m else -1,
        "reason": reason_m.group(1) if reason_m else raw[:200],
    }


# ---------------------------------------------------------------------------
# Run a single task through the RL loop
# ---------------------------------------------------------------------------
def run_task(client: OpenAI, base_url: str, task_id: str) -> Dict[str, Any]:
    """
    RL loop for one task:
      1. POST /reset       → get manuscript + citations
      2. POST /step search → search each citation in DB
      3. LLM decides       → flag_hallucination or approve
      4. POST /step flag   → terminal action, scored
    """
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Step 1: Reset
        reset_resp = requests.post(
            f"{base_url}/reset", json={"task_id": task_id}, timeout=60,
        ).json()

        obs = reset_resp.get("observation", {})
        manuscript = obs.get("manuscript_excerpt", "")
        citations = obs.get("citations_list", [])
        search_history = ""
        step_count = 0
        cumulative_reward = 0.0

        # Step 2: Search each citation via env.step()
        for citation in citations:
            query = citation.get("title", "")
            step_resp = requests.post(
                f"{base_url}/step",
                json={"action": {
                    "task_id": task_id,
                    "action_type": "search",
                    "query": query,
                    "search_history": search_history,
                    "step_count": step_count,
                    "cumulative_reward": cumulative_reward,
                }},
                timeout=60,
            ).json()

            step_obs = step_resp.get("observation", {})
            step_reward = step_resp.get("reward", 0.0)
            step_done = step_resp.get("done", False)
            search_result = step_obs.get("search_results", "")
            error_msg = step_obs.get("metadata", {}).get("error") if isinstance(step_obs.get("metadata"), dict) else None

            search_history += f"\n--- Search: '{query}' ---\n{search_result}\n"
            step_count = step_obs.get("step_count", step_count + 1)
            cumulative_reward += step_reward

            steps_taken += 1
            rewards.append(step_reward)
            log_step(step=steps_taken, action=f"search('{query[:40]}')", reward=step_reward, done=step_done, error=error_msg)

            if step_done:
                break

        # Step 3: LLM analysis (using OpenAI Client)
        if not step_done:
            citations_text = "\n".join(
                f"  [{c['id']}] {c['title']} — {', '.join(c.get('authors', []))} ({c.get('year', '?')})"
                for c in citations
            )
            prompt = f"""Analyze the following manuscript for citation issues.

MANUSCRIPT:
{manuscript}

CITATIONS:
{citations_text}

DATABASE SEARCH RESULTS:
{search_history}

Based on the search results, determine if any citation is:
- A ghost paper (not found in the database at all)
- Misattributed (wrong authors or year vs the DB entry)
- Contradicting (the manuscript's claim contradicts what the cited paper actually says)
- Misquoted statistic (manuscript reports a different number than the cited paper)
- Causality reversal (paper shows only correlation, manuscript claims proven causation)

Respond with JSON only:
{{"action_type": "flag_hallucination", "citation_id": <int>, "reason": "<1-2 sentence explanation>"}}
Or if all citations are correct:
{{"action_type": "approve", "citation_id": -1, "reason": "All citations verified"}}"""

            raw = get_llm_response(client, prompt)
            action_dict = parse_action(raw)

            # Step 4: Submit terminal action via env.step()
            term_resp = requests.post(
                f"{base_url}/step",
                json={"action": {
                    "task_id": task_id,
                    "action_type": action_dict.get("action_type", "approve"),
                    "citation_id": action_dict.get("citation_id", -1),
                    "reason": action_dict.get("reason", ""),
                    "search_history": search_history,
                    "step_count": step_count,
                    "cumulative_reward": cumulative_reward,
                }},
                timeout=60,
            ).json()

            term_reward = term_resp.get("reward", 0.0)
            term_done = term_resp.get("done", True)
            term_action_str = f"{action_dict.get('action_type', 'approve')}(cid={action_dict.get('citation_id', -1)})"

            steps_taken += 1
            rewards.append(term_reward)
            log_step(step=steps_taken, action=term_action_str, reward=term_reward, done=term_done, error=None)

        # Compute score: normalize to [0, 1]
        # Terminal reward is the main signal (can range from -1.0 to +1.1)
        # We use the final step's reward, clamped to [0, 1]
        final_reward = rewards[-1] if rewards else 0.0
        score = min(max(final_reward, 0.0), 1.0)
        success = score > 0.0

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} error: {exc}", flush=True)
        if not rewards:
            rewards.append(0.0)
        steps_taken = max(steps_taken, 1)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return {"task_id": task_id, "score": score, "rewards": rewards, "steps": steps_taken}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    base_url = ENV_URL.rstrip("/")

    # Initialize OpenAI client (MANDATORY)
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Check server health
    try:
        health = requests.get(f"{base_url}/health", timeout=30).json()
        print(f"[DEBUG] Server: {base_url} | Status: {health.get('status', 'unknown')}", flush=True)
    except Exception as e:
        print(f"[DEBUG] Health check failed: {e}", flush=True)
        # Don't exit — try running tasks anyway

    # Get task list
    try:
        tasks_resp = requests.get(f"{base_url}/tasks", timeout=30).json()
        task_ids = [t["task_id"] for t in tasks_resp.get("tasks", [])]
    except Exception:
        # Fallback to known task IDs
        task_ids = ["task_1", "task_2", "task_3", "task_4", "task_5"]

    print(f"[DEBUG] Tasks: {task_ids}", flush=True)

    # Run each task
    all_results = []
    for task_id in task_ids:
        result = run_task(client, base_url, task_id)
        all_results.append(result)

    # Summary
    avg_score = sum(r["score"] for r in all_results) / max(len(all_results), 1)
    print(f"[DEBUG] Overall average score: {avg_score:.4f}", flush=True)


if __name__ == "__main__":
    main()
