"""
Citation Detective (Forensic Peer Reviewer) Environment Implementation.

An agent reviews a scientific manuscript excerpt and its citation list.
It can search a mock database, then either flag a hallucinated citation
or approve the manuscript. Seven tasks of increasing difficulty:

  task_1 — easy   — The Ghost Paper (citation doesn't exist)
  task_2 — medium — The Identity Theft (wrong authors/year)
  task_3 — hard   — The Contradiction (claim contradicts cited paper)
  task_4 — medium — The Misquoted Statistic (fabricated number)
  task_5 — hard   — The Causality Reversal (correlation → causation)
  task_6 — hard   — The Selective Omission (cherry-picked findings)
  task_7 — expert — The Temporal Fabrication (future paper that doesn't exist)

Design decisions:
  - Multi-step episodes: search → analyze → flag/approve
  - Composite reward: BASE + IDENTIFICATION + REASON_QUALITY ∈ (0.05, 0.90)
  - Efficiency bonus: fewer steps → slightly higher terminal score
  - Ground truth feedback: terminal observations include the correct answer
  - Random task selection: reset() without task_id picks a random task
  - Stateless HTTP support: state hydrated from action payload each step
"""

import random
from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ForensicAction, ForensicObservation
except ImportError:
    from models import ForensicAction, ForensicObservation

try:
    from .graders import GRADERS, SCENARIOS, search_database
except ImportError:
    from graders import GRADERS, SCENARIOS, search_database


class CitationDetectiveEnvironment(Environment):
    """
    Forensic peer-review environment.

    Each episode:
      reset(task_id) — returns manuscript + citations
      step(search)   — returns database search results (up to MAX_STEPS)
      step(flag/approve) — terminal action, graded and scored

    Reward design:
      - Search steps: +0.1 (found results), -0.1 (no results)
      - Terminal: composite grader score in (0.05, 0.90)
      - Efficiency bonus: up to +0.05 for solving in fewer steps

    Supports concurrent WebSocket sessions.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    MAX_STEPS: int = 5

    # Efficiency bonus: reward solving quickly (max +0.05 for 1-step solve)
    EFFICIENCY_BONUS_PER_STEP_SAVED: float = 0.01

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task_id: str = "task_1"
        self._episode_done: bool = False
        self._cumulative_reward: float = 0.0

    def reset(
        self,
        task_id: Optional[str] = None,
        episode_id: Optional[str] = None,
    ) -> ForensicObservation:
        """
        Start a new episode.

        Args:
            task_id: Specific task to run. If None, picks a random task
                     (useful for RL training with diverse episodes).
            episode_id: Optional episode ID for tracking.

        Returns:
            ForensicObservation with manuscript and citations.
        """
        # Random task selection for RL training
        if task_id is None or task_id not in SCENARIOS:
            task_id = random.choice(list(SCENARIOS.keys()))

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._episode_done = False
        self._cumulative_reward = 0.0
        self._current_task_id = task_id

        scenario = SCENARIOS[task_id]

        return ForensicObservation(
            manuscript_excerpt=scenario["manuscript_excerpt"],
            citations_list=scenario["citations_list"],
            search_results="",
            step_count=0,
            task_id=task_id,
            done=False,
            reward=0.0,
            metadata={
                "difficulty": scenario["difficulty"],
                "description": scenario["description"],
                "num_citations": len(scenario["citations_list"]),
                "max_steps": self.MAX_STEPS,
            },
        )

    def step(self, action: ForensicAction) -> ForensicObservation:  # type: ignore[override]
        """
        Execute one step.

        action_type='search': query database, return results
        action_type='flag_hallucination': terminal — grade and end
        action_type='approve': terminal — penalize missed hallucination

        NOTE: For stateless HTTP, step_count and cumulative_reward are
        hydrated from the action payload (client passes them back each call).
        """
        # Hydrate state from client — bypasses stateless HTTP env recreation
        self._state.step_count = action.step_count + 1
        self._cumulative_reward = action.cumulative_reward

        # Use task_id from action (stateless HTTP fix)
        task_id = action.task_id
        if task_id not in SCENARIOS:
            task_id = "task_1"

        scenario = SCENARIOS[task_id]

        # --- Already done? ---
        if self._episode_done:
            return ForensicObservation(
                manuscript_excerpt="",
                citations_list=[],
                search_results="Episode already finished. Call reset() to start a new one.",
                step_count=self._state.step_count,
                task_id=task_id,
                done=True,
                reward=0.05,  # minimum valid score
            )

        action_type = action.action_type.strip().lower()

        # =================================================================
        # ACTION: search
        # =================================================================
        if action_type == "search":
            results = search_database(action.query)

            # Reward shaping for search quality
            if results and results != "No results found." and "No results found" not in results:
                search_reward = 0.1  # good search — found relevant entries
            else:
                search_reward = -0.1  # unproductive search

            self._cumulative_reward += search_reward

            # Check step limit
            if self._state.step_count >= self.MAX_STEPS:
                self._episode_done = True
                return ForensicObservation(
                    manuscript_excerpt=scenario["manuscript_excerpt"],
                    citations_list=scenario["citations_list"],
                    search_results=results,
                    step_count=self._state.step_count,
                    task_id=task_id,
                    done=True,
                    reward=0.05,  # timed out — minimum score
                    metadata={
                        "info": "Maximum steps reached. Episode ended with no verdict.",
                        "ground_truth": scenario["ground_truth"],
                    },
                )

            return ForensicObservation(
                manuscript_excerpt=scenario["manuscript_excerpt"],
                citations_list=scenario["citations_list"],
                search_results=results,
                step_count=self._state.step_count,
                task_id=task_id,
                done=False,
                reward=search_reward,
                metadata={
                    "steps_remaining": self.MAX_STEPS - self._state.step_count,
                },
            )

        # =================================================================
        # ACTION: flag_hallucination  (terminal)
        # =================================================================
        if action_type == "flag_hallucination":
            self._episode_done = True

            action_dict = {
                "action_type": action.action_type,
                "citation_id": action.citation_id,
                "reason": action.reason,
            }
            grader = GRADERS[task_id]
            base_score = grader(action_dict)

            # Efficiency bonus: reward faster investigation
            # Max bonus = (MAX_STEPS - 1) * BONUS_PER_STEP = 4 * 0.01 = 0.04
            steps_used = self._state.step_count
            steps_saved = max(0, self.MAX_STEPS - steps_used)
            efficiency_bonus = steps_saved * self.EFFICIENCY_BONUS_PER_STEP_SAVED

            # Final score: grader + efficiency, capped at 0.99
            score = min(base_score + efficiency_bonus, 0.99)

            gt = scenario["ground_truth"]

            return ForensicObservation(
                manuscript_excerpt=scenario["manuscript_excerpt"],
                citations_list=scenario["citations_list"],
                search_results=(
                    f"Episode complete. "
                    f"Grader score: {base_score:.4f}. "
                    f"Efficiency bonus: +{efficiency_bonus:.4f}. "
                    f"Final score: {score:.4f}."
                ),
                step_count=self._state.step_count,
                task_id=task_id,
                done=True,
                reward=score,
                metadata={
                    "grader_score": base_score,
                    "efficiency_bonus": efficiency_bonus,
                    "final_score": score,
                    "steps_used": steps_used,
                    "citation_id_submitted": action.citation_id,
                    "reason_submitted": action.reason,
                    # Ground truth for learning
                    "correct_citation_id": gt["hallucinated_citation_id"],
                    "correct_issue_type": gt["issue_type"],
                    "correct_explanation": gt["explanation"],
                    "cumulative_search_reward": self._cumulative_reward,
                },
            )

        # =================================================================
        # ACTION: approve  (terminal)
        # =================================================================
        if action_type == "approve":
            self._episode_done = True

            action_dict = {
                "action_type": action.action_type,
                "citation_id": action.citation_id,
                "reason": action.reason,
            }
            grader = GRADERS[task_id]
            score = grader(action_dict)  # returns 0.05 for approve

            gt = scenario["ground_truth"]

            return ForensicObservation(
                manuscript_excerpt=scenario["manuscript_excerpt"],
                citations_list=scenario["citations_list"],
                search_results=(
                    f"Episode complete. "
                    f"You approved the manuscript, but it contained a "
                    f"hallucinated/problematic citation ({gt['issue_type']}). "
                    f"Grader score: {score:.4f}."
                ),
                step_count=self._state.step_count,
                task_id=task_id,
                done=True,
                reward=score,
                metadata={
                    "grader_score": score,
                    # Ground truth for learning
                    "correct_citation_id": gt["hallucinated_citation_id"],
                    "correct_issue_type": gt["issue_type"],
                    "correct_explanation": gt["explanation"],
                    "cumulative_search_reward": self._cumulative_reward,
                },
            )

        # =================================================================
        # Unknown action type
        # =================================================================
        return ForensicObservation(
            manuscript_excerpt=scenario["manuscript_excerpt"],
            citations_list=scenario["citations_list"],
            search_results=f"Unknown action_type '{action.action_type}'. Use 'search', 'flag_hallucination', or 'approve'.",
            step_count=self._state.step_count,
            task_id=task_id,
            done=False,
            reward=0.0,
        )

    @property
    def state(self) -> State:
        """Return current episode metadata."""
        return self._state
