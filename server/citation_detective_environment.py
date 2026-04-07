"""
Citation Detective (Forensic Peer Reviewer) Environment Implementation.

An agent reviews a scientific manuscript excerpt and its citation list.
It can search a mock database, then either flag a hallucinated citation
or approve the manuscript. Three tasks of increasing difficulty:

  task_1 — easy   — The Ghost Paper (citation doesn't exist)
  task_2 — medium — The Identity Theft (wrong authors/year)
  task_3 — hard   — The Contradiction (claim contradicts cited paper)
"""

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

    Supports concurrent WebSocket sessions.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    MAX_STEPS: int = 5

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task_id: str = "task_1"
        self._episode_done: bool = False
        self._cumulative_reward: float = 0.0

    def reset(self, task_id: str = "task_1") -> ForensicObservation:
        """
        Start a new episode for a specific task.

        Args:
            task_id: "task_1", "task_2", or "task_3"

        Returns:
            ForensicObservation with manuscript and citations.
        """
        if task_id not in SCENARIOS:
            task_id = "task_1"

        self._state = State(episode_id=str(uuid4()), step_count=0)
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
                reward=0.0,
            )

        action_type = action.action_type.strip().lower()

        # =================================================================
        # ACTION: search
        # =================================================================
        if action_type == "search":
            results = search_database(action.query)

            # Reward shaping for search quality
            if results and results != "No results found." and "No results found" not in results:
                search_reward = 0.1  # good search
            else:
                search_reward = -0.1  # bad/empty search

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
                    reward=0.0,  # timed out
                    metadata={"info": "Maximum steps reached. Episode ended with no verdict."},
                )

            return ForensicObservation(
                manuscript_excerpt=scenario["manuscript_excerpt"],
                citations_list=scenario["citations_list"],
                search_results=results,
                step_count=self._state.step_count,
                task_id=task_id,
                done=False,
                reward=search_reward,
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
            score = grader(action_dict)

            # Terminal reward IS the grader score (always in (0, 1))
            # Search rewards were intermediate signals; the grader score
            # is the definitive episode outcome
            gt_id = scenario["ground_truth"]["hallucinated_citation_id"]

            return ForensicObservation(
                manuscript_excerpt=scenario["manuscript_excerpt"],
                citations_list=scenario["citations_list"],
                search_results=(
                    f"Episode complete. "
                    f"Grader score: {score:.4f}. "
                    f"Correct citation: {gt_id}. "
                    f"Your citation: {action.citation_id}."
                ),
                step_count=self._state.step_count,
                task_id=task_id,
                done=True,
                reward=score,
                metadata={
                    "grader_score": score,
                    "citation_id_submitted": action.citation_id,
                    "reason_submitted": action.reason,
                    "correct_citation_id": gt_id,
                    "cumulative_search_reward": self._cumulative_reward,
                },
            )

        # =================================================================
        # ACTION: approve  (terminal)
        # =================================================================
        if action_type == "approve":
            self._episode_done = True

            # Run through grader — approve gets low score (0.05) since
            # all tasks have hallucinations
            action_dict = {
                "action_type": action.action_type,
                "citation_id": action.citation_id,
                "reason": action.reason,
            }
            grader = GRADERS[task_id]
            score = grader(action_dict)

            return ForensicObservation(
                manuscript_excerpt=scenario["manuscript_excerpt"],
                citations_list=scenario["citations_list"],
                search_results=(
                    f"Episode complete. "
                    f"You approved the manuscript, but it contained a "
                    f"hallucinated/problematic citation. "
                    f"Grader score: {score:.4f}."
                ),
                step_count=self._state.step_count,
                task_id=task_id,
                done=True,
                reward=score,
                metadata={
                    "grader_score": score,
                    "issue_type": scenario["ground_truth"]["issue_type"],
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
