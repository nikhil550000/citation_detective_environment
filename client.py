# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Citation Detective Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import ForensicAction, ForensicObservation


class CitationDetectiveEnv(
    EnvClient[ForensicAction, ForensicObservation, State]
):
    """
    Client for the Citation Detective Environment.

    Example:
        >>> with CitationDetectiveEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     result = client.step(ForensicAction(
        ...         task_id="task_1", action_type="search", query="Neural Pathways"
        ...     ))
    """

    def _step_payload(self, action: ForensicAction) -> Dict:
        """Convert ForensicAction to JSON payload for step message."""
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[ForensicObservation]:
        """Parse server response into StepResult[ForensicObservation]."""
        obs_data = payload.get("observation", {})
        observation = ForensicObservation(
            manuscript_excerpt=obs_data.get("manuscript_excerpt", ""),
            citations_list=obs_data.get("citations_list", []),
            search_results=obs_data.get("search_results", ""),
            step_count=obs_data.get("step_count", 0),
            task_id=obs_data.get("task_id", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
