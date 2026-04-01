# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Citation Detective (Forensic Peer Reviewer) Environment.

An agent reviews scientific manuscript excerpts to detect hallucinated,
misattributed, or contradicting citations by cross-referencing against
a mock database.
"""

from typing import List, Dict, Any, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class ForensicAction(Action):
    """
    Action the agent takes during a forensic peer-review episode.

    Three action types:
      - "search"             : query the citation database
      - "flag_hallucination" : flag a specific citation as problematic
      - "approve"            : approve the manuscript as citation-clean
    """

    task_id: str = Field(
        default="task_1",
        description="Which task to run (task_1, task_2, task_3). "
                    "Required for stateless HTTP — passed every step.",
    )
    action_type: str = Field(
        ...,
        description="One of: 'search', 'flag_hallucination', 'approve'",
    )
    query: str = Field(
        default="",
        description="Search query string (used when action_type='search')",
    )
    citation_id: int = Field(
        default=-1,
        description="ID of the citation to flag (used when action_type='flag_hallucination')",
    )
    reason: str = Field(
        default="",
        description="Explanation for flagging (used when action_type='flag_hallucination')",
    )
    search_history: str = Field(
        default="",
        description="Accumulated search results from previous steps, "
                    "passed back by the agent so the environment can track "
                    "history in stateless HTTP mode.",
    )
    step_count: int = Field(
        default=0,
        description="Steps taken so far — passed back by client each step "
                    "to track progress across stateless HTTP calls.",
    )
    cumulative_reward: float = Field(
        default=0.0,
        description="Running reward total — passed back by client each step "
                    "to accumulate search rewards across stateless HTTP calls.",
    )


class ForensicObservation(Observation):
    """
    What the agent observes at each step.

    On reset: manuscript_excerpt + citations_list, empty search_results.
    After search: search_results populated with database matches.
    On terminal action: done=True with final reward.
    """

    manuscript_excerpt: str = Field(
        default="",
        description="2-3 paragraph scientific text with inline citations [1], [2], etc.",
    )
    citations_list: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of citation dicts, each with id, title, authors, year.",
    )
    search_results: str = Field(
        default="",
        description="Output of the agent's previous database search (empty on first step).",
    )
    step_count: int = Field(
        default=0,
        description="Number of actions taken so far in this episode.",
    )
    task_id: str = Field(
        default="",
        description="Current task identifier.",
    )
