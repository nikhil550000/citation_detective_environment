"""
Graders and mock database for the Citation Detective environment.

Contains:
  - MOCK_DATABASE: hardcoded citation database for deterministic grading
  - SCENARIOS: the 3 tasks (easy/medium/hard) with manuscript excerpts
  - search_database(): fuzzy title/author/abstract search
  - grade_task_1/2/3(): grader functions returning 0.0 - 1.0
  - GRADERS: dict mapping task_id -> grader function
"""

from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Mock citation database — deterministic, no external APIs
# ---------------------------------------------------------------------------

MOCK_DATABASE = {
    "johnson_2021": {
        "title": "Deep Learning Applications in Medical Imaging",
        "authors": ["Johnson, A.", "Smith, B.", "Lee, C."],
        "year": 2021,
        "abstract": (
            "This paper presents novel deep learning architectures for "
            "medical image segmentation. We demonstrate state-of-the-art "
            "performance on three benchmark datasets using a modified U-Net "
            "architecture with attention mechanisms."
        ),
    },
    "patel_2019": {
        "title": "Drug X shows no significant effect on Disease Y",
        "authors": ["Patel, R.", "Kumar, S."],
        "year": 2019,
        "abstract": (
            "In a double-blind trial of 500 patients, Drug X showed no "
            "significant effect on Disease Y outcomes compared to placebo "
            "(p=0.43). Secondary endpoints including symptom severity scores "
            "and quality of life measures also showed no meaningful improvement."
        ),
    },
    "williams_2022": {
        "title": "Transformer Models for Natural Language Understanding",
        "authors": ["Williams, E.", "Chen, F."],
        "year": 2022,
        "abstract": (
            "We present a comprehensive survey of transformer architectures "
            "for natural language understanding tasks. Our analysis covers "
            "BERT, GPT, and T5 family models across 12 benchmark tasks."
        ),
    },
}


# ---------------------------------------------------------------------------
# Scenarios — the raw data served to agents
# ---------------------------------------------------------------------------

SCENARIOS = {
    "task_1": {
        "difficulty": "easy",
        "description": "The Ghost Paper — a citation that does not exist in the database.",
        "manuscript_excerpt": (
            "Recent advances in neuroscience have leveraged deep learning to map "
            "cognitive processes with unprecedented accuracy. Several studies have "
            "demonstrated the potential of neural network models to predict brain "
            "activity patterns from functional MRI data [1]. Furthermore, groundbreaking "
            "work on neural pathway modeling has shown that artificial neural networks "
            "can simulate cognitive enhancement processes with remarkable fidelity [2]. "
            "These findings suggest a convergence between computational neuroscience "
            "and artificial intelligence that may revolutionize our understanding of "
            "human cognition.\n\n"
            "The application of transformer architectures to neuroscience data has "
            "also yielded promising results [3], enabling researchers to process "
            "large-scale brain imaging datasets more efficiently than traditional methods."
        ),
        "citations_list": [
            {
                "id": 1,
                "title": "Deep Learning Applications in Medical Imaging",
                "authors": ["Johnson, A.", "Smith, B.", "Lee, C."],
                "year": 2021,
            },
            {
                "id": 2,
                "title": "Neural Pathways in Cognitive Enhancement",
                "authors": ["Brown, T."],
                "year": 2020,
            },
            {
                "id": 3,
                "title": "Transformer Models for Natural Language Understanding",
                "authors": ["Williams, E.", "Chen, F."],
                "year": 2022,
            },
        ],
        "ground_truth": {
            "hallucinated_citation_id": 2,
            "issue_type": "ghost_paper",
            "explanation": (
                "Citation [2] 'Neural Pathways in Cognitive Enhancement' by Brown, T. (2020) "
                "does not exist in the database. It is a completely fabricated reference."
            ),
        },
    },
    "task_2": {
        "difficulty": "medium",
        "description": "The Identity Theft — real paper title but wrong authors and year.",
        "manuscript_excerpt": (
            "Deep learning has transformed medical image analysis, enabling automated "
            "detection and segmentation of pathological structures with human-level "
            "accuracy. Pioneering work by Brown and Davis demonstrated the efficacy "
            "of convolutional neural networks for medical image segmentation [1], "
            "achieving state-of-the-art results on multiple benchmark datasets.\n\n"
            "Building on these foundations, subsequent research has explored the "
            "application of attention mechanisms and transformer architectures to "
            "further improve segmentation accuracy [2]. The combination of these "
            "approaches has led to significant advances in clinical diagnostic tools."
        ),
        "citations_list": [
            {
                "id": 1,
                "title": "Deep Learning Applications in Medical Imaging",
                "authors": ["Brown, T.", "Davis, K."],
                "year": 2018,
            },
            {
                "id": 2,
                "title": "Transformer Models for Natural Language Understanding",
                "authors": ["Williams, E.", "Chen, F."],
                "year": 2022,
            },
        ],
        "ground_truth": {
            "hallucinated_citation_id": 1,
            "issue_type": "identity_theft",
            "explanation": (
                "Citation [1] has the correct title 'Deep Learning Applications in "
                "Medical Imaging' but the wrong authors (Brown, T. and Davis, K. instead "
                "of Johnson, A., Smith, B., and Lee, C.) and wrong year (2018 instead "
                "of 2021)."
            ),
        },
    },
    "task_3": {
        "difficulty": "hard",
        "description": "The Contradiction — correct citation but the manuscript claim contradicts the paper.",
        "manuscript_excerpt": (
            "Pharmacological interventions for Disease Y have shown considerable promise "
            "in recent clinical trials. Notably, Patel and Kumar demonstrated that Drug X "
            "significantly reduces symptoms of Disease Y in a large-scale double-blind "
            "trial [1]. Their study of 500 patients showed statistically significant "
            "improvements in both primary and secondary outcome measures, establishing "
            "Drug X as a viable treatment option.\n\n"
            "These positive findings have spurred further investigation into related "
            "compounds, with several Phase III trials currently underway. The success "
            "of Drug X represents a major breakthrough in Disease Y treatment."
        ),
        "citations_list": [
            {
                "id": 1,
                "title": "Drug X shows no significant effect on Disease Y",
                "authors": ["Patel, R.", "Kumar, S."],
                "year": 2019,
            },
        ],
        "ground_truth": {
            "hallucinated_citation_id": 1,
            "issue_type": "contradiction",
            "explanation": (
                "Citation [1] is correctly attributed (right title, authors, year), "
                "but the manuscript claims 'Drug X significantly reduces symptoms of "
                "Disease Y' while the actual paper states 'Drug X showed no significant "
                "effect on Disease Y (p=0.43)'. This is a direct contradiction."
            ),
        },
    },
}


# ---------------------------------------------------------------------------
# Database search function
# ---------------------------------------------------------------------------

def search_database(query: str) -> str:
    """
    Search the mock database by case-insensitive substring match
    on title, authors, and abstract.

    Args:
        query: search string

    Returns:
        Formatted string with matching entries, or "No results found."
    """
    if not query or not query.strip():
        return "No results found. Please provide a search query."

    query_lower = query.strip().lower()
    results: List[str] = []

    for key, entry in MOCK_DATABASE.items():
        # Check title
        title_match = query_lower in entry["title"].lower()
        # Check authors
        author_match = any(query_lower in a.lower() for a in entry["authors"])
        # Check abstract
        abstract_match = query_lower in entry["abstract"].lower()

        if title_match or author_match or abstract_match:
            authors_str = ", ".join(entry["authors"])
            results.append(
                f"--- DATABASE ENTRY: {key} ---\n"
                f"Title: {entry['title']}\n"
                f"Authors: {authors_str}\n"
                f"Year: {entry['year']}\n"
                f"Abstract: {entry['abstract']}\n"
            )

    if results:
        return "\n".join(results)
    return "No results found."


# ---------------------------------------------------------------------------
# Grader functions — each returns float 0.0 to 1.0
# ---------------------------------------------------------------------------

def grade_task_1(action_dict: Dict[str, Any]) -> float:
    """
    Grade task 1 — The Ghost Paper (easy).

    Full credit (1.0): flag_hallucination on citation_id=2
    Partial (0.3): flag_hallucination on wrong citation_id
    Penalty (-0.5): flag wrong citation (clamped to 0.0)
    """
    gt = SCENARIOS["task_1"]["ground_truth"]
    action_type = str(action_dict.get("action_type", "")).strip().lower()
    citation_id = action_dict.get("citation_id", -1)

    if action_type == "approve":
        return 0.0  # -1.0 reward handled in environment, grader returns 0.0

    if action_type == "flag_hallucination":
        if citation_id == gt["hallucinated_citation_id"]:
            return 1.0
        else:
            return 0.3  # flagged something, just wrong one

    return 0.0


def grade_task_2(action_dict: Dict[str, Any]) -> float:
    """
    Grade task 2 — The Identity Theft (medium).

    Full credit (1.0): correct citation_id + reason mentions author/year mismatch
    High (0.7): correct citation_id + reason mentions either author or year
    Partial (0.5): correct citation_id, no relevant reason
    Low (0.3): wrong citation_id flagged
    """
    gt = SCENARIOS["task_2"]["ground_truth"]
    action_type = str(action_dict.get("action_type", "")).strip().lower()
    citation_id = action_dict.get("citation_id", -1)
    reason = str(action_dict.get("reason", "")).strip().lower()

    if action_type == "approve":
        return 0.0

    if action_type == "flag_hallucination":
        if citation_id == gt["hallucinated_citation_id"]:
            # Check reason quality
            author_keywords = ["author", "brown", "davis", "johnson", "smith", "lee", "wrong author", "different author"]
            year_keywords = ["year", "2018", "2021", "wrong year", "different year", "date"]

            has_author = any(kw in reason for kw in author_keywords)
            has_year = any(kw in reason for kw in year_keywords)

            if has_author and has_year:
                return 1.0
            elif has_author or has_year:
                return 0.7
            else:
                return 0.5
        else:
            return 0.3

    return 0.0


def grade_task_3(action_dict: Dict[str, Any]) -> float:
    """
    Grade task 3 — The Contradiction (hard).

    Full credit (1.0): correct citation_id + reason mentions contradiction
    High (0.7): correct citation_id + reason mentions the drug/effect
    Partial (0.5): correct citation_id, generic reason
    Low (0.3): wrong citation_id flagged
    """
    gt = SCENARIOS["task_3"]["ground_truth"]
    action_type = str(action_dict.get("action_type", "")).strip().lower()
    citation_id = action_dict.get("citation_id", -1)
    reason = str(action_dict.get("reason", "")).strip().lower()

    if action_type == "approve":
        return 0.0  # -0.5 reward for task_3 handled in environment

    if action_type == "flag_hallucination":
        if citation_id == gt["hallucinated_citation_id"]:
            contradiction_keywords = [
                "contradict", "opposite", "conflict", "inconsistent",
                "no significant", "no effect", "does not support",
                "misrepresent", "contrary", "disagree",
            ]
            drug_keywords = ["drug x", "disease y", "p=0.43", "placebo", "no significant effect"]

            has_contradiction = any(kw in reason for kw in contradiction_keywords)
            has_drug = any(kw in reason for kw in drug_keywords)

            if has_contradiction:
                return 1.0
            elif has_drug:
                return 0.7
            else:
                return 0.5
        else:
            return 0.3

    return 0.0


# ---------------------------------------------------------------------------
# Grader registry
# ---------------------------------------------------------------------------

GRADERS = {
    "task_1": grade_task_1,
    "task_2": grade_task_2,
    "task_3": grade_task_3,
}
