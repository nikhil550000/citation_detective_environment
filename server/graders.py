"""
Graders and mock database for the Citation Detective environment.

Contains:
  - MOCK_DATABASE: hardcoded citation database for deterministic grading
  - SCENARIOS: 5 tasks (easy → hard) with manuscript excerpts
  - search_database(): fuzzy title/author/abstract search
  - grade_task_1/2/3/4/5(): composite grader functions
  - GRADERS: dict mapping task_id -> grader function

Reward Design Philosophy:
  Each grader returns a COMPOSITE score in the open interval (0, 1)
  computed from three orthogonal learning dimensions:

    score = BASE + IDENTIFICATION + REASON_QUALITY

  - BASE (0.05):       Minimum for attempting any action
  - IDENTIFICATION:     Did the agent find the right problematic citation?
                        Correct citation: +0.45, Wrong citation: +0.15, Approve: +0.0
  - REASON_QUALITY:     Does the explanation demonstrate real understanding?
                        Excellent (multiple key insights): +0.40
                        Good (one key insight):            +0.25
                        Partial (vague but relevant):      +0.15
                        None (no useful reasoning):        +0.05

  Natural range: [0.05, 0.90] — never reaches 0.0 or 1.0
  This provides a smooth, learnable gradient for RL training.
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
    "chen_2020": {
        "title": "Effect of Online Learning on STEM Student Performance",
        "authors": ["Chen, L.", "Park, J."],
        "year": 2020,
        "abstract": (
            "Our meta-analysis of 47 studies found that online learning formats "
            "produced a 12% improvement in standardized test scores for STEM "
            "subjects compared to traditional in-person instruction. Effect sizes "
            "varied significantly across disciplines and grade levels."
        ),
    },
    "okafor_2022": {
        "title": "Social Media Use Correlates with Increased Anxiety in Teenagers",
        "authors": ["Okafor, A.", "Singh, P."],
        "year": 2022,
        "abstract": (
            "Cross-sectional study of 2,000 teenagers shows a moderate correlation "
            "between social media use exceeding 3 hours per day and self-reported "
            "anxiety symptoms (r=0.42, p<0.001). Critically, our cross-sectional "
            "design cannot establish causality — teenagers with pre-existing anxiety "
            "may independently seek more social media engagement."
        ),
    },
    "martinez_2021": {
        "title": "Antioxidant Supplementation and Cardiovascular Outcomes: A Randomized Trial",
        "authors": ["Martinez, G.", "Thompson, H.", "Lewis, M."],
        "year": 2021,
        "abstract": (
            "This randomized controlled trial of 3,200 participants over 5 years found "
            "that high-dose antioxidant supplementation (vitamins C, E, and beta-carotene) "
            "showed a modest 8% reduction in oxidative stress biomarkers (p=0.03) but "
            "NO significant reduction in major cardiovascular events (HR=0.96, 95% CI "
            "0.84-1.10, p=0.55). Importantly, the beta-carotene arm showed a "
            "non-significant trend toward INCREASED all-cause mortality (HR=1.12, "
            "p=0.08). The study concludes that antioxidant supplementation cannot be "
            "recommended for cardiovascular disease prevention."
        ),
    },
    "nakamura_2023": {
        "title": "Quantum Error Correction in Topological Qubits: Progress and Limitations",
        "authors": ["Nakamura, Y.", "Petrov, D."],
        "year": 2023,
        "abstract": (
            "We report progress on topological qubit architectures achieving a "
            "logical error rate of 10^-4 per gate cycle. While this represents a "
            "100x improvement over previous surface code implementations, we show "
            "that fault-tolerant quantum computation still requires approximately "
            "1,000 physical qubits per logical qubit, limiting near-term practical "
            "applications. Current systems remain far from the threshold needed for "
            "cryptographically relevant quantum computation."
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
    "task_4": {
        "difficulty": "medium",
        "description": "The Misquoted Statistic — real paper, real authors, but manuscript fabricates the numerical result.",
        "manuscript_excerpt": (
            "The shift to online education has been transformative. Chen and Park's "
            "landmark meta-analysis demonstrated that online learning achieves "
            "a 67% improvement in STEM student performance compared to traditional "
            "in-person instruction [1], establishing online delivery as unequivocally "
            "superior across all educational contexts.\n\n"
            "These extraordinary gains have driven widespread adoption of digital "
            "platforms in universities and K-12 schools alike. Educators cite the "
            "Chen and Park study [1] as definitive evidence that in-person teaching "
            "is obsolete for STEM subjects."
        ),
        "citations_list": [
            {
                "id": 1,
                "title": "Effect of Online Learning on STEM Student Performance",
                "authors": ["Chen, L.", "Park, J."],
                "year": 2020,
            },
        ],
        "ground_truth": {
            "hallucinated_citation_id": 1,
            "issue_type": "misquoted_statistic",
            "explanation": (
                "The paper by Chen and Park reports a 12% improvement, not 67%. "
                "The manuscript fabricates the statistic and overstates the paper's "
                "conclusion — the actual paper notes that effect sizes vary significantly "
                "and does not claim superiority across all educational contexts."
            ),
        },
    },
    "task_5": {
        "difficulty": "hard",
        "description": "The Causality Reversal — paper shows correlation only, manuscript claims proven causation.",
        "manuscript_excerpt": (
            "The mental health crisis among teenagers is now conclusively linked "
            "to social media. Okafor and Singh's rigorous study [1] proved that "
            "social media use directly causes anxiety in adolescents, establishing "
            "a clear causal mechanism supported by their data from 2,000 participants. "
            "Their findings make it scientifically certain that reducing social media "
            "use will reduce teenage anxiety.\n\n"
            "Policy makers and educators should treat this causal relationship as "
            "established fact when designing interventions. The Okafor-Singh study "
            "provides the definitive scientific basis for social media regulation."
        ),
        "citations_list": [
            {
                "id": 1,
                "title": "Social Media Use Correlates with Increased Anxiety in Teenagers",
                "authors": ["Okafor, A.", "Singh, P."],
                "year": 2022,
            },
        ],
        "ground_truth": {
            "hallucinated_citation_id": 1,
            "issue_type": "causality_reversal",
            "explanation": (
                "The Okafor & Singh paper explicitly states its cross-sectional design "
                "'cannot establish causality'. The manuscript falsely claims the study "
                "'proved' causation and calls it 'scientifically certain'. "
                "This is a direct misrepresentation of what the paper concluded."
            ),
        },
    },
    "task_6": {
        "difficulty": "hard",
        "description": "The Selective Omission — manuscript cherry-picks one minor positive result while ignoring the paper's main negative conclusion.",
        "manuscript_excerpt": (
            "Antioxidant supplementation has emerged as a promising approach for "
            "cardiovascular disease prevention. The landmark randomized trial by "
            "Martinez, Thompson, and Lewis demonstrated significant benefits: their "
            "study of 3,200 participants showed that antioxidant supplementation "
            "significantly reduced oxidative stress biomarkers [1], providing strong "
            "evidence for the cardioprotective effects of vitamins C, E, and "
            "beta-carotene.\n\n"
            "Based on these findings, the American Heart Health Foundation recommends "
            "daily antioxidant supplementation as a primary prevention strategy. "
            "The Martinez et al. [1] study represents the strongest clinical evidence "
            "to date supporting this recommendation."
        ),
        "citations_list": [
            {
                "id": 1,
                "title": "Antioxidant Supplementation and Cardiovascular Outcomes: A Randomized Trial",
                "authors": ["Martinez, G.", "Thompson, H.", "Lewis, M."],
                "year": 2021,
            },
        ],
        "ground_truth": {
            "hallucinated_citation_id": 1,
            "issue_type": "selective_omission",
            "explanation": (
                "The manuscript cherry-picks the minor finding (8% reduction in "
                "oxidative stress biomarkers) while completely omitting the study's "
                "MAIN conclusion: NO significant reduction in cardiovascular events "
                "(p=0.55) and a concerning trend toward increased mortality in the "
                "beta-carotene arm (HR=1.12). The paper explicitly concludes that "
                "antioxidant supplementation CANNOT be recommended for cardiovascular "
                "disease prevention — the opposite of what the manuscript claims."
            ),
        },
    },
    "task_7": {
        "difficulty": "expert",
        "description": "The Temporal Fabrication — manuscript cites a paper with a future publication year that doesn't exist in the database.",
        "manuscript_excerpt": (
            "Recent breakthroughs in quantum computing have brought us closer than "
            "ever to practical quantum advantage. Nakamura and Petrov's earlier work "
            "on topological qubits demonstrated significant progress in error "
            "correction [1]. More recently, their follow-up study published in 2025 "
            "achieved the long-sought threshold for fault-tolerant quantum computation, "
            "demonstrating that just 50 physical qubits per logical qubit are sufficient "
            "for cryptographically relevant computations [2].\n\n"
            "This breakthrough, reported in the 2025 Nakamura-Petrov paper [2], "
            "effectively renders current encryption standards obsolete and necessitates "
            "immediate transition to post-quantum cryptographic protocols."
        ),
        "citations_list": [
            {
                "id": 1,
                "title": "Quantum Error Correction in Topological Qubits: Progress and Limitations",
                "authors": ["Nakamura, Y.", "Petrov, D."],
                "year": 2023,
            },
            {
                "id": 2,
                "title": "Fault-Tolerant Quantum Computation with Minimal Qubit Overhead",
                "authors": ["Nakamura, Y.", "Petrov, D."],
                "year": 2025,
            },
        ],
        "ground_truth": {
            "hallucinated_citation_id": 2,
            "issue_type": "temporal_fabrication",
            "explanation": (
                "Citation [2] claims to be a 2025 publication by Nakamura and Petrov, "
                "but no such paper exists in the database. The real 2023 paper [1] "
                "by the same authors explicitly states that 1,000 physical qubits per "
                "logical qubit are needed and that practical quantum computation remains "
                "far from achievable — the exact opposite of what the fabricated 2025 "
                "reference claims. This is both a ghost paper AND a contradiction of "
                "the actual research."
            ),
        },
    },
}


# ---------------------------------------------------------------------------
# Database search
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
        title_match = query_lower in entry["title"].lower()
        author_match = any(query_lower in a.lower() for a in entry["authors"])
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
# Composite grader functions
#
# Each returns a float strictly in (0, 1):
#   score = BASE(0.05) + IDENTIFICATION(0..0.45) + REASON_QUALITY(0..0.40)
#
# The three-component design ensures:
#   - An agent always gets SOME signal (base > 0)
#   - A perfect answer never reaches 1.0 (max = 0.90)
#   - The gradient is smooth across many distinct quality levels
# ---------------------------------------------------------------------------

BASE_SCORE = 0.05          # minimum for any attempt
ID_CORRECT = 0.45          # bonus for correct citation ID
ID_WRONG_FLAG = 0.15       # partial credit for flagging (wrong citation)
ID_APPROVE = 0.0           # no credit for approving a hallucination

REASON_EXCELLENT = 0.40    # multiple key insights demonstrated
REASON_GOOD = 0.25         # one key insight
REASON_PARTIAL = 0.15      # vague but relevant
REASON_MINIMAL = 0.05      # no useful reasoning but action was taken


def _compute_score(identification: float, reason_quality: float) -> float:
    """Composite score from identification accuracy + reason quality."""
    raw = BASE_SCORE + identification + reason_quality
    # Safety clamp — should never trigger with our constants, but guarantees (0, 1)
    return min(max(raw, 0.01), 0.99)


def grade_task_1(action_dict: Dict[str, Any]) -> float:
    """
    Grade task 1 — The Ghost Paper (easy).

    The agent should flag citation_id=2 as a fabricated reference
    because it doesn't exist in the database.
    """
    gt = SCENARIOS["task_1"]["ground_truth"]
    action_type = str(action_dict.get("action_type", "")).strip().lower()
    citation_id = action_dict.get("citation_id", -1)
    reason = str(action_dict.get("reason", "")).strip().lower()

    # Identification dimension
    if action_type == "flag_hallucination" and citation_id == gt["hallucinated_citation_id"]:
        identification = ID_CORRECT
    elif action_type == "flag_hallucination":
        identification = ID_WRONG_FLAG
    else:
        identification = ID_APPROVE

    # Reason dimension (only meaningful for correct identification)
    ghost_keywords = ["not found", "not exist", "no result", "fabricat", "ghost",
                      "fake", "doesn't exist", "does not exist", "no match",
                      "no database", "no entry", "couldn't find", "not in database"]

    if identification == ID_CORRECT:
        hits = sum(1 for kw in ghost_keywords if kw in reason)
        if hits >= 2:
            reason_q = REASON_EXCELLENT
        elif hits == 1:
            reason_q = REASON_GOOD
        elif len(reason) > 20:
            reason_q = REASON_PARTIAL
        else:
            reason_q = REASON_MINIMAL
    elif identification == ID_WRONG_FLAG:
        reason_q = REASON_MINIMAL
    else:
        reason_q = 0.0

    return _compute_score(identification, reason_q)


def grade_task_2(action_dict: Dict[str, Any]) -> float:
    """
    Grade task 2 — The Identity Theft (medium).

    The agent should flag citation_id=1 because the authors and year
    don't match the database entry.
    """
    gt = SCENARIOS["task_2"]["ground_truth"]
    action_type = str(action_dict.get("action_type", "")).strip().lower()
    citation_id = action_dict.get("citation_id", -1)
    reason = str(action_dict.get("reason", "")).strip().lower()

    if action_type == "flag_hallucination" and citation_id == gt["hallucinated_citation_id"]:
        identification = ID_CORRECT
    elif action_type == "flag_hallucination":
        identification = ID_WRONG_FLAG
    else:
        identification = ID_APPROVE

    author_keywords = ["author", "brown", "davis", "johnson", "smith", "lee",
                       "wrong author", "different author", "misattribut",
                       "not the real author", "attributed"]
    year_keywords = ["year", "2018", "2021", "wrong year", "different year",
                     "date", "published"]

    if identification == ID_CORRECT:
        has_author = any(kw in reason for kw in author_keywords)
        has_year = any(kw in reason for kw in year_keywords)
        if has_author and has_year:
            reason_q = REASON_EXCELLENT
        elif has_author or has_year:
            reason_q = REASON_GOOD
        elif len(reason) > 20:
            reason_q = REASON_PARTIAL
        else:
            reason_q = REASON_MINIMAL
    elif identification == ID_WRONG_FLAG:
        reason_q = REASON_MINIMAL
    else:
        reason_q = 0.0

    return _compute_score(identification, reason_q)


def grade_task_3(action_dict: Dict[str, Any]) -> float:
    """
    Grade task 3 — The Contradiction (hard).

    The agent should flag citation_id=1 because the manuscript claims
    Drug X works, but the paper says it has no significant effect.
    """
    gt = SCENARIOS["task_3"]["ground_truth"]
    action_type = str(action_dict.get("action_type", "")).strip().lower()
    citation_id = action_dict.get("citation_id", -1)
    reason = str(action_dict.get("reason", "")).strip().lower()

    if action_type == "flag_hallucination" and citation_id == gt["hallucinated_citation_id"]:
        identification = ID_CORRECT
    elif action_type == "flag_hallucination":
        identification = ID_WRONG_FLAG
    else:
        identification = ID_APPROVE

    contradiction_keywords = ["contradict", "opposite", "conflict", "inconsistent",
                              "no significant", "no effect", "does not support",
                              "misrepresent", "contrary", "disagree", "mismatch"]
    drug_keywords = ["drug x", "disease y", "p=0.43", "placebo",
                     "no significant effect", "no improvement"]

    if identification == ID_CORRECT:
        has_contradiction = any(kw in reason for kw in contradiction_keywords)
        has_drug = any(kw in reason for kw in drug_keywords)
        if has_contradiction and has_drug:
            reason_q = REASON_EXCELLENT
        elif has_contradiction:
            reason_q = REASON_GOOD
        elif has_drug:
            reason_q = REASON_PARTIAL + 0.05  # 0.20 — knows the topic
        elif len(reason) > 20:
            reason_q = REASON_PARTIAL
        else:
            reason_q = REASON_MINIMAL
    elif identification == ID_WRONG_FLAG:
        reason_q = REASON_MINIMAL
    else:
        reason_q = 0.0

    return _compute_score(identification, reason_q)


def grade_task_4(action_dict: Dict[str, Any]) -> float:
    """
    Grade task 4 — The Misquoted Statistic (medium).

    The agent should flag citation_id=1 because the manuscript claims
    67% improvement when the paper says 12%.
    """
    gt = SCENARIOS["task_4"]["ground_truth"]
    action_type = str(action_dict.get("action_type", "")).strip().lower()
    citation_id = action_dict.get("citation_id", -1)
    reason = str(action_dict.get("reason", "")).strip().lower()

    if action_type == "flag_hallucination" and citation_id == gt["hallucinated_citation_id"]:
        identification = ID_CORRECT
    elif action_type == "flag_hallucination":
        identification = ID_WRONG_FLAG
    else:
        identification = ID_APPROVE

    number_keywords = ["12", "67", "12%", "67%"]
    fabrication_keywords = ["fabricat", "misquot", "incorrect", "wrong number",
                            "inflat", "exaggerat", "overstat", "inaccurate",
                            "false", "distort", "different number", "actual"]

    if identification == ID_CORRECT:
        has_number = any(kw in reason for kw in number_keywords)
        has_fabrication = any(kw in reason for kw in fabrication_keywords)
        if has_number and has_fabrication:
            reason_q = REASON_EXCELLENT
        elif has_number:
            reason_q = REASON_GOOD  # knows the actual numbers
        elif has_fabrication:
            reason_q = REASON_PARTIAL + 0.05  # 0.20 — knows it's fabricated
        elif len(reason) > 20:
            reason_q = REASON_PARTIAL
        else:
            reason_q = REASON_MINIMAL
    elif identification == ID_WRONG_FLAG:
        reason_q = REASON_MINIMAL
    else:
        reason_q = 0.0

    return _compute_score(identification, reason_q)


def grade_task_5(action_dict: Dict[str, Any]) -> float:
    """
    Grade task 5 — The Causality Reversal (hard).

    The agent should flag citation_id=1 because the paper only shows
    correlation but the manuscript claims proven causation.
    """
    gt = SCENARIOS["task_5"]["ground_truth"]
    action_type = str(action_dict.get("action_type", "")).strip().lower()
    citation_id = action_dict.get("citation_id", -1)
    reason = str(action_dict.get("reason", "")).strip().lower()

    if action_type == "flag_hallucination" and citation_id == gt["hallucinated_citation_id"]:
        identification = ID_CORRECT
    elif action_type == "flag_hallucination":
        identification = ID_WRONG_FLAG
    else:
        identification = ID_APPROVE

    causality_keywords = ["causal", "cause", "causation", "correlation", "correlat"]
    cannot_keywords = ["cannot", "can't", "does not establish", "no causal",
                       "cross-sectional", "not proven", "not causal",
                       "only correlation", "doesn't prove", "does not prove"]

    if identification == ID_CORRECT:
        has_causality = any(kw in reason for kw in causality_keywords)
        has_cannot = any(kw in reason for kw in cannot_keywords)
        if has_causality and has_cannot:
            reason_q = REASON_EXCELLENT
        elif has_cannot:
            reason_q = REASON_GOOD
        elif has_causality:
            reason_q = REASON_PARTIAL  # mentions causality but not the limitation
        elif len(reason) > 20:
            reason_q = REASON_PARTIAL
        else:
            reason_q = REASON_MINIMAL
    elif identification == ID_WRONG_FLAG:
        reason_q = REASON_MINIMAL
    else:
        reason_q = 0.0

    return _compute_score(identification, reason_q)


# ---------------------------------------------------------------------------
# Grader registry
# ---------------------------------------------------------------------------

def grade_task_6(action_dict: Dict[str, Any]) -> float:
    """
    Grade task 6 — The Selective Omission (hard).

    The agent should flag citation_id=1 because the manuscript cherry-picks
    one minor positive result (8% biomarker reduction) while omitting the
    main conclusion (no cardiovascular benefit, possible increased mortality).
    """
    gt = SCENARIOS["task_6"]["ground_truth"]
    action_type = str(action_dict.get("action_type", "")).strip().lower()
    citation_id = action_dict.get("citation_id", -1)
    reason = str(action_dict.get("reason", "")).strip().lower()

    if action_type == "flag_hallucination" and citation_id == gt["hallucinated_citation_id"]:
        identification = ID_CORRECT
    elif action_type == "flag_hallucination":
        identification = ID_WRONG_FLAG
    else:
        identification = ID_APPROVE

    omission_keywords = ["omit", "cherry-pick", "selective", "ignore", "left out",
                         "hide", "conceal", "incomplete", "misleading", "partial"]
    result_keywords = ["no significant", "cardiovascular", "p=0.55", "mortality",
                       "beta-carotene", "cannot be recommended", "8%",
                       "not recommended", "negative"]

    if identification == ID_CORRECT:
        has_omission = any(kw in reason for kw in omission_keywords)
        has_result = any(kw in reason for kw in result_keywords)
        if has_omission and has_result:
            reason_q = REASON_EXCELLENT
        elif has_omission:
            reason_q = REASON_GOOD
        elif has_result:
            reason_q = REASON_PARTIAL + 0.05  # 0.20 — knows specific details
        elif len(reason) > 20:
            reason_q = REASON_PARTIAL
        else:
            reason_q = REASON_MINIMAL
    elif identification == ID_WRONG_FLAG:
        reason_q = REASON_MINIMAL
    else:
        reason_q = 0.0

    return _compute_score(identification, reason_q)


def grade_task_7(action_dict: Dict[str, Any]) -> float:
    """
    Grade task 7 — The Temporal Fabrication (expert).

    The agent should flag citation_id=2 because it claims to be a 2025
    paper that doesn't exist in the database, while also contradicting
    the real 2023 paper's findings.
    """
    gt = SCENARIOS["task_7"]["ground_truth"]
    action_type = str(action_dict.get("action_type", "")).strip().lower()
    citation_id = action_dict.get("citation_id", -1)
    reason = str(action_dict.get("reason", "")).strip().lower()

    if action_type == "flag_hallucination" and citation_id == gt["hallucinated_citation_id"]:
        identification = ID_CORRECT
    elif action_type == "flag_hallucination":
        identification = ID_WRONG_FLAG
    else:
        identification = ID_APPROVE

    temporal_keywords = ["2025", "future", "doesn't exist", "does not exist",
                         "not found", "fabricat", "ghost", "not in database",
                         "no record", "no such paper"]
    contradiction_keywords = ["contradict", "1000", "1,000", "50", "opposite",
                              "limitations", "far from", "not achievable"]

    if identification == ID_CORRECT:
        has_temporal = any(kw in reason for kw in temporal_keywords)
        has_contradiction = any(kw in reason for kw in contradiction_keywords)
        if has_temporal and has_contradiction:
            reason_q = REASON_EXCELLENT
        elif has_temporal:
            reason_q = REASON_GOOD
        elif has_contradiction:
            reason_q = REASON_PARTIAL + 0.05  # 0.20
        elif len(reason) > 20:
            reason_q = REASON_PARTIAL
        else:
            reason_q = REASON_MINIMAL
    elif identification == ID_WRONG_FLAG:
        reason_q = REASON_MINIMAL
    else:
        reason_q = 0.0

    return _compute_score(identification, reason_q)


# ---------------------------------------------------------------------------
# Grader registry
# ---------------------------------------------------------------------------

GRADERS = {
    "task_1": grade_task_1,
    "task_2": grade_task_2,
    "task_3": grade_task_3,
    "task_4": grade_task_4,
    "task_5": grade_task_5,
    "task_6": grade_task_6,
    "task_7": grade_task_7,
}
