"""Evaluation utilities for answer-set exact match and reporting.

The primary metric is strict normalized answer-set exact match. This file also
contains the parser that rejects explanations, malformed multi-answer strings,
and other outputs outside the locked V0 answer format.
"""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable


def normalize_answer_component(text: str) -> str:
    """Apply only the normalization allowed by the protocol."""

    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()

    if text.endswith("."):
        text = text[:-1].strip()

    return text


def normalize_answer_set(answers: Iterable[str]) -> set[str]:
    """Normalize a collection of gold or predicted answer strings."""

    return {
        normalized
        for answer in answers
        if (normalized := normalize_answer_component(str(answer)))
    }


@dataclass
class ParsedPrediction:
    """Result of strict V0 prediction parsing."""

    valid: bool
    answers: set[str]
    reason: str = ""


def parse_prediction(text: str, max_components: int = 3) -> ParsedPrediction:
    """Parse the generated answer string according to the locked protocol."""

    stripped = text.strip()
    if not stripped:
        return ParsedPrediction(False, set(), "empty output")
    if "```" in stripped:
        return ParsedPrediction(False, set(), "code block")

    non_empty_lines = [line for line in stripped.splitlines() if line.strip()]
    if len(non_empty_lines) != 1:
        return ParsedPrediction(False, set(), "not exactly one non-empty line")

    parts = [part.strip() for part in non_empty_lines[0].split("||")]
    if any(not part for part in parts):
        return ParsedPrediction(False, set(), "empty answer component")
    if len(parts) > max_components:
        return ParsedPrediction(False, set(), "too many answer components")

    answers = normalize_answer_set(parts)
    if not answers:
        return ParsedPrediction(False, set(), "normalizes to empty answer set")
    return ParsedPrediction(True, answers)


@dataclass
class ExampleScore:
    """Per-example strict answer-set score."""

    question_id: str
    correct: bool
    valid: bool
    precision_num: int
    precision_den: int
    recall_num: int
    recall_den: int
    evidence_present: bool | None = None


def score_prediction(
    question_id: str,
    prediction_text: str,
    gold_answers: Iterable[str],
    evidence_present: bool | None = None,
) -> ExampleScore:
    """Score one prediction with strict set exact match plus micro counts."""

    parsed = parse_prediction(prediction_text)
    gold_set = normalize_answer_set(gold_answers)
    pred_set = parsed.answers if parsed.valid else set()
    overlap = pred_set & gold_set

    return ExampleScore(
        question_id=question_id,
        correct=parsed.valid and pred_set == gold_set,
        valid=parsed.valid,
        precision_num=len(overlap),
        precision_den=len(pred_set),
        recall_num=len(overlap),
        recall_den=len(gold_set),
        evidence_present=evidence_present,
    )


def summarize_scores(scores: list[ExampleScore]) -> dict[str, float]:
    """Aggregate SetEM, valid rate, and micro precision/recall/F1."""

    if not scores:
        return {
            "set_em": 0.0,
            "valid_format_rate": 0.0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
        }

    correct = sum(int(score.correct) for score in scores)
    valid = sum(int(score.valid) for score in scores)
    p_num = sum(score.precision_num for score in scores)
    p_den = sum(score.precision_den for score in scores)
    r_num = sum(score.recall_num for score in scores)
    r_den = sum(score.recall_den for score in scores)

    precision = p_num / p_den if p_den else 0.0
    recall = r_num / r_den if r_den else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "set_em": correct / len(scores),
        "valid_format_rate": valid / len(scores),
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
    }


def paired_bootstrap_ci(
    kg_correct: list[float],
    sft_correct: list[float],
    n_bootstrap: int = 10000,
    seed: int = 2026,
) -> tuple[float, float, float]:
    """Paired bootstrap CI for mean(KG - SFT) over shared test questions."""

    if len(kg_correct) != len(sft_correct):
        raise ValueError("kg_correct and sft_correct must have the same length")
    if not kg_correct:
        raise ValueError("cannot bootstrap an empty score vector")

    rng = random.Random(seed)
    deltas = [kg - sft for kg, sft in zip(kg_correct, sft_correct)]
    observed = sum(deltas) / len(deltas)

    boot = []
    n = len(deltas)
    for _ in range(n_bootstrap):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        boot.append(sum(sample) / n)

    boot.sort()
    low = boot[int(0.025 * (n_bootstrap - 1))]
    high = boot[int(0.975 * (n_bootstrap - 1))]
    return observed, low, high
