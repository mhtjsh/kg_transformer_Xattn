"""Aggregate per-seed prediction files and compute the V0 decision summary."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kg_xattn.metrics import paired_bootstrap_ci, score_prediction, summarize_scores


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", nargs=3, required=True, help="Three SFT prediction JSONL files")
    parser.add_argument("--kg", nargs=3, required=True, help="Three KG prediction JSONL files")
    parser.add_argument("--output", default="runs/v0/final_report.json")
    args = parser.parse_args()

    sft_seed_scores = []
    kg_seed_scores = []
    per_question = defaultdict(lambda: {"sft": [], "kg": []})

    for system_name, files, target in [
        ("sft", args.sft, sft_seed_scores),
        ("kg", args.kg, kg_seed_scores),
    ]:
        for path in files:
            rows = load_jsonl(Path(path))
            scores = [
                score_prediction(row["id"], row["raw_prediction"], row["gold"])
                for row in rows
            ]
            target.append(summarize_scores(scores))
            for score in scores:
                per_question[score.question_id][system_name].append(float(score.correct))

    sft_mean = sum(row["set_em"] for row in sft_seed_scores) / len(sft_seed_scores)
    kg_mean = sum(row["set_em"] for row in kg_seed_scores) / len(kg_seed_scores)
    sft_correct = []
    kg_correct = []
    for question_id in sorted(per_question):
        values = per_question[question_id]
        if len(values["sft"]) == 3 and len(values["kg"]) == 3:
            sft_correct.append(sum(values["sft"]) / 3.0)
            kg_correct.append(sum(values["kg"]) / 3.0)

    observed, low, high = paired_bootstrap_ci(kg_correct, sft_correct)
    kg_wins = sum(
        int(kg_seed_scores[index]["set_em"] > sft_seed_scores[index]["set_em"])
        for index in range(3)
    )

    report = {
        "sft_seed_metrics": sft_seed_scores,
        "kg_seed_metrics": kg_seed_scores,
        "sft_mean_set_em": sft_mean,
        "kg_mean_set_em": kg_mean,
        "absolute_difference": kg_mean - sft_mean,
        "paired_bootstrap_difference": observed,
        "paired_bootstrap_95_ci": [low, high],
        "kg_wins_out_of_three": kg_wins,
        "positive_core_criteria": {
            "mean_gain_at_least_2_points": kg_mean - sft_mean >= 0.02,
            "ci_lower_bound_above_zero": low > 0.0,
            "wins_at_least_two_seeds": kg_wins >= 2,
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
