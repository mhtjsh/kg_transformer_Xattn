"""Prepare RiTeK/PharmKG files for the V0 experiment.

This script downloads or reads the Hugging Face dataset snapshot, extracts KG
triples into a simple JSONL schema, copies QA files, and creates
question_id -> topic entity mappings. Topic mapping prefers explicit dataset
fields; if those are absent, it falls back to exact entity-name mention matching
in the question text. Gold answers are never used as retrieval topics.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kg_xattn.data import coerce_qa_example


def slug(text: str) -> str:
    """Stable ID for datasets where KG rows provide names but no IDs."""

    text = re.sub(r"\s+", " ", text.strip().casefold())
    text = re.sub(r"[^a-z0-9_.:-]+", "_", text)
    return text.strip("_") or "empty"


def split_triple_line(line: str) -> tuple[str, str, str] | None:
    """Parse common triple formats: tab, comma, or pipe separated."""

    line = line.strip()
    if not line or line.startswith("#"):
        return None
    for sep in ["\t", "|", ","]:
        parts = [part.strip() for part in line.split(sep)]
        if len(parts) >= 3 and all(parts[:3]):
            return parts[0], parts[1], parts[2]
    return None


def find_first(root: Path, patterns: list[str]) -> Path:
    """Find the first file matching any glob pattern under root."""

    for pattern in patterns:
        matches = sorted(root.rglob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find any of: {patterns} under {root}")


def load_json_records(path: Path) -> list[dict[str, Any]]:
    """Load JSON or JSONL records for topic extraction."""

    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        if "data" in payload:
            payload = payload["data"]
        elif "examples" in payload:
            payload = payload["examples"]
        else:
            payload = list(payload.values())
    return list(payload)


def explicit_topics(row: dict[str, Any]) -> list[str]:
    """Read topic entity IDs/names when the QA record already provides them."""

    for key in ["topic_entities", "topic_entity", "entities", "question_entities"]:
        if key in row and row[key]:
            value = row[key]
            if isinstance(value, str):
                return [value]
            if isinstance(value, dict):
                return [str(item) for item in value.values()]
            return [str(item) for item in value]
    return []


def mention_topics(question: str, entity_name_to_id: dict[str, str], max_topics: int = 4) -> list[str]:
    """Fallback exact mention linker from question text to KG entity IDs."""

    lowered = question.casefold()
    hits = []
    for name, entity_id in entity_name_to_id.items():
        if len(name) < 4:
            continue
        if name in lowered:
            hits.append((len(name), entity_id))
    hits.sort(reverse=True)
    return [entity_id for _, entity_id in hits[:max_topics]]


def prepare_edges(kg_path: Path, output_path: Path) -> dict[str, str]:
    """Convert Pharm_KG.txt-style triples into processed KG edge JSONL."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    entity_name_to_id: dict[str, str] = {}
    n_edges = 0

    with kg_path.open("r", encoding="utf-8", errors="replace") as src:
        with output_path.open("w", encoding="utf-8") as dst:
            for line in src:
                triple = split_triple_line(line)
                if triple is None:
                    continue
                head, relation, tail = triple
                head_id = slug(head)
                tail_id = slug(tail)
                relation_id = slug(relation)
                entity_name_to_id[head.casefold()] = head_id
                entity_name_to_id[tail.casefold()] = tail_id
                row = {
                    "source_id": head_id,
                    "source_name": head,
                    "relation_id": relation_id,
                    "relation_name": relation,
                    "target_id": tail_id,
                    "target_name": tail,
                }
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_edges += 1

    if n_edges == 0:
        raise RuntimeError(f"No KG triples parsed from {kg_path}")
    return entity_name_to_id


def prepare_qa_and_topics(
    source_files: dict[str, Path],
    output_dir: Path,
    entity_name_to_id: dict[str, str],
) -> None:
    """Copy QA files and build a topic mapping for all splits."""

    output_dir.mkdir(parents=True, exist_ok=True)
    topic_rows = []

    for split, path in source_files.items():
        target_path = output_dir / f"{split}.json"
        shutil.copyfile(path, target_path)

        for index, row in enumerate(load_json_records(path)):
            example = coerce_qa_example(row, index)
            topics = explicit_topics(row)
            if not topics:
                topics = mention_topics(example.question, entity_name_to_id)
            topics = [slug(topic) if topic.casefold() in entity_name_to_id else topic for topic in topics]
            topic_rows.append(
                {
                    "id": example.question_id,
                    "topic_entities": topics,
                    "source_split": split,
                }
            )

    topic_path = output_dir / "processed" / "question_topics.jsonl"
    topic_path.parent.mkdir(parents=True, exist_ok=True)
    with topic_path.open("w", encoding="utf-8") as handle:
        for row in topic_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--output-dir", default="data/ritek")
    parser.add_argument(
        "--repo-id",
        default="ChenAI2015/A-Dataset-for-Complex-Reasoning-over-Textual-Knowledge-Graphs-in-Medicine",
    )
    args = parser.parse_args()

    if args.source_dir:
        source_root = Path(args.source_dir)
    else:
        from huggingface_hub import snapshot_download

        source_root = Path(snapshot_download(repo_id=args.repo_id, repo_type="dataset"))

    output_dir = Path(args.output_dir)
    kg_path = find_first(source_root, ["Pharm_KG.txt", "*KG*.txt", "*kg*.txt"])
    train_path = find_first(source_root, ["*train*.json", "*Train*.json", "*train*.jsonl"])
    dev_path = find_first(source_root, ["*dev*.json", "*valid*.json", "*val*.json", "*dev*.jsonl"])
    test_path = find_first(source_root, ["*test*.json", "*Test*.json", "*test*.jsonl"])

    entity_name_to_id = prepare_edges(
        kg_path=kg_path,
        output_path=output_dir / "processed" / "kg_edges.jsonl",
    )
    prepare_qa_and_topics(
        source_files={"train": train_path, "dev": dev_path, "test": test_path},
        output_dir=output_dir,
        entity_name_to_id=entity_name_to_id,
    )

    print(f"Prepared RiTeK files under {output_dir}")
    print(f"KG source: {kg_path}")


if __name__ == "__main__":
    main()
