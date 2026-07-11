"""Precompute frozen BioLORD entity and relation embeddings for KG memory.

The graph adapter never trains BioLORD. This script runs the sentence encoder
once, L2-normalizes all vectors, and saves torch dictionaries consumed by
FrozenEdgeEmbeddingEncoder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_edges(path: Path) -> list[dict]:
    """Load processed KG edge JSONL."""

    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def batched(items: list[str], batch_size: int):
    """Yield fixed-size batches from a list."""

    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def encode_texts(model, texts: list[str], batch_size: int) -> torch.Tensor:
    """Encode text strings and return L2-normalized float32 tensors."""

    chunks = []
    for batch in batched(texts, batch_size):
        vectors = model.encode(
            batch,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        chunks.append(vectors.detach().cpu().float())
    return F.normalize(torch.cat(chunks, dim=0), p=2, dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg-edges", default="data/ritek/processed/kg_edges.jsonl")
    parser.add_argument("--output-dir", default="data/ritek/cache")
    parser.add_argument("--model-name", default="FremyCompany/BioLORD-2023")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    edges = load_edges(Path(args.kg_edges))
    entity_text_by_id: dict[str, str] = {}
    relation_text_by_id: dict[str, str] = {}

    for edge in edges:
        entity_text_by_id[edge["source_id"]] = (
            f"{edge['source_name']}. Definition: {edge.get('source_description', '')}"
        )
        entity_text_by_id[edge["target_id"]] = (
            f"{edge['target_name']}. Definition: {edge.get('target_description', '')}"
        )
        relation_text_by_id[edge["relation_id"]] = edge["relation_name"]

    model = SentenceTransformer(args.model_name)
    model.eval()

    entity_ids = sorted(entity_text_by_id)
    entity_texts = [entity_text_by_id[entity_id] for entity_id in entity_ids]
    entity_vectors = encode_texts(model, entity_texts, args.batch_size)
    entity_embeddings = {
        entity_id: entity_vectors[index] for index, entity_id in enumerate(entity_ids)
    }

    relation_ids = sorted(relation_text_by_id)
    forward_texts = [
        f"forward relation: {relation_text_by_id[relation_id]}"
        for relation_id in relation_ids
    ]
    reverse_texts = [
        f"reverse relation: inverse of {relation_text_by_id[relation_id]}"
        for relation_id in relation_ids
    ]
    forward_vectors = encode_texts(model, forward_texts, args.batch_size)
    reverse_vectors = encode_texts(model, reverse_texts, args.batch_size)
    relation_embeddings = {
        "forward": {
            relation_id: forward_vectors[index]
            for index, relation_id in enumerate(relation_ids)
        },
        "reverse": {
            relation_id: reverse_vectors[index]
            for index, relation_id in enumerate(relation_ids)
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(entity_embeddings, output_dir / "entity_embeddings.pt")
    torch.save(relation_embeddings, output_dir / "relation_embeddings.pt")

    print(f"Saved {len(entity_embeddings):,} entity embeddings")
    print(f"Saved {len(relation_ids):,} relation embeddings in two directions")


if __name__ == "__main__":
    main()
