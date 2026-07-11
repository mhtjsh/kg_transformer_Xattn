"""CPU smoke tests for dependency-light V0 components.

This script does not download Qwen or BioLORD. It checks the local pieces that
can fail before SSH/GPU execution: graph token construction, projection shape,
cross-attention masking, strict parsing, and answer-only loss.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kg_xattn.graph_memory import FrozenEdgeEmbeddingEncoder
from kg_xattn.graph_modules import GatedGraphCrossAttention, GraphMemoryProjector
from kg_xattn.metrics import parse_prediction, score_prediction
from kg_xattn.train_utils import answer_only_loss


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_graph_tokens_and_projector() -> None:
    """Verify that non-padding slots map to real KG edges and project cleanly."""

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        kg_path = tmp_path / "kg_edges.jsonl"
        topics_path = tmp_path / "question_topics.jsonl"
        entity_path = tmp_path / "entity_embeddings.pt"
        relation_path = tmp_path / "relation_embeddings.pt"

        write_jsonl(
            kg_path,
            [
                {
                    "source_id": "drug_a",
                    "source_name": "drug A",
                    "relation_id": "treats",
                    "relation_name": "treats",
                    "target_id": "disease_b",
                    "target_name": "disease B",
                }
            ],
        )
        write_jsonl(topics_path, [{"id": "q1", "topic_entities": ["drug_a"]}])
        torch.save(
            {
                "drug_a": torch.ones(768),
                "disease_b": torch.ones(768) * 2,
            },
            entity_path,
        )
        torch.save(
            {
                "forward": {"treats": torch.ones(768) * 3},
                "reverse": {"treats": torch.ones(768) * 4},
            },
            relation_path,
        )

        encoder = FrozenEdgeEmbeddingEncoder(
            kg_edges_path=kg_path,
            question_topics_path=topics_path,
            entity_embeddings_path=entity_path,
            relation_embeddings_path=relation_path,
            max_memory_tokens=48,
        )
        graph = encoder.encode(["q1"])
        assert graph.raw_memory_tokens.shape == (1, 48, 2304)
        assert graph.memory_mask[0, 0].item() is True
        assert graph.provenance[0][0]["source_id"] == "drug_a"
        assert graph.provenance[0][0]["relation_id"] == "treats"
        assert graph.provenance[0][0]["target_id"] == "disease_b"

        projector = GraphMemoryProjector()
        projected = projector(
            graph.raw_memory_tokens,
            graph.hop_ids,
            graph.direction_ids,
            graph.rank_ids,
            graph.memory_mask,
        )
        assert projected.shape == (1, 48, 1536)
        assert torch.all(projected[0, 1:] == 0)


def test_cross_attention_mask() -> None:
    """Verify padded graph slots get exactly zero attention."""

    torch.manual_seed(17)
    module = GatedGraphCrossAttention()
    hidden = torch.randn(2, 5, 1536)
    memory = torch.randn(2, 48, 1536)
    graph_mask = torch.zeros(2, 48, dtype=torch.bool)
    graph_mask[0, :3] = True
    graph_mask[1, :7] = True
    text_mask = torch.ones(2, 5, dtype=torch.bool)

    result = module(
        hidden_states=hidden,
        graph_memory=memory,
        graph_mask=graph_mask,
        text_mask=text_mask,
        return_weights=True,
    )
    weights = result.attention_weights
    assert weights is not None
    assert torch.isfinite(weights).all()
    assert torch.all(weights[0, :, :, 3:] == 0)
    assert torch.all(weights[1, :, :, 7:] == 0)
    valid_sums = weights.sum(dim=-1)
    assert torch.allclose(valid_sums, torch.ones_like(valid_sums), atol=1e-5)


def test_loss_and_metrics() -> None:
    """Check answer-only loss ignores -100 labels and parser is strict."""

    logits = torch.randn(2, 6, 11)
    labels = torch.full((2, 6), -100, dtype=torch.long)
    labels[:, -2:] = torch.tensor([[4, 5], [6, 7]])
    loss = answer_only_loss(logits, labels)
    assert torch.isfinite(loss)

    valid = parse_prediction("melanoma || esophageal neoplasms")
    invalid = parse_prediction("answer\nbecause...")
    assert valid.valid
    assert not invalid.valid

    score = score_prediction(
        "q1",
        "Esophageal Neoplasms || melanoma.",
        ["melanoma", "esophageal neoplasms"],
    )
    assert score.correct


def main() -> None:
    test_graph_tokens_and_projector()
    test_cross_attention_mask()
    test_loss_and_metrics()
    print("All local smoke tests passed.")


if __name__ == "__main__":
    main()
