"""Graph-memory encoder interfaces and the V0 frozen edge encoder.

The model must not know how raw graph vectors are built. It only consumes the
GraphMemoryInput tensors defined here. That keeps V0 simple while leaving a
clean replacement point for later R-GCN or Graph Transformer encoders.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from kg_xattn.metrics import normalize_answer_component


@dataclass
class GraphMemoryInput:
    """Batch of raw graph-memory inputs before LM-space projection."""

    raw_memory_tokens: torch.Tensor
    memory_mask: torch.Tensor
    hop_ids: torch.Tensor
    direction_ids: torch.Tensor
    rank_ids: torch.Tensor
    provenance: list[list[dict[str, Any]]]

    def to(self, device: torch.device | str) -> "GraphMemoryInput":
        """Move tensors to a device while keeping provenance on CPU."""

        return GraphMemoryInput(
            raw_memory_tokens=self.raw_memory_tokens.to(device),
            memory_mask=self.memory_mask.to(device),
            hop_ids=self.hop_ids.to(device),
            direction_ids=self.direction_ids.to(device),
            rank_ids=self.rank_ids.to(device),
            provenance=self.provenance,
        )


class BaseGraphMemoryEncoder(ABC):
    """Abstract interface consumed by the KG_XATTN model and data collator."""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimension of each raw memory vector before GraphMemoryProjector."""

    @abstractmethod
    def encode(self, question_ids: list[str]) -> GraphMemoryInput:
        """Encode a batch of question IDs into raw graph-memory tensors."""


@dataclass(frozen=True)
class KGEdge:
    """Canonical directed KG edge loaded from processed JSONL."""

    source_id: str
    source_name: str
    relation_id: str
    relation_name: str
    target_id: str
    target_name: str


@dataclass(frozen=True)
class OrientedEdge:
    """Traversal-specific edge used as one graph-memory slot."""

    source_id: str
    source_name: str
    relation_id: str
    relation_name: str
    target_id: str
    target_name: str
    direction: str
    hop: int
    retrieval_score: float


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL rows from disk."""

    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _coerce_edge(row: dict[str, Any]) -> KGEdge:
    """Accept several common edge key conventions and produce KGEdge."""

    source_id = str(row.get("source_id", row.get("head_id", row.get("head"))))
    target_id = str(row.get("target_id", row.get("tail_id", row.get("tail"))))
    relation_id = str(row.get("relation_id", row.get("relation", row.get("rel"))))

    return KGEdge(
        source_id=source_id,
        source_name=str(row.get("source_name", row.get("head_name", source_id))),
        relation_id=relation_id,
        relation_name=str(row.get("relation_name", row.get("relation", relation_id))),
        target_id=target_id,
        target_name=str(row.get("target_name", row.get("tail_name", target_id))),
    )


def _load_embedding_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Load and L2-normalize an embedding dictionary saved with torch.save."""

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a dictionary")

    normalized: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.ndim != 1:
            raise ValueError(f"embedding for {key!r} must be 1-D")
        normalized[str(key)] = F.normalize(tensor, p=2, dim=0)
    return normalized


def _load_relation_embeddings(path: str | Path) -> dict[tuple[str, str], torch.Tensor]:
    """Load directional relation embeddings.

    The preferred file format is:
        {"forward": {rel_id: tensor}, "reverse": {rel_id: tensor}}

    For convenience, flat keys such as "REL::forward" are also accepted.
    """

    payload = torch.load(path, map_location="cpu")
    result: dict[tuple[str, str], torch.Tensor] = {}

    if "forward" in payload and "reverse" in payload:
        for direction in ["forward", "reverse"]:
            for rel_id, value in payload[direction].items():
                result[(str(rel_id), direction)] = F.normalize(
                    torch.as_tensor(value, dtype=torch.float32), p=2, dim=0
                )
    else:
        for key, value in payload.items():
            text_key = str(key)
            if "::" not in text_key:
                raise ValueError(
                    "flat relation embedding keys must look like 'relation::forward'"
                )
            rel_id, direction = text_key.rsplit("::", 1)
            result[(rel_id, direction)] = F.normalize(
                torch.as_tensor(value, dtype=torch.float32), p=2, dim=0
            )

    return result


class FrozenEdgeEmbeddingEncoder(BaseGraphMemoryEncoder):
    """V0 graph-memory encoder: one oriented edge becomes one memory token.

    Retrieval is question anchored through topic entities only. Gold answers are
    never used to decide which edges enter graph memory.
    """

    def __init__(
        self,
        kg_edges_path: str | Path,
        question_topics_path: str | Path,
        entity_embeddings_path: str | Path,
        relation_embeddings_path: str | Path,
        max_memory_tokens: int = 48,
        max_hops: int = 3,
        embedding_dim: int = 768,
    ) -> None:
        self.max_memory_tokens = max_memory_tokens
        self.max_hops = max_hops
        self.embedding_dim = embedding_dim
        self._output_dim = embedding_dim * 3

        self.entity_embeddings = _load_embedding_dict(entity_embeddings_path)
        self.relation_embeddings = _load_relation_embeddings(relation_embeddings_path)
        self.edges = [_coerce_edge(row) for row in _load_jsonl(kg_edges_path)]
        self.question_topics = self._load_question_topics(question_topics_path)
        self.adjacency = self._build_adjacency(self.edges)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def _load_question_topics(self, path: str | Path) -> dict[str, list[str]]:
        """Load question_id -> topic entity IDs."""

        mapping: dict[str, list[str]] = {}
        for row in _load_jsonl(path):
            question_id = str(row.get("id", row.get("qid", row.get("question_id"))))
            topics = row.get("topic_entities", row.get("topic_entity", []))
            if isinstance(topics, str):
                topics = [topics]
            mapping[question_id] = [str(topic) for topic in topics]
        return mapping

    def _build_adjacency(
        self, edges: list[KGEdge]
    ) -> dict[str, list[tuple[str, KGEdge]]]:
        """Build forward and reverse traversal adjacency without changing KG."""

        adjacency: dict[str, list[tuple[str, KGEdge]]] = defaultdict(list)
        for edge in edges:
            adjacency[edge.source_id].append(("forward", edge))
            adjacency[edge.target_id].append(("reverse", edge))
        return adjacency

    def _embedding_or_zero(self, entity_id: str) -> torch.Tensor:
        """Return an entity embedding, or zero when the cache lacks it."""

        return self.entity_embeddings.get(
            entity_id, torch.zeros(self.embedding_dim, dtype=torch.float32)
        )

    def _relation_embedding_or_zero(
        self, relation_id: str, direction: str
    ) -> torch.Tensor:
        """Return a directional relation embedding, or zero if unavailable."""

        return self.relation_embeddings.get(
            (relation_id, direction),
            torch.zeros(self.embedding_dim, dtype=torch.float32),
        )

    def retrieve_edges(self, question_id: str) -> list[OrientedEdge]:
        """Retrieve up to max_memory_tokens oriented edges by BFS hop order."""

        topic_entities = self.question_topics.get(str(question_id), [])
        if not topic_entities:
            return []

        queue = deque((entity_id, 0) for entity_id in topic_entities)
        visited_entities = set(topic_entities)
        seen_edges: set[tuple[str, str, str]] = set()
        memory_edges: list[OrientedEdge] = []

        while queue and len(memory_edges) < self.max_memory_tokens:
            current_entity, depth = queue.popleft()
            if depth >= self.max_hops:
                continue

            next_hop = depth + 1
            for direction, edge in self.adjacency.get(current_entity, []):
                if direction == "forward":
                    source_id = edge.source_id
                    source_name = edge.source_name
                    target_id = edge.target_id
                    target_name = edge.target_name
                    next_entity = edge.target_id
                else:
                    source_id = edge.target_id
                    source_name = edge.target_name
                    target_id = edge.source_id
                    target_name = edge.source_name
                    next_entity = edge.source_id

                # Keep one memory token per canonical KG edge. If the edge is
                # first reached from the tail side, that one token is reverse;
                # if it is first reached from the head side, it is forward.
                edge_key = (edge.source_id, edge.relation_id, edge.target_id)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)

                score = 1.0 / float(next_hop)
                memory_edges.append(
                    OrientedEdge(
                        source_id=source_id,
                        source_name=source_name,
                        relation_id=edge.relation_id,
                        relation_name=edge.relation_name,
                        target_id=target_id,
                        target_name=target_name,
                        direction=direction,
                        hop=next_hop,
                        retrieval_score=score,
                    )
                )

                if next_entity not in visited_entities:
                    visited_entities.add(next_entity)
                    queue.append((next_entity, next_hop))

                if len(memory_edges) >= self.max_memory_tokens:
                    break

        return memory_edges

    def _edge_to_raw_vector(self, edge: OrientedEdge) -> torch.Tensor:
        """Concatenate source, directional relation, and target vectors."""

        source = self._embedding_or_zero(edge.source_id)
        relation = self._relation_embedding_or_zero(edge.relation_id, edge.direction)
        target = self._embedding_or_zero(edge.target_id)
        return torch.cat([source, relation, target], dim=0)

    def encode(self, question_ids: list[str]) -> GraphMemoryInput:
        """Create a padded graph-memory batch for a list of question IDs."""

        batch_size = len(question_ids)
        raw = torch.zeros(
            batch_size,
            self.max_memory_tokens,
            self.output_dim,
            dtype=torch.float32,
        )
        mask = torch.zeros(batch_size, self.max_memory_tokens, dtype=torch.bool)
        hop_ids = torch.zeros(batch_size, self.max_memory_tokens, dtype=torch.long)
        direction_ids = torch.full(
            (batch_size, self.max_memory_tokens), 2, dtype=torch.long
        )
        rank_ids = torch.full(
            (batch_size, self.max_memory_tokens),
            self.max_memory_tokens,
            dtype=torch.long,
        )
        provenance: list[list[dict[str, Any]]] = []

        for batch_index, question_id in enumerate(question_ids):
            edges = self.retrieve_edges(question_id)
            row_provenance: list[dict[str, Any]] = []
            for rank, edge in enumerate(edges[: self.max_memory_tokens]):
                raw[batch_index, rank] = self._edge_to_raw_vector(edge)
                mask[batch_index, rank] = True
                hop_ids[batch_index, rank] = edge.hop
                direction_ids[batch_index, rank] = 0 if edge.direction == "forward" else 1
                rank_ids[batch_index, rank] = rank
                row_provenance.append(
                    {
                        "memory_index": rank,
                        "source_id": edge.source_id,
                        "source_name": edge.source_name,
                        "relation_id": edge.relation_id,
                        "relation_name": edge.relation_name,
                        "target_id": edge.target_id,
                        "target_name": edge.target_name,
                        "direction": edge.direction,
                        "hop": edge.hop,
                        "retrieval_score": edge.retrieval_score,
                    }
                )
            provenance.append(row_provenance)

        return GraphMemoryInput(
            raw_memory_tokens=raw,
            memory_mask=mask,
            hop_ids=hop_ids,
            direction_ids=direction_ids,
            rank_ids=rank_ids,
            provenance=provenance,
        )

    def evidence_flags(
        self,
        question_id: str,
        gold_answers: list[str],
    ) -> dict[str, bool]:
        """Report whether normalized gold answer names appear in memory."""

        edges = self.retrieve_edges(question_id)
        memory_names = set()
        for edge in edges:
            memory_names.add(normalize_answer_component(edge.source_name))
            memory_names.add(normalize_answer_component(edge.target_name))

        gold = {
            normalize_answer_component(answer)
            for answer in gold_answers
            if normalize_answer_component(answer)
        }
        present = gold & memory_names
        return {
            "gold_answer_reachable_within_3_hops": bool(present),
            "gold_any_present_in_memory": bool(present),
            "gold_all_present_in_memory": bool(gold) and gold.issubset(memory_names),
        }
