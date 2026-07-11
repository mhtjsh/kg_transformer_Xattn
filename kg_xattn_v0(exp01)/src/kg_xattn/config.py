"""Typed configuration objects for the graph-memory experiment.

The training and evaluation scripts load YAML into these dataclasses. Keeping
the experiment settings typed avoids scattering layer numbers, memory sizes,
and optimizer settings through the code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Model and adapter sizes fixed by the V0 protocol."""

    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    graph_layers: list[int] = field(default_factory=lambda: [8, 13, 18, 23])
    hidden_size: int = 1536
    num_query_heads: int = 12
    num_kv_heads: int = 2
    head_dim: int = 128
    max_memory_tokens: int = 48
    raw_graph_dim: int = 2304
    graph_projector_hidden_dim: int = 512
    rms_norm_eps: float = 1e-6
    graph_gate_init: float = 0.01


@dataclass
class LoRAConfig:
    """LoRA settings for the parameter-matched text-only SFT baseline."""

    r: int = 21
    alpha: int = 42
    dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class DataConfig:
    """Paths and prompt settings shared by all three systems."""

    train_path: str = "data/ritek/train.json"
    dev_path: str = "data/ritek/dev.json"
    test_path: str = "data/ritek/test.json"
    kg_edges_path: str = "data/ritek/processed/kg_edges.jsonl"
    question_topics_path: str = "data/ritek/processed/question_topics.jsonl"
    entity_embeddings_path: str = "data/ritek/cache/entity_embeddings.pt"
    relation_embeddings_path: str = "data/ritek/cache/relation_embeddings.pt"
    max_length: int = 512
    max_new_tokens: int = 64
    prompt_template: str = (
        "Answer the question using only entity names from the medical knowledge "
        "graph. If there are multiple answers, separate them with \" || \". "
        "Return only the answer string, with no explanation.\n\n"
        "Question: {question}\nAnswer:"
    )


@dataclass
class TrainingConfig:
    """Optimization settings locked across SFT_LORA and KG_XATTN."""

    seeds: list[int] = field(default_factory=lambda: [17, 42, 2026])
    max_epochs: int = 6
    min_epochs_before_stopping: int = 3
    early_stopping_patience: int = 2
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2.0e-4
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1.0e-8
    matrix_weight_decay: float = 0.01
    norm_and_gate_weight_decay: float = 0.0
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.10
    max_grad_norm: float = 1.0
    dtype: str = "bfloat16"
    use_cache_during_training: bool = False
    gradient_checkpointing: bool = False
    output_dir: str = "runs/v0"


@dataclass
class ExperimentConfig:
    """Top-level configuration object used by scripts."""

    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _merge_dataclass(default_obj: Any, updates: dict[str, Any]) -> Any:
    """Recursively merge a YAML dictionary into a dataclass instance."""

    for key, value in updates.items():
        current = getattr(default_obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(default_obj, key, value)
    return default_obj


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML config while retaining defaults for omitted fields."""

    config = ExperimentConfig()
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return _merge_dataclass(config, raw)
