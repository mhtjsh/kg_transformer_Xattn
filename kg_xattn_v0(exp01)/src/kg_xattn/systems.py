"""Construction helpers for BASE, SFT_LORA, and KG_XATTN systems."""

from __future__ import annotations

from typing import Any

import torch

from kg_xattn.config import ExperimentConfig
from kg_xattn.modeling_qwen import GraphMemoryQwenForCausalLM


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    """Map config dtype names to torch dtypes."""

    normalized = dtype_name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_base_system(cfg: ExperimentConfig, **from_pretrained_kwargs: Any) -> Any:
    """Load the frozen BASE system without LoRA or graph modules."""

    from transformers import AutoModelForCausalLM

    dtype = resolve_torch_dtype(cfg.training.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model_name,
        torch_dtype=dtype,
        **from_pretrained_kwargs,
    )
    for param in model.parameters():
        param.requires_grad = False
    model.config.use_cache = False
    return model


def load_sft_lora_system(cfg: ExperimentConfig, **from_pretrained_kwargs: Any) -> Any:
    """Load Qwen with only LoRA adapters trainable."""

    from peft import LoraConfig as PeftLoraConfig
    from peft import TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    dtype = resolve_torch_dtype(cfg.training.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model_name,
        torch_dtype=dtype,
        **from_pretrained_kwargs,
    )
    model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad = False

    lora_config = PeftLoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=cfg.lora.target_modules,
    )
    return get_peft_model(model, lora_config)


def load_kg_xattn_system(cfg: ExperimentConfig, **from_pretrained_kwargs: Any) -> Any:
    """Load frozen Qwen with trainable graph projector and cross-attention."""

    dtype = resolve_torch_dtype(cfg.training.dtype)
    model = GraphMemoryQwenForCausalLM.from_pretrained(
        cfg.model.base_model_name,
        graph_layers=cfg.model.graph_layers,
        torch_dtype=dtype,
        **from_pretrained_kwargs,
    )
    model.base_model.config.use_cache = False
    return model


def assert_system_separation(system_name: str, model: Any) -> None:
    """Check that headline systems do not contain the wrong adaptation path."""

    names = [name for name, _ in model.named_modules()]
    has_lora = any("lora" in name.lower() for name in names)
    has_graph = any("cross_attentions" in name or "graph_projector" in name for name in names)

    if system_name == "sft_lora" and has_graph:
        raise AssertionError("SFT_LORA must not contain graph modules")
    if system_name == "kg_xattn" and has_lora:
        raise AssertionError("KG_XATTN must not contain LoRA modules")
