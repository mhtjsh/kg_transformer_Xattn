"""Shared training utilities for SFT_LORA and KG_XATTN.

The important detail here is the answer-only, per-example normalized loss. The
Hugging Face default causal-LM loss averages over all labeled answer tokens in a
batch. The protocol asks us to normalize each example by its own answer-token
count first, then average examples.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    """Set Python and PyTorch seeds for a repeatable run."""

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of parameters with requires_grad=True."""

    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def count_all_parameters(model: nn.Module) -> int:
    """Return total parameter count, trainable or frozen."""

    return sum(param.numel() for param in model.parameters())


def answer_only_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute per-example normalized autoregressive QA cross-entropy.

    labels must already contain -100 for system/question/assistant-prefix/padding
    tokens and real token IDs for answer tokens plus the assistant end token.
    """

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    flat_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    )
    token_loss = flat_loss.view(shift_labels.shape)
    token_mask = shift_labels.ne(-100)

    per_example_den = token_mask.sum(dim=1).clamp_min(1)
    per_example_num = (token_loss * token_mask).sum(dim=1)
    per_example_loss = per_example_num / per_example_den
    return per_example_loss.mean()


def qa_nll_per_example(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return per-example normalized QA NLL for checkpoint selection."""

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    flat_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    )
    token_loss = flat_loss.view(shift_labels.shape)
    token_mask = shift_labels.ne(-100)
    return (token_loss * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp_min(1)


def build_optimizer(
    model: nn.Module,
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    matrix_weight_decay: float,
    norm_and_gate_weight_decay: float,
) -> torch.optim.Optimizer:
    """Create AdamW groups that avoid weight decay on norms, embeddings, gates."""

    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lower = name.lower()
        if (
            param.ndim < 2
            or "norm" in lower
            or "embedding" in lower
            or lower.endswith("alpha")
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": matrix_weight_decay},
            {"params": no_decay_params, "weight_decay": norm_and_gate_weight_decay},
        ],
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=epsilon,
    )


class CosineWithWarmup:
    """Small scheduler wrapper independent of Accelerate/Trainer."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_ratio: float,
        min_lr_ratio: float,
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = max(1, total_steps)
        self.warmup_steps = max(1, int(total_steps * warmup_ratio))
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_index = 0

    def step(self) -> None:
        self.step_index += 1
        if self.step_index <= self.warmup_steps:
            scale = self.step_index / self.warmup_steps
        else:
            progress = (self.step_index - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            scale = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

        for lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = lr * scale


def assert_parameter_budget_match(
    sft_trainable: int,
    kg_trainable: int,
    tolerance: float = 0.01,
) -> None:
    """Enforce the required <=1 percent trainable-parameter difference."""

    if kg_trainable <= 0:
        raise ValueError("kg_trainable must be positive")
    relative = abs(sft_trainable - kg_trainable) / kg_trainable
    if relative > tolerance:
        raise AssertionError(
            "SFT_LORA and KG_XATTN trainable parameter counts differ by "
            f"{relative:.4%}, above the allowed {tolerance:.2%}. "
            f"sft={sft_trainable}, kg={kg_trainable}"
        )


def save_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    """Write dictionaries as JSONL without pulling in pandas."""

    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
