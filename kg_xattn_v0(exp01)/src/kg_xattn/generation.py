"""Deterministic generation helpers.

The KG_XATTN system uses a custom greedy loop instead of relying on
model.generate. This makes graph-memory persistence explicit on every decoding
step and avoids accidental loss of custom graph arguments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class GenerationTrace:
    """Records whether graph memory reached every selected layer per step."""

    layer_hits_by_step: list[set[int]] = field(default_factory=list)

    def assert_layers_seen_every_step(self, expected_layers: list[int]) -> None:
        """Raise if any generation step missed a selected graph layer."""

        expected = set(expected_layers)
        for step_index, seen in enumerate(self.layer_hits_by_step):
            if seen != expected:
                raise AssertionError(
                    f"generation step {step_index} saw layers {sorted(seen)}, "
                    f"expected {sorted(expected)}"
                )


@torch.no_grad()
def greedy_generate_graph(
    model: Any,
    tokenizer: Any,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    raw_graph_tokens: torch.Tensor | None = None,
    graph_mask: torch.BoolTensor | None = None,
    hop_ids: torch.LongTensor | None = None,
    direction_ids: torch.LongTensor | None = None,
    rank_ids: torch.LongTensor | None = None,
    graph_scale: float = 1.0,
    return_trace: bool = False,
) -> tuple[torch.LongTensor, GenerationTrace | None]:
    """Greedy decode while passing graph tensors at every step."""

    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    generated = input_ids
    current_mask = attention_mask
    trace = GenerationTrace()

    for _ in range(max_new_tokens):
        output = model(
            input_ids=generated,
            attention_mask=current_mask,
            raw_graph_tokens=raw_graph_tokens,
            graph_mask=graph_mask,
            hop_ids=hop_ids,
            direction_ids=direction_ids,
            rank_ids=rank_ids,
            graph_scale=graph_scale,
            return_graph_attentions=return_trace,
        )
        next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        current_mask = torch.cat(
            [
                current_mask,
                torch.ones(
                    current_mask.shape[0],
                    1,
                    dtype=current_mask.dtype,
                    device=current_mask.device,
                ),
            ],
            dim=1,
        )

        if return_trace:
            seen = set(output.graph_attentions.keys() if output.graph_attentions else [])
            trace.layer_hits_by_step.append(seen)

        if eos_token_id is not None and torch.all(next_token.squeeze(-1) == eos_token_id):
            break

    return generated, trace if return_trace else None


def decode_new_tokens(
    tokenizer: Any,
    generated_ids: torch.LongTensor,
    prompt_lengths: list[int],
) -> list[str]:
    """Decode only tokens generated after each example prompt."""

    decoded = []
    for row, prompt_len in zip(generated_ids, prompt_lengths):
        new_ids = row[prompt_len:]
        decoded.append(tokenizer.decode(new_ids, skip_special_tokens=True).strip())
    return decoded
