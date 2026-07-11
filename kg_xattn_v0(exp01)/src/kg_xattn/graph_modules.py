"""Neural modules for continuous graph-memory infusion.

The two central modules are:

1. GraphMemoryProjector: maps raw edge vectors into the LM hidden space.
2. GatedGraphCrossAttention: lets LM hidden states attend to graph memory.

These modules are dependency-light and can be smoke-tested on CPU without
downloading Qwen or the RiTeK dataset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Qwen-style RMSNorm used to avoid importing Transformers in smoke tests."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class GraphMemoryProjector(nn.Module):
    """Project raw oriented-edge vectors into Qwen hidden space.

    V0 intentionally keeps graph encoding simple: every memory slot is one
    oriented edge represented as source || directional-relation || target.
    Structural embeddings then tell the LM which hop, direction, and retrieval
    rank this memory slot came from.
    """

    def __init__(
        self,
        raw_graph_dim: int = 2304,
        projector_hidden_dim: int = 512,
        lm_hidden_dim: int = 1536,
        max_memory_tokens: int = 48,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.raw_graph_dim = raw_graph_dim
        self.lm_hidden_dim = lm_hidden_dim
        self.max_memory_tokens = max_memory_tokens

        self.input_norm = RMSNorm(raw_graph_dim, eps=eps)
        self.down = nn.Linear(raw_graph_dim, projector_hidden_dim, bias=False)
        self.up = nn.Linear(projector_hidden_dim, lm_hidden_dim, bias=False)

        # 0 is reserved for padding; 1, 2, and 3 are actual retrieval hops.
        self.hop_embedding = nn.Embedding(4, lm_hidden_dim)

        # 0 = forward, 1 = reverse, 2 = padding.
        self.direction_embedding = nn.Embedding(3, lm_hidden_dim)

        # Ranks 0..47 plus padding rank 48.
        self.rank_embedding = nn.Embedding(max_memory_tokens + 1, lm_hidden_dim)
        self.output_norm = RMSNorm(lm_hidden_dim, eps=eps)

    def forward(
        self,
        raw_graph_tokens: torch.Tensor,
        hop_ids: torch.Tensor,
        direction_ids: torch.Tensor,
        rank_ids: torch.Tensor,
        graph_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return graph memory with shape [B, M, lm_hidden_dim]."""

        x = self.input_norm(raw_graph_tokens.float())
        x = self.down(x)
        x = F.silu(x)
        x = self.up(x)

        x = (
            x
            + self.hop_embedding(hop_ids)
            + self.direction_embedding(direction_ids)
            + self.rank_embedding(rank_ids)
        )

        x = self.output_norm(x)

        # Padded graph slots are kept exactly zero so they cannot leak through
        # keys/values if a later mask bug appears.
        x = x * graph_mask.unsqueeze(-1).to(x.dtype)
        return x


@dataclass
class CrossAttentionResult:
    """Output container for graph cross-attention."""

    hidden_states: torch.Tensor
    attention_weights: torch.Tensor | None


class GatedGraphCrossAttention(nn.Module):
    """Grouped-query cross-attention from text states to graph memory.

    Queries come from the current LM hidden states. Keys and values come from
    projected graph-memory tokens. The scalar gate starts small so the modified
    model begins close to the frozen base model while gradients still reach all
    graph adapter projections on the first backward pass.
    """

    def __init__(
        self,
        hidden_size: int = 1536,
        num_query_heads: int = 12,
        num_kv_heads: int = 2,
        head_dim: int = 128,
        eps: float = 1e-6,
        gate_init: float = 0.01,
    ) -> None:
        super().__init__()
        if num_query_heads % num_kv_heads != 0:
            raise ValueError("num_query_heads must be divisible by num_kv_heads")
        if hidden_size != num_query_heads * head_dim:
            raise ValueError("hidden_size must equal num_query_heads * head_dim")

        self.hidden_size = hidden_size
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_repeat = num_query_heads // num_kv_heads

        self.query_norm = RMSNorm(hidden_size, eps=eps)
        self.memory_norm = RMSNorm(hidden_size, eps=eps)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.alpha = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(
        self,
        hidden_states: torch.Tensor,
        graph_memory: torch.Tensor,
        graph_mask: torch.Tensor,
        text_mask: torch.Tensor,
        graph_scale: float = 1.0,
        return_weights: bool = False,
    ) -> CrossAttentionResult:
        batch_size, text_length, _ = hidden_states.shape
        memory_length = graph_memory.shape[1]

        q_input = self.query_norm(hidden_states)
        kv_input = self.memory_norm(graph_memory)

        q = self.q_proj(q_input)
        k = self.k_proj(kv_input)
        v = self.v_proj(kv_input)

        q = q.view(batch_size, text_length, self.num_query_heads, self.head_dim)
        q = q.transpose(1, 2)

        k = k.view(batch_size, memory_length, self.num_kv_heads, self.head_dim)
        k = k.transpose(1, 2)

        v = v.view(batch_size, memory_length, self.num_kv_heads, self.head_dim)
        v = v.transpose(1, 2)

        k = k.repeat_interleave(self.kv_repeat, dim=1)
        v = v.repeat_interleave(self.kv_repeat, dim=1)

        attention_logits = torch.matmul(q.float(), k.float().transpose(-1, -2))
        attention_logits = attention_logits / math.sqrt(self.head_dim)

        valid_memory = graph_mask.to(torch.bool)
        invalid_memory = ~valid_memory[:, None, None, :]
        attention_logits = attention_logits.masked_fill(invalid_memory, -1.0e30)

        # Rows with no valid memory would otherwise softmax to NaN. We zero
        # them after softmax and renormalize only rows that have memory.
        attention_weights = torch.softmax(attention_logits, dim=-1)
        attention_weights = attention_weights * valid_memory[:, None, None, :].to(
            attention_weights.dtype
        )
        normalizer = attention_weights.sum(dim=-1, keepdim=True)
        attention_weights = torch.where(
            normalizer > 0,
            attention_weights / normalizer.clamp_min(1.0e-12),
            torch.zeros_like(attention_weights),
        )

        context = torch.matmul(attention_weights, v.float())
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, text_length, self.hidden_size)
        context = self.o_proj(context.to(hidden_states.dtype))

        # Padding text positions must not receive graph residual updates.
        context = context * text_mask.unsqueeze(-1).to(context.dtype)
        gated_context = float(graph_scale) * torch.tanh(self.alpha) * context
        output = hidden_states + gated_context

        return CrossAttentionResult(
            hidden_states=output,
            attention_weights=attention_weights if return_weights else None,
        )


def trainable_parameter_count(module: nn.Module) -> int:
    """Count trainable parameters in a PyTorch module."""

    return sum(param.numel() for param in module.parameters() if param.requires_grad)
