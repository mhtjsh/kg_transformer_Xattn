"""Graph-memory cross-attention experiment package.

This package implements the V0 experiment described in the protocol:
BASE, text-only LoRA SFT, and KG-memory cross-attention against
Qwen2.5-1.5B-Instruct.
"""

__all__ = [
    "config",
    "data",
    "generation",
    "graph_memory",
    "graph_modules",
    "metrics",
    "modeling_qwen",
    "train_utils",
]
