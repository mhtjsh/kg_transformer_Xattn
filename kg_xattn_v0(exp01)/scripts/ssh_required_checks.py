"""Run V0 checks that require Transformers and the Qwen checkpoint.

Run this on the SSH/GPU machine after dependency installation. It verifies the
implementation-specific checks that cannot be proven by local CPU smoke tests.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kg_xattn.config import load_config
from kg_xattn.data import QAExample, QACollator, render_prompt
from kg_xattn.generation import greedy_generate_graph
from kg_xattn.systems import load_base_system, load_kg_xattn_system, load_sft_lora_system
from kg_xattn.train_utils import (
    answer_only_loss,
    assert_parameter_budget_match,
    count_trainable_parameters,
)


def toy_graph(batch_size: int, cfg, device: torch.device) -> dict[str, torch.Tensor]:
    """Create deterministic fake graph tensors with valid and padded slots."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(17)
    raw = torch.randn(
        batch_size,
        cfg.model.max_memory_tokens,
        cfg.model.raw_graph_dim,
        generator=generator,
        dtype=torch.float32,
    )
    mask = torch.zeros(batch_size, cfg.model.max_memory_tokens, dtype=torch.bool)
    mask[:, :5] = True
    hop = torch.zeros(batch_size, cfg.model.max_memory_tokens, dtype=torch.long)
    hop[:, :5] = torch.tensor([1, 1, 2, 2, 3])
    direction = torch.full((batch_size, cfg.model.max_memory_tokens), 2, dtype=torch.long)
    direction[:, :5] = torch.tensor([0, 1, 0, 1, 0])
    rank = torch.full(
        (batch_size, cfg.model.max_memory_tokens),
        cfg.model.max_memory_tokens,
        dtype=torch.long,
    )
    rank[:, :5] = torch.arange(5)
    return {
        "raw_graph_tokens": raw.to(device),
        "graph_mask": mask.to(device),
        "hop_ids": hop.to(device),
        "direction_ids": direction.to(device),
        "rank_ids": rank.to(device),
    }


def test_qa_labels(tokenizer, cfg) -> None:
    """Assert only answer/end tokens have non--100 labels."""

    collator = QACollator(
        tokenizer=tokenizer,
        prompt_template=cfg.data.prompt_template,
        max_length=cfg.data.max_length,
    )
    batch = collator([QAExample("q1", "What treats disease B?", ["drug A"])])
    labels = batch["labels"][0]
    input_ids = batch["input_ids"][0]
    non_ignored = torch.where(labels.ne(-100))[0]
    assert len(non_ignored) > 0
    first = int(non_ignored[0])
    assert torch.all(labels[:first] == -100)
    assert torch.equal(labels[non_ignored], input_ids[non_ignored])


@torch.no_grad()
def test_base_equivalence(tokenizer, cfg, device: torch.device) -> float:
    """Compare original Qwen and graph-wrapped Qwen at graph_scale=0."""

    base = load_base_system(cfg).to(device).eval()
    graph_model = load_kg_xattn_system(cfg).to(device).eval()
    text = render_prompt("What treats disease B?", cfg.data.prompt_template)
    encoded = tokenizer(text, return_tensors="pt").to(device)
    graph = toy_graph(1, cfg, device)

    base_logits = base(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        use_cache=False,
    ).logits
    wrapped_logits = graph_model(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        graph_scale=0.0,
        **graph,
    ).logits
    max_diff = float((base_logits - wrapped_logits).abs().max().detach().cpu())
    del base
    del graph_model
    torch.cuda.empty_cache()
    return max_diff


def test_gradients(tokenizer, cfg, device: torch.device) -> dict[str, bool]:
    """Verify frozen Qwen has no grads and every graph adapter path has grads."""

    model = load_kg_xattn_system(cfg).to(device)
    model.train()
    collator = QACollator(
        tokenizer=tokenizer,
        prompt_template=cfg.data.prompt_template,
        max_length=cfg.data.max_length,
    )
    batch = collator([QAExample("q1", "What treats disease B?", ["drug A"])])
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    graph = toy_graph(1, cfg, device)

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        graph_scale=1.0,
        **graph,
    )
    loss = answer_only_loss(output.logits, labels)
    loss.backward()

    original_qwen_ok = all(
        param.grad is None
        for name, param in model.base_model.named_parameters()
        if "cross_attention" not in name and "graph_projector" not in name
    )
    projector_ok = any(
        param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
        for param in model.graph_projector.parameters()
    )
    qkvo_ok = {}
    for layer_idx, module in model.cross_attentions.items():
        for label, submodule in [
            ("q", module.q_proj),
            ("k", module.k_proj),
            ("v", module.v_proj),
            ("o", module.o_proj),
        ]:
            qkvo_ok[f"layer_{layer_idx}_{label}"] = all(
                param.grad is not None
                and torch.isfinite(param.grad).all()
                and param.grad.abs().sum() > 0
                for param in submodule.parameters()
            )
        qkvo_ok[f"layer_{layer_idx}_gate"] = (
            module.alpha.grad is not None and torch.isfinite(module.alpha.grad).all()
        )

    result = {
        "original_qwen_gradients_none": original_qwen_ok,
        "graph_projector_gradients_nonzero": projector_ok,
        **qkvo_ok,
    }
    del model
    torch.cuda.empty_cache()
    return result


@torch.no_grad()
def test_generation_persistence(tokenizer, cfg, device: torch.device) -> None:
    """Assert graph memory reaches all selected layers at every decode step."""

    model = load_kg_xattn_system(cfg).to(device).eval()
    text = render_prompt("What treats disease B?", cfg.data.prompt_template)
    encoded = tokenizer(text, return_tensors="pt").to(device)
    graph = toy_graph(1, cfg, device)
    _, trace = greedy_generate_graph(
        model=model,
        tokenizer=tokenizer,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        max_new_tokens=2,
        return_trace=True,
        **graph,
    )
    trace.assert_layers_seen_every_step(cfg.model.graph_layers)
    del model
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v0_qwen25_15b.yaml")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    test_qa_labels(tokenizer, cfg)

    max_diff = test_base_equivalence(tokenizer, cfg, device)
    threshold = 2.0e-2 if cfg.training.dtype.lower() in {"bf16", "bfloat16"} else 1.0e-5
    assert max_diff <= threshold, f"base equivalence max diff {max_diff} > {threshold}"

    gradient_report = test_gradients(tokenizer, cfg, device)
    assert all(gradient_report.values()), gradient_report

    test_generation_persistence(tokenizer, cfg, device)

    base = load_base_system(cfg)
    sft = load_sft_lora_system(cfg)
    kg = load_kg_xattn_system(cfg)
    assert count_trainable_parameters(base) == 0
    assert_parameter_budget_match(
        count_trainable_parameters(sft),
        count_trainable_parameters(kg),
        tolerance=0.01,
    )

    print(
        json.dumps(
            {
                "qa_label_test": "passed",
                "base_equivalence_max_abs_diff": max_diff,
                "gradient_report": gradient_report,
                "generation_persistence": "passed",
                "parameter_budget": {
                    "base_trainable": count_trainable_parameters(base),
                    "sft_trainable": count_trainable_parameters(sft),
                    "kg_trainable": count_trainable_parameters(kg),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
