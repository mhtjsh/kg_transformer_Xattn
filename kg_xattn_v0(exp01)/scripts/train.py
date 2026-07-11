"""Train either SFT_LORA or KG_XATTN for the V0 experiment.

Run from the kg_xattn_v0 directory after installing requirements:

    python scripts/train.py --config configs/v0_qwen25_15b.yaml --system kg_xattn --seed 17

The script intentionally uses a plain PyTorch loop instead of Trainer so the
answer-only per-example loss and graph-memory arguments are visible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kg_xattn.config import load_config
from kg_xattn.data import PromptCollator, QACollator, QADataset, load_qa_examples
from kg_xattn.generation import decode_new_tokens, greedy_generate_graph
from kg_xattn.graph_memory import FrozenEdgeEmbeddingEncoder, GraphMemoryInput
from kg_xattn.metrics import score_prediction, summarize_scores
from kg_xattn.systems import (
    assert_system_separation,
    load_kg_xattn_system,
    load_sft_lora_system,
)
from kg_xattn.train_utils import (
    CosineWithWarmup,
    answer_only_loss,
    build_optimizer,
    count_trainable_parameters,
    qa_nll_per_example,
    set_seed,
)


def make_graph_encoder(cfg: Any) -> FrozenEdgeEmbeddingEncoder:
    """Construct the V0 frozen edge encoder from processed KG caches."""

    return FrozenEdgeEmbeddingEncoder(
        kg_edges_path=cfg.data.kg_edges_path,
        question_topics_path=cfg.data.question_topics_path,
        entity_embeddings_path=cfg.data.entity_embeddings_path,
        relation_embeddings_path=cfg.data.relation_embeddings_path,
        max_memory_tokens=cfg.model.max_memory_tokens,
        max_hops=3,
        embedding_dim=768,
    )


def graph_to_device(graph: GraphMemoryInput, device: torch.device) -> GraphMemoryInput:
    """Move graph tensors while leaving provenance metadata on CPU."""

    return graph.to(device)


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor fields from a collated batch to the training device."""

    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, GraphMemoryInput):
            moved[key] = graph_to_device(value, device)
        else:
            moved[key] = value
    return moved


def forward_logits(
    model: Any,
    system: str,
    batch: dict[str, Any],
) -> torch.Tensor:
    """Run one forward pass and return logits for either trainable system."""

    if system == "kg_xattn":
        graph = batch["graph"]
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            raw_graph_tokens=graph.raw_memory_tokens,
            graph_mask=graph.memory_mask,
            hop_ids=graph.hop_ids,
            direction_ids=graph.direction_ids,
            rank_ids=graph.rank_ids,
            graph_scale=1.0,
        )
        return output.logits

    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    return output.logits


@torch.no_grad()
def evaluate_teacher_forced_nll(
    model: Any,
    system: str,
    tokenizer: Any,
    examples: list[Any],
    cfg: Any,
    graph_encoder: FrozenEdgeEmbeddingEncoder | None,
    device: torch.device,
) -> float:
    """Compute dev QA NLL using the same answer-only labels as training."""

    collator = QACollator(
        tokenizer=tokenizer,
        prompt_template=cfg.data.prompt_template,
        max_length=cfg.data.max_length,
        graph_encoder=graph_encoder,
    )
    loader = DataLoader(
        QADataset(examples),
        batch_size=cfg.training.micro_batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    model.eval()
    nll_values = []
    for batch in loader:
        batch = batch_to_device(batch, device)
        logits = forward_logits(model, system, batch)
        per_example = qa_nll_per_example(logits, batch["labels"])
        nll_values.extend(per_example.detach().float().cpu().tolist())
    return float(sum(nll_values) / max(1, len(nll_values)))


@torch.no_grad()
def evaluate_generation(
    model: Any,
    system: str,
    tokenizer: Any,
    examples: list[Any],
    cfg: Any,
    graph_encoder: FrozenEdgeEmbeddingEncoder | None,
    device: torch.device,
) -> dict[str, Any]:
    """Generate dev answers deterministically and compute strict metrics."""

    collator = PromptCollator(
        tokenizer=tokenizer,
        prompt_template=cfg.data.prompt_template,
        max_length=cfg.data.max_length,
        graph_encoder=graph_encoder,
    )
    loader = DataLoader(
        QADataset(examples),
        batch_size=cfg.training.micro_batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    model.eval()
    scores = []
    predictions = []

    for batch in loader:
        batch = batch_to_device(batch, device)
        if system == "kg_xattn":
            graph = batch["graph"]
            generated, trace = greedy_generate_graph(
                model=model,
                tokenizer=tokenizer,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=cfg.data.max_new_tokens,
                raw_graph_tokens=graph.raw_memory_tokens,
                graph_mask=graph.memory_mask,
                hop_ids=graph.hop_ids,
                direction_ids=graph.direction_ids,
                rank_ids=graph.rank_ids,
                graph_scale=1.0,
                return_trace=True,
            )
            trace.assert_layers_seen_every_step(cfg.model.graph_layers)
        else:
            generated = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                do_sample=False,
                num_beams=1,
                max_new_tokens=cfg.data.max_new_tokens,
                use_cache=False,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        texts = decode_new_tokens(tokenizer, generated, batch["prompt_lengths"])
        for qid, text, gold in zip(batch["question_ids"], texts, batch["answers"]):
            score = score_prediction(qid, text, gold)
            scores.append(score)
            predictions.append({"id": qid, "prediction": text, "gold": gold})

    return {"metrics": summarize_scores(scores), "predictions": predictions}


def save_checkpoint(
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    system: str,
    metadata: dict[str, Any],
) -> None:
    """Save a system-specific checkpoint and metadata."""

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir / "tokenizer")

    if system == "kg_xattn":
        model.save_graph_adapter(output_dir)
    else:
        model.save_pretrained(output_dir / "adapter")

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v0_qwen25_15b.yaml")
    parser.add_argument("--system", choices=["sft_lora", "kg_xattn"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    graph_encoder = make_graph_encoder(cfg) if args.system == "kg_xattn" else None

    if args.system == "kg_xattn":
        model = load_kg_xattn_system(cfg)
    else:
        model = load_sft_lora_system(cfg)
    assert_system_separation(args.system, model)
    model.to(device)

    train_examples = load_qa_examples(cfg.data.train_path)
    dev_examples = load_qa_examples(cfg.data.dev_path)
    if args.train_limit is not None:
        train_examples = train_examples[: args.train_limit]
    if args.dev_limit is not None:
        dev_examples = dev_examples[: args.dev_limit]

    train_collator = QACollator(
        tokenizer=tokenizer,
        prompt_template=cfg.data.prompt_template,
        max_length=cfg.data.max_length,
        graph_encoder=graph_encoder,
    )
    train_loader = DataLoader(
        QADataset(train_examples),
        batch_size=cfg.training.micro_batch_size,
        shuffle=True,
        collate_fn=train_collator,
    )

    updates_per_epoch = max(
        1,
        len(train_loader) // cfg.training.gradient_accumulation_steps,
    )
    total_updates = updates_per_epoch * cfg.training.max_epochs
    optimizer = build_optimizer(
        model=model,
        learning_rate=cfg.training.learning_rate,
        beta1=cfg.training.beta1,
        beta2=cfg.training.beta2,
        epsilon=cfg.training.epsilon,
        matrix_weight_decay=cfg.training.matrix_weight_decay,
        norm_and_gate_weight_decay=cfg.training.norm_and_gate_weight_decay,
    )
    scheduler = CosineWithWarmup(
        optimizer,
        total_steps=total_updates,
        warmup_ratio=cfg.training.warmup_ratio,
        min_lr_ratio=cfg.training.min_lr_ratio,
    )

    run_dir = Path(cfg.training.output_dir) / args.system / f"seed_{args.seed}"
    best_em = -1.0
    best_nll = float("inf")
    bad_epochs = 0

    print(f"System: {args.system}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    for epoch in range(1, cfg.training.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            batch = batch_to_device(batch, device)
            logits = forward_logits(model, args.system, batch)
            loss = answer_only_loss(logits, batch["labels"])
            (loss / cfg.training.gradient_accumulation_steps).backward()
            running_loss += float(loss.detach().cpu())

            should_step = step % cfg.training.gradient_accumulation_steps == 0
            is_last = step == len(train_loader)
            if should_step or is_last:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.training.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        gen_eval = evaluate_generation(
            model=model,
            system=args.system,
            tokenizer=tokenizer,
            examples=dev_examples,
            cfg=cfg,
            graph_encoder=graph_encoder,
            device=device,
        )
        dev_nll = evaluate_teacher_forced_nll(
            model=model,
            system=args.system,
            tokenizer=tokenizer,
            examples=dev_examples,
            cfg=cfg,
            graph_encoder=graph_encoder,
            device=device,
        )
        dev_em = gen_eval["metrics"]["set_em"]
        avg_train_loss = running_loss / max(1, len(train_loader))

        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "dev_set_em": dev_em,
                    "dev_nll": dev_nll,
                    "dev_valid_rate": gen_eval["metrics"]["valid_format_rate"],
                },
                indent=2,
            )
        )

        improved = dev_em > best_em or (dev_em == best_em and dev_nll < best_nll)
        if improved:
            best_em = dev_em
            best_nll = dev_nll
            bad_epochs = 0
            save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                output_dir=run_dir / "best",
                system=args.system,
                metadata={
                    "system": args.system,
                    "seed": args.seed,
                    "epoch": epoch,
                    "dev_metrics": gen_eval["metrics"],
                    "dev_nll": dev_nll,
                    "trainable_parameters": count_trainable_parameters(model),
                },
            )
        else:
            bad_epochs += 1

        if (
            epoch >= cfg.training.min_epochs_before_stopping
            and bad_epochs >= cfg.training.early_stopping_patience
        ):
            break


if __name__ == "__main__":
    main()
