"""Evaluate BASE, SFT_LORA, or KG_XATTN checkpoints on dev/test data."""

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
from kg_xattn.data import PromptCollator, QADataset, load_qa_examples
from kg_xattn.generation import decode_new_tokens, greedy_generate_graph
from kg_xattn.graph_memory import FrozenEdgeEmbeddingEncoder, GraphMemoryInput
from kg_xattn.metrics import score_prediction, summarize_scores
from kg_xattn.systems import load_base_system, load_kg_xattn_system


def make_graph_encoder(cfg: Any) -> FrozenEdgeEmbeddingEncoder:
    return FrozenEdgeEmbeddingEncoder(
        kg_edges_path=cfg.data.kg_edges_path,
        question_topics_path=cfg.data.question_topics_path,
        entity_embeddings_path=cfg.data.entity_embeddings_path,
        relation_embeddings_path=cfg.data.relation_embeddings_path,
        max_memory_tokens=cfg.model.max_memory_tokens,
        max_hops=3,
        embedding_dim=768,
    )


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, GraphMemoryInput):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def load_checkpointed_model(args: Any, cfg: Any, device: torch.device) -> Any:
    """Load model weights for the requested system."""

    if args.system == "base":
        model = load_base_system(cfg)
    elif args.system == "sft_lora":
        from peft import PeftModel

        model = load_base_system(cfg)
        if args.checkpoint:
            model = PeftModel.from_pretrained(model, Path(args.checkpoint) / "adapter")
    else:
        model = load_kg_xattn_system(cfg)
        if args.checkpoint:
            model.load_graph_adapter(Path(args.checkpoint) / "graph_adapter.pt")

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v0_qwen25_15b.yaml")
    parser.add_argument("--system", choices=["base", "sft_lora", "kg_xattn"], required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="runs/v0/eval_predictions.jsonl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--graph-scale", type=float, default=1.0)
    args = parser.parse_args()

    cfg = load_config(args.config)

    from transformers import AutoTokenizer

    tokenizer_path = (
        Path(args.checkpoint) / "tokenizer"
        if args.checkpoint and (Path(args.checkpoint) / "tokenizer").exists()
        else cfg.model.base_model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    graph_encoder = make_graph_encoder(cfg) if args.system == "kg_xattn" else None
    model = load_checkpointed_model(args, cfg, device)
    if args.system == "kg_xattn":
        model.set_graph_scale(args.graph_scale)

    examples = load_qa_examples(cfg.data.test_path if args.split == "test" else cfg.data.dev_path)
    if args.limit is not None:
        examples = examples[: args.limit]

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

    output_rows = []
    scores = []
    for batch in loader:
        batch = batch_to_device(batch, device)
        if args.system == "kg_xattn":
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
                graph_scale=args.graph_scale,
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
        for index, (qid, text, gold) in enumerate(
            zip(batch["question_ids"], texts, batch["answers"])
        ):
            evidence = None
            if graph_encoder is not None:
                evidence = graph_encoder.evidence_flags(qid, gold)
            score = score_prediction(
                qid,
                text,
                gold,
                evidence_present=evidence["gold_any_present_in_memory"]
                if evidence
                else None,
            )
            scores.append(score)
            output_rows.append(
                {
                    "id": qid,
                    "answers": [part for part in text.split("||") if part.strip()],
                    "raw_prediction": text,
                    "gold": gold,
                    "evidence": evidence,
                }
            )

    metrics = summarize_scores(scores)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_path.with_suffix(".metrics.json")).open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
