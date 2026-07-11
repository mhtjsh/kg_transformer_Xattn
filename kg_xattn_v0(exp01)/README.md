# KG Cross-Attention V0

This folder implements the V0 experiment from the protocol:

1. `BASE`: frozen `Qwen/Qwen2.5-1.5B-Instruct`, no training.
2. `SFT_LORA`: frozen Qwen plus LoRA rank 21 on Q/K/V/O and MLP projections.
3. `KG_XATTN`: frozen Qwen plus trainable graph-memory projector and cross-attention adapters at layers `[8, 13, 18, 23]`.

The central research question is:

> Does a parameter-matched internal graph-memory cross-attention adapter produce better held-out medical KG QA than ordinary text-only LoRA-SFT?

## What Is Implemented

`src/kg_xattn/graph_memory.py`

This file defines the graph-memory interface. The model receives only tensors:

```text
raw_memory_tokens: [B, 48, 2304]
memory_mask:       [B, 48]
hop_ids:           [B, 48]
direction_ids:     [B, 48]
rank_ids:          [B, 48]
```

`FrozenEdgeEmbeddingEncoder` retrieves up to 48 oriented KG edges from the question topic entities. Each edge becomes:

```text
source_embedding || directional_relation_embedding || target_embedding
```

Gold answers are not used for retrieval. Provenance is stored beside the tensors so graph attention can later be mapped back to exact edges.

`src/kg_xattn/graph_modules.py`

This file contains the trainable graph modules:

```text
GraphMemoryProjector: 2304 -> 512 -> 1536
GatedGraphCrossAttention: LM hidden states query graph-memory keys/values
```

The cross-attention shape is:

```text
Q from text:  [B, 12, T, 128]
K from graph: [B, 12, 48, 128]
V from graph: [B, 12, 48, 128]
attention:    [B, 12, T, 48]
```

Padded graph slots get exactly zero attention.

`src/kg_xattn/modeling_qwen.py`

This wraps Qwen decoder layers. At selected layers the flow is:

```text
self-attention
residual add
graph cross-attention
MLP
residual add
```

With `graph_scale=0`, the wrapper calls the original Qwen layer directly. That is used for the base-equivalence diagnostic.

`src/kg_xattn/data.py`

This renders the identical prompt for all systems and builds answer-only labels. System/question/assistant-prefix/padding labels are `-100`; only answer tokens and the final end token are trained.

`scripts/train.py`

Plain PyTorch training loop for `sft_lora` and `kg_xattn`. It uses the same answer-only loss, optimizer, scheduler, prompt, and dev checkpoint rule.

`scripts/evaluate.py`

Deterministic evaluation for `base`, `sft_lora`, or `kg_xattn`. KG generation uses a custom greedy loop so graph tensors are passed at every decoding step.

## Local Smoke Test

Your local machine does not need GPU or Transformers for the basic smoke test:

```bash
cd kg_xattn_v0
python scripts/smoke_test.py
```

This checks:

- graph token shape `[1, 48, 2304]`
- projected graph token shape `[1, 48, 1536]`
- provenance for source/relation/target/hop/direction
- padded graph slots are zero
- cross-attention padded slots receive zero attention
- valid attention sums to 1
- answer-only loss is finite
- strict answer parser works

## SSH Setup

On the SSH machine:

```bash
cd kg_xattn_v0
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If PyTorch must match a specific CUDA version on that server, install the server-specific PyTorch wheel first, then run `pip install -r requirements.txt`.

## Prepare Data

Download and process the RiTeK/PharmKG dataset:

```bash
python scripts/prepare_ritek.py --output-dir data/ritek
```

If you already downloaded the dataset snapshot:

```bash
python scripts/prepare_ritek.py --source-dir /path/to/snapshot --output-dir data/ritek
```

Then precompute frozen BioLORD embeddings:

```bash
python scripts/precompute_biolord_embeddings.py \
  --kg-edges data/ritek/processed/kg_edges.jsonl \
  --output-dir data/ritek/cache \
  --model-name FremyCompany/BioLORD-2023
```

## Required SSH Checks

Before long training:

```bash
python scripts/ssh_required_checks.py --config configs/v0_qwen25_15b.yaml
```

This checks:

- QA labels mask only non-answer tokens
- `graph_scale=0` matches the original frozen Qwen logits
- frozen Qwen receives no gradients
- graph projector and all Q/K/V/O graph projections receive gradients
- graph memory reaches layers `8, 13, 18, 23` at every decode step
- `SFT_LORA` and `KG_XATTN` trainable parameter counts differ by no more than 1 percent

## Training

Run all three SFT seeds:

```bash
python scripts/train.py --config configs/v0_qwen25_15b.yaml --system sft_lora --seed 17
python scripts/train.py --config configs/v0_qwen25_15b.yaml --system sft_lora --seed 42
python scripts/train.py --config configs/v0_qwen25_15b.yaml --system sft_lora --seed 2026
```

Run all three KG cross-attention seeds:

```bash
python scripts/train.py --config configs/v0_qwen25_15b.yaml --system kg_xattn --seed 17
python scripts/train.py --config configs/v0_qwen25_15b.yaml --system kg_xattn --seed 42
python scripts/train.py --config configs/v0_qwen25_15b.yaml --system kg_xattn --seed 2026
```

For a short server sanity run:

```bash
python scripts/train.py --config configs/v0_qwen25_15b.yaml --system kg_xattn --seed 17 --train-limit 32 --dev-limit 16
```

## Evaluation

Evaluate BASE:

```bash
python scripts/evaluate.py \
  --config configs/v0_qwen25_15b.yaml \
  --system base \
  --split test \
  --output runs/v0/base_test.jsonl
```

Evaluate a trained SFT checkpoint:

```bash
python scripts/evaluate.py \
  --config configs/v0_qwen25_15b.yaml \
  --system sft_lora \
  --checkpoint runs/v0/sft_lora/seed_17/best \
  --split test \
  --output runs/v0/sft_lora_seed17_test.jsonl
```

Evaluate a trained KG checkpoint:

```bash
python scripts/evaluate.py \
  --config configs/v0_qwen25_15b.yaml \
  --system kg_xattn \
  --checkpoint runs/v0/kg_xattn/seed_17/best \
  --split test \
  --output runs/v0/kg_xattn_seed17_test.jsonl
```

Run the graph-off diagnostic on the same trained KG checkpoint:

```bash
python scripts/evaluate.py \
  --config configs/v0_qwen25_15b.yaml \
  --system kg_xattn \
  --checkpoint runs/v0/kg_xattn/seed_17/best \
  --split test \
  --graph-scale 0.0 \
  --output runs/v0/kg_xattn_seed17_graph_off_test.jsonl
```

## Final Report

After evaluating three SFT and three KG seeds:

```bash
python scripts/report_results.py \
  --sft runs/v0/sft_lora_seed17_test.jsonl runs/v0/sft_lora_seed42_test.jsonl runs/v0/sft_lora_seed2026_test.jsonl \
  --kg runs/v0/kg_xattn_seed17_test.jsonl runs/v0/kg_xattn_seed42_test.jsonl runs/v0/kg_xattn_seed2026_test.jsonl \
  --output runs/v0/final_report.json
```

The report computes mean SetEM, per-seed scores, KG minus SFT difference, paired-bootstrap 95 percent CI, and the core positive-result criteria.

## Notes

- The KG model does not serialize KG triples into the prompt.
- The KG model does not use auxiliary graph losses.
- The KG model does not train BioLORD.
- The KG model does not train retrieval.
- Random/shuffled/corrupted graph memory is intentionally left for the later causal-ablation phase.
