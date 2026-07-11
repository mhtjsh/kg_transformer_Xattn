"""QA loading, prompt rendering, and answer-only label construction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class QAExample:
    """One QA record used by BASE, SFT_LORA, and KG_XATTN."""

    question_id: str
    question: str
    answers: list[str]


def _first_present(raw: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    """Read one of several likely key names from heterogeneous QA JSON."""

    for key in keys:
        if key in raw and raw[key] is not None:
            return raw[key]
    return default


def coerce_qa_example(raw: dict[str, Any], fallback_index: int) -> QAExample:
    """Coerce a dataset-specific JSON object into the minimal QA schema."""

    question_id = str(_first_present(raw, ["id", "qid", "question_id"], fallback_index))
    question = str(_first_present(raw, ["question", "query", "input"], "")).strip()
    if not question:
        instruction = str(_first_present(raw, ["instruction"], "")).strip()
        question = instruction

    answers = _first_present(raw, ["answers", "answer", "output", "target"], [])
    if isinstance(answers, str):
        answers = [part.strip() for part in answers.split("||") if part.strip()]
    elif isinstance(answers, dict):
        answers = list(answers.values())
    else:
        answers = [str(answer) for answer in answers]

    if not question:
        raise ValueError(f"QA example {question_id} has no question text")
    if not answers:
        raise ValueError(f"QA example {question_id} has no answer")

    return QAExample(question_id=question_id, question=question, answers=answers)


def load_qa_examples(path: str | Path) -> list[QAExample]:
    """Load JSON, JSONL, or a list-valued JSON file into QAExample objects."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    examples: list[QAExample] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if line.strip():
                    examples.append(coerce_qa_example(json.loads(line), index))
    else:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            if "data" in payload:
                payload = payload["data"]
            elif "examples" in payload:
                payload = payload["examples"]
            else:
                payload = list(payload.values())
        for index, row in enumerate(payload):
            examples.append(coerce_qa_example(row, index))
    return examples


def render_prompt(question: str, prompt_template: str) -> str:
    """Render the exact shared text prompt used by all systems."""

    return prompt_template.format(question=question.strip())


def render_target(answers: list[str]) -> str:
    """Render gold answers in the required delimiter format."""

    return " || ".join(answer.strip() for answer in answers if answer.strip())


def build_answer_only_encoding(
    tokenizer: Any,
    prompt: str,
    target: str,
    max_length: int,
) -> dict[str, list[int]]:
    """Tokenize prompt+target and mask all non-answer labels with -100."""

    eos = tokenizer.eos_token or ""
    prompt_text = prompt.rstrip() + " "
    full_text = prompt_text + target.strip() + eos

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    full_ids = tokenizer(
        full_text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    labels = [-100] * len(full_ids)
    answer_start = min(len(prompt_ids), len(full_ids))
    for index in range(answer_start, len(full_ids)):
        labels[index] = full_ids[index]

    return {"input_ids": full_ids, "labels": labels}


def build_prompt_only_encoding(
    tokenizer: Any,
    prompt: str,
    max_length: int,
) -> dict[str, list[int]]:
    """Tokenize only the prompt for evaluation-time generation."""

    prompt_text = prompt.rstrip() + " "
    return tokenizer(
        prompt_text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )


class QADataset(torch.utils.data.Dataset):
    """Tiny Dataset wrapper around a list of QAExample records."""

    def __init__(self, examples: list[QAExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> QAExample:
        return self.examples[index]


class QACollator:
    """Collate QA examples and optionally attach graph-memory tensors."""

    def __init__(
        self,
        tokenizer: Any,
        prompt_template: str,
        max_length: int,
        graph_encoder: Any | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.max_length = max_length
        self.graph_encoder = graph_encoder

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, examples: list[QAExample]) -> dict[str, Any]:
        encoded = []
        for example in examples:
            prompt = render_prompt(example.question, self.prompt_template)
            target = render_target(example.answers)
            encoded.append(
                build_answer_only_encoding(
                    self.tokenizer,
                    prompt=prompt,
                    target=target,
                    max_length=self.max_length,
                )
            )

        max_len = max(len(row["input_ids"]) for row in encoded)
        input_ids = []
        labels = []
        attention_mask = []
        for row in encoded:
            pad_len = max_len - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
            labels.append(row["labels"] + [-100] * pad_len)
            attention_mask.append([1] * len(row["input_ids"]) + [0] * pad_len)

        batch: dict[str, Any] = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "question_ids": [example.question_id for example in examples],
            "questions": [example.question for example in examples],
            "answers": [example.answers for example in examples],
        }

        if self.graph_encoder is not None:
            batch["graph"] = self.graph_encoder.encode(batch["question_ids"])

        return batch


class PromptCollator:
    """Collate prompt-only batches for deterministic generation."""

    def __init__(
        self,
        tokenizer: Any,
        prompt_template: str,
        max_length: int,
        graph_encoder: Any | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.max_length = max_length
        self.graph_encoder = graph_encoder

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, examples: list[QAExample]) -> dict[str, Any]:
        encoded = [
            build_prompt_only_encoding(
                self.tokenizer,
                render_prompt(example.question, self.prompt_template),
                self.max_length,
            )
            for example in examples
        ]

        max_len = max(len(row["input_ids"]) for row in encoded)
        input_ids = []
        attention_mask = []
        prompt_lengths = []
        for row in encoded:
            ids = row["input_ids"]
            pad_len = max_len - len(ids)
            # Decoder-only generation is safer with left padding because the
            # last non-padding token is the prompt end for every row.
            input_ids.append([self.tokenizer.pad_token_id] * pad_len + ids)
            attention_mask.append([0] * pad_len + [1] * len(ids))
            prompt_lengths.append(max_len)

        batch: dict[str, Any] = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "prompt_lengths": prompt_lengths,
            "question_ids": [example.question_id for example in examples],
            "questions": [example.question for example in examples],
            "answers": [example.answers for example in examples],
        }

        if self.graph_encoder is not None:
            batch["graph"] = self.graph_encoder.encode(batch["question_ids"])

        return batch
