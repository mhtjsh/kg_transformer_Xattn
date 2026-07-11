"""Check frozen/trainable parameter counts for the V0 headline systems."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kg_xattn.config import load_config
from kg_xattn.systems import (
    assert_system_separation,
    load_base_system,
    load_kg_xattn_system,
    load_sft_lora_system,
)
from kg_xattn.train_utils import (
    assert_parameter_budget_match,
    count_all_parameters,
    count_trainable_parameters,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v0_qwen25_15b.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    base = load_base_system(cfg)
    sft = load_sft_lora_system(cfg)
    kg = load_kg_xattn_system(cfg)

    assert_system_separation("sft_lora", sft)
    assert_system_separation("kg_xattn", kg)

    base_trainable = count_trainable_parameters(base)
    sft_trainable = count_trainable_parameters(sft)
    kg_trainable = count_trainable_parameters(kg)
    assert base_trainable == 0
    assert_parameter_budget_match(sft_trainable, kg_trainable, tolerance=0.01)

    report = {
        "base": {
            "total": count_all_parameters(base),
            "trainable": base_trainable,
        },
        "sft_lora": {
            "total": count_all_parameters(sft),
            "trainable": sft_trainable,
        },
        "kg_xattn": {
            "total": count_all_parameters(kg),
            "trainable": kg_trainable,
        },
        "relative_trainable_difference": abs(sft_trainable - kg_trainable)
        / kg_trainable,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
