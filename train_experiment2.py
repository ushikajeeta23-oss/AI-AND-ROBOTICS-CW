#!/usr/bin/env python
"""Validate or run the CPU YOLOv8n-cls Experiment 2 transfer-learning recipe."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "experiment2_yolov8n_cls.yaml"
DEFAULT_OUTPUT = ROOT / "models" / "experiment2" / "officemate_yolov8n_cls.pt"
EXPERIMENT1_MODEL = ROOT / "models" / "officemate_yolov8n_cls.pt"


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Training config must be a mapping: {path}")
    return config


def validate_config(config: dict, output_model: Path) -> dict:
    config = dict(config)
    data = Path(config.get("data", ""))
    if not data.is_absolute():
        data = ROOT / data
    if not (data / "train").is_dir() or not (data / "val").is_dir():
        raise FileNotFoundError(
            f"Experiment 2 data must contain train/ and val/: {data}. "
            "Run phase2_build_experiment2.py first."
        )
    if (data / "test").exists():
        raise ValueError(
            "Experiment 2 must not use a test folder during development. "
            "The final test set must be genuinely new and evaluated once."
        )
    if output_model.exists():
        raise FileExistsError(
            f"Experiment 2 model already exists: {output_model}. "
            "Keep the previous experiment unchanged."
        )
    if output_model.resolve() == EXPERIMENT1_MODEL.resolve():
        raise ValueError("Refusing to overwrite the Experiment 1 model.")
    config["data"] = str(data.resolve())
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-model", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually start training. Without this flag the script only validates and prints.",
    )
    args = parser.parse_args()

    config = validate_config(load_config(args.config), args.output_model)
    print(json.dumps(config, indent=2))
    print(f"Planned output model: {args.output_model}")
    print(f"Experiment 1 model is protected: {EXPERIMENT1_MODEL}")

    if not args.execute:
        print("Dry run only. No training started. Pass --execute to train.")
        return

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install the Ultralytics requirements first.") from exc

    torch.set_num_threads(4)
    weights = ROOT / config.pop("model", "yolov8n-cls.pt")
    if not weights.is_file():
        raise FileNotFoundError(f"Local transfer-learning weights missing: {weights}")
    config["project"] = str(ROOT / config["project"])
    config["name"] = str(config["name"])
    config["exist_ok"] = True
    run = Path(config["project"]) / config["name"]
    run.mkdir(parents=True, exist_ok=False)
    model = YOLO(str(weights))
    model.train(**config)
    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Training finished without a best checkpoint: {best}")
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    with best.open("rb") as source, args.output_model.open("xb") as destination:
        shutil.copyfileobj(source, destination)
    print(f"Experiment 2 model written to {args.output_model}")
    print("The Experiment 1 model and results were not modified.")


if __name__ == "__main__":
    main()
