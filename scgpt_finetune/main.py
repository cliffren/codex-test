"""Command line interface for basic scGPT fine-tuning.

This script demonstrates how to invoke the simplified training routine
provided in :mod:`scgpt_finetune.train`. It is intentionally minimal and
serves as a placeholder for a more feature-rich CLI or REST API.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data import load_csv, load_h5ad, discretize
from .train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="scGPT fine-tuning demo")
    parser.add_argument("data", type=Path, help="Path to CSV or h5ad file")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.data.suffix == ".csv":
        expr = load_csv(str(args.data))
    elif args.data.suffix == ".h5ad":
        expr = load_h5ad(str(args.data))
    else:
        raise ValueError("Unsupported file format")

    df = discretize(expr.data)
    train(df, epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
