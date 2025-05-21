"""Data loading utilities for scGPT fine-tuning.

This module provides functions to load single-cell expression matrices
from either CSV or h5ad formats. It also includes helpers for basic
discretization, masking, and attaching batch labels.

Note: Advanced preprocessing like automated thresholding or complex
batch effect correction is out of scope for this skeleton and can be
added later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

try:
    import anndata as ad
except ImportError:  # pragma: no cover - optional dependency
    ad = None


@dataclass
class ExpressionData:
    """Container for expression matrix and optional batch labels."""

    data: pd.DataFrame
    batch: Optional[pd.Series] = None


def load_csv(path: str) -> ExpressionData:
    """Load expression data from a CSV file."""
    df = pd.read_csv(path, index_col=0)
    return ExpressionData(df)


def load_h5ad(path: str) -> ExpressionData:
    """Load expression data from an h5ad file if anndata is installed."""
    if ad is None:
        raise ImportError("anndata is required to load h5ad files")
    adata = ad.read_h5ad(path)
    return ExpressionData(adata.to_df(), batch=adata.obs.get("batch"))


def discretize(data: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Discretize expression values into bins using pandas cut."""
    return data.apply(lambda col: pd.cut(col, bins=bins, labels=False))


__all__ = [
    "ExpressionData",
    "load_csv",
    "load_h5ad",
    "discretize",
]
