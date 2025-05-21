"""Training utilities for scGPT fine-tuning.

This is a minimal training loop skeleton that demonstrates how the
components defined in this repository can be assembled. It lacks many of
the advanced features described in the project plan (multi-task losses,
rejection mechanism, gradient harmonization, etc.), but provides a
starting point for experimentation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from .model import ScGPT, ScGPTConfig


class ExpressionDataset(Dataset):
    """Dataset wrapper around a pandas DataFrame."""

    def __init__(self, df):
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        return torch.tensor(row.values, dtype=torch.float)


def train(
    df,
    *,
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str | torch.device = "cpu",
) -> ScGPT:
    dataset = ExpressionDataset(df)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = ScGPT(ScGPTConfig())
    model.to(device)

    criterion = nn.MSELoss()  # placeholder
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        for batch in loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            # In a real scenario we'd tokenize input and compute multiple losses.
            outputs = model(batch.long(), torch.ones_like(batch, dtype=torch.long))
            loss = criterion(outputs, torch.zeros_like(outputs))
            loss.backward()
            optimizer.step()
    return model


__all__ = ["train"]
