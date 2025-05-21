"""Model components for scGPT fine-tuning.

This module defines a small wrapper around a Transformer model with a
prepended CLS token. The implementation here is lightweight and uses
Hugging Face's `transformers` library for demonstration purposes. In a
real setup, this would load the official scGPT weights and may include
additional heads for multi-task objectives.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


@dataclass
class ScGPTConfig:
    """Configuration for the scGPT model wrapper."""

    model_name: str = "distilbert-base-uncased"  # placeholder


class ScGPT(nn.Module):
    """Simplified scGPT-like model with a CLS token."""

    def __init__(self, config: ScGPTConfig) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.transformer = AutoModel.from_pretrained(config.model_name)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.transformer.config.hidden_size))
        # TODO: add additional heads for cell type classification, masking, etc.

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size = input_ids.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = self.transformer.embeddings(input_ids)
        embeddings = torch.cat([cls_tokens, embeddings], dim=1)
        attention_mask = torch.cat([
            torch.ones((batch_size, 1), device=attention_mask.device, dtype=attention_mask.dtype),
            attention_mask,
        ], dim=1)
        outputs = self.transformer(inputs_embeds=embeddings, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0]  # CLS representation


__all__ = ["ScGPT", "ScGPTConfig"]
