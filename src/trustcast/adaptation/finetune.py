"""Domain adaptation (Program 2): does fine-tuning pay off?

Every study compares three arms on the same back-test:
  1. zero-shot foundation model
  2. the same model fine-tuned (e.g. LoRA) on domain data
  3. classical baselines
and reports the gain together with the data and compute it cost.

Fine-tuned weights go to the lab's Hugging Face organization, only where the
base model's license allows redistribution.
"""

from __future__ import annotations


def lora_finetune(model_name: str, train_series, config: dict):
    """Placeholder: implement per model family (Chronos and TimesFM first)."""
    raise NotImplementedError
