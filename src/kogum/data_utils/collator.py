"""Data collators for KoGum pre-training and fine-tuning."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollatorForPackedSequences:
    """Data collator for packed sequences (no padding needed).

    Used when sequences are pre-packed to fixed length.
    Simply stacks sequences into a batch.
    """

    tokenizer: PreTrainedTokenizerBase
    return_tensors: str = "pt"

    def __call__(self, instances: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Stack input_ids (all should be same length after packing)
        input_ids = torch.stack(
            [
                torch.tensor(instance["input_ids"], dtype=torch.long)
                if not isinstance(instance["input_ids"], torch.Tensor)
                else instance["input_ids"]
                for instance in instances
            ]
        )

        # Labels are same as input_ids for causal LM
        labels = input_ids.clone()

        # For packed sequences, we don't mask anything
        # (padding was removed during packing)

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


@dataclass
class DataCollatorWithDocumentBoundaries:
    """Data collator that tracks document boundaries for intra-document masking.

    Used with packed sequences where multiple documents are concatenated.
    Tracks boundaries to prevent attention across documents.
    """

    tokenizer: PreTrainedTokenizerBase
    return_tensors: str = "pt"

    def __call__(self, instances: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Stack input_ids
        input_ids = torch.stack(
            [
                torch.tensor(instance["input_ids"], dtype=torch.long)
                if not isinstance(instance["input_ids"], torch.Tensor)
                else instance["input_ids"]
                for instance in instances
            ]
        )

        # Labels
        labels = input_ids.clone()

        # Document boundaries (if provided in instances)
        # Each instance should have "document_ids" indicating which document each token belongs to
        if "document_ids" in instances[0]:
            document_ids = torch.stack(
                [
                    torch.tensor(instance["document_ids"], dtype=torch.long)
                    if not isinstance(instance["document_ids"], torch.Tensor)
                    else instance["document_ids"]
                    for instance in instances
                ]
            )
            return {
                "input_ids": input_ids,
                "labels": labels,
                "document_ids": document_ids,
            }

        return {
            "input_ids": input_ids,
            "labels": labels,
        }
