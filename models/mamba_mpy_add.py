"""MambaAdd model for PyHealth 2.0 datasets.

This variant sums all feature embeddings first and feeds the result through a
single stack of Mamba blocks, as in the original EHRMamba paper. 
Only SequenceProcessor inputs are supported.
"""

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.models.embedding import EmbeddingModel
from pyhealth.processors import SequenceProcessor

try:
    from mambapy.mamba import MambaConfig, ResidualBlock
except Exception as exc:
    ResidualBlock = None
    MambaConfig = None
    _mambapy_import_error = exc
else:
    _mambapy_import_error = None


class MambaLayers(nn.Module):
    """Stacked mambapy residual blocks returning sequence and CLS embeddings."""

    def __init__(
        self,
        feature_size: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        if ResidualBlock is None or MambaConfig is None:
            raise ImportError(
                "mambapy is required for the Mamba model; install mambapy first."
            ) from _mambapy_import_error

        config = MambaConfig(
            d_model=feature_size,
            n_layers=num_layers,
            d_state=d_state,
            expand_factor=expand,
            d_conv=d_conv,
            use_cuda=True,
        )

        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _apply_mask(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return x
        mask = mask.to(x.device)
        while mask.dim() < x.dim():
            mask = mask.unsqueeze(-1)
        return x * mask.float()

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        for layer in self.layers:
            x = self._apply_mask(x, mask)
            x = layer(x)
            x = self.dropout(x)
            x = self._apply_mask(x, mask)
        emb = x
        cls_emb = x[:, 0, :]
        return emb, cls_emb


class MambaAdd(BaseModel):
    """Mamba model that adds feature embeddings before encoding."""

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__(dataset=dataset)
        if ResidualBlock is None or MambaConfig is None:
            raise ImportError(
                "mambapy is required for the Mamba model; install mambapy first."
            ) from _mambapy_import_error

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        assert (
            len(self.label_keys) == 1
        ), "Only one label key is supported if MambaAdd is initialized"
        self.label_key = self.label_keys[0]
        self.mode = self.dataset.output_schema[self.label_key]

        self.embedding_model = EmbeddingModel(dataset, embedding_dim)
        self.feature_processors = {
            feature_key: self.dataset.input_processors[feature_key]
            for feature_key in self.feature_keys
        }
        self._validate_processors()

        feature_size = embedding_dim
        self.input_dropout = nn.Dropout(dropout)
        self.input_layernorm = nn.LayerNorm(feature_size)
        self.mamba_layer = MambaLayers(
            feature_size=feature_size,
            num_layers=num_layers,
            dropout=dropout,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        output_size = self.get_output_size()
        self.fc = nn.Linear(feature_size, output_size)

    def _validate_processors(self) -> None:
        invalid = [
            feature_key
            for feature_key, processor in self.feature_processors.items()
            if not isinstance(processor, SequenceProcessor)
        ]
        if invalid:
            raise ValueError(
                "MambaAdd supports only SequenceProcessor inputs; "
                f"found incompatible processors for features: {invalid}"
            )

    @staticmethod
    def _split_temporal(feature: Any) -> Tuple[Optional[torch.Tensor], Any]:
        if isinstance(feature, tuple) and len(feature) == 2:
            return feature
        return None, feature

    @staticmethod
    def _ensure_tensor(value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        return torch.tensor(value, dtype=torch.long)

    @staticmethod
    def _normalize_mask(mask: torch.Tensor) -> torch.Tensor:
        if mask.dim() == 1:
            mask = mask.unsqueeze(1)
        while mask.dim() > 2:
            mask = mask.any(dim=-1)
        mask = mask.bool()
        invalid_rows = ~mask.any(dim=1)
        if invalid_rows.any():
            mask[invalid_rows, 0] = True
        return mask

    def _create_mask(self, value: torch.Tensor) -> torch.Tensor:
        mask = value != 0
        return self._normalize_mask(mask)

    def _mask_from_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.any(torch.abs(x) > 0, dim=-1)
        return self._normalize_mask(mask)

    @staticmethod
    def _ensure_sequence_embedding(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError(
                "Expected embedding tensor with 3 dimensions (batch, seq_len, embed_dim); "
                f"got shape {tuple(x.shape)}"
            )
        return x

    def _add_embeddings(self, embedded: Dict[str, torch.Tensor]) -> torch.Tensor:
        sequences = []
        seq_lens = set()
        for feature_key in self.feature_keys:
            x = self._ensure_sequence_embedding(embedded[feature_key])
            sequences.append(x)
            seq_lens.add(x.size(1))
        max_len = max(seq_lens)
        padded = [self._pad_sequence(seq, max_len) for seq in sequences]
        stacked = torch.stack(padded, dim=0)
        return stacked.sum(dim=0)

    def _merge_masks(self, masks: Dict[str, torch.Tensor]) -> torch.Tensor:
        normalized_masks = [
            self._normalize_mask(masks[feature_key])
            for feature_key in self.feature_keys
        ]
        max_len = max(mask.size(1) for mask in normalized_masks)
        padded_masks = [self._pad_mask(mask, max_len) for mask in normalized_masks]
        stacked = torch.stack(padded_masks, dim=-1)
        merged = stacked.any(dim=-1)
        return self._normalize_mask(merged)

    @staticmethod
    def _pad_sequence(x: torch.Tensor, target_len: int) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len == target_len:
            return x
        if seq_len > target_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds target length {target_len}."
            )
        pad_len = target_len - seq_len
        return F.pad(x, (0, 0, 0, pad_len, 0, 0), value=0.0)

    @staticmethod
    def _pad_mask(mask: torch.Tensor, target_len: int) -> torch.Tensor:
        seq_len = mask.size(1)
        if seq_len == target_len:
            return mask
        if seq_len > target_len:
            raise ValueError(
                f"Mask length {seq_len} exceeds target length {target_len}."
            )
        pad_len = target_len - seq_len
        return F.pad(mask, (0, pad_len), value=False)

    def _apply_input_masking(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = mask.to(self.device)
        mask_f = mask.unsqueeze(-1).float()
        x = x * mask_f
        x = self.input_dropout(x)
        x = self.input_layernorm(x)
        x = x * mask_f
        return x, mask

    def forward_from_embedding(
        self,
        feature_embeddings: Dict[str, torch.Tensor],
        time_info: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        del time_info

        sequence_embeddings = {
            feature_key: self._ensure_sequence_embedding(
                feature_embeddings[feature_key].to(self.device)
            )
            for feature_key in self.feature_keys
        }
        masks = {
            feature_key: self._mask_from_embeddings(sequence_embeddings[feature_key])
            for feature_key in self.feature_keys
        }

        merged_mask = self._merge_masks(masks).to(self.device)
        added_emb = self._add_embeddings(sequence_embeddings).to(self.device)
        added_emb, merged_mask = self._apply_input_masking(added_emb, merged_mask)
        _, cls_emb = self.mamba_layer(added_emb, merged_mask)
        logits = self.fc(cls_emb)

        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        results = {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}
        if kwargs.get("embed", False):
            results["embed"] = cls_emb
        return results

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        embedding_inputs: Dict[str, torch.Tensor] = {}
        masks: Dict[str, torch.Tensor] = {}

        for feature_key in self.feature_keys:
            _, value = self._split_temporal(kwargs[feature_key])
            value_tensor = self._ensure_tensor(value)
            embedding_inputs[feature_key] = value_tensor
            masks[feature_key] = self._create_mask(value_tensor)

        embedded = self.embedding_model(embedding_inputs)
        merged_mask = self._merge_masks(masks).to(self.device)

        added_emb = self._add_embeddings(embedded).to(self.device)
        added_emb, merged_mask = self._apply_input_masking(added_emb, merged_mask)
        _, cls_emb = self.mamba_layer(added_emb, merged_mask)

        logits = self.fc(cls_emb)
        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        results = {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}
        if kwargs.get("embed", False):
            results["embed"] = cls_emb
        return results
