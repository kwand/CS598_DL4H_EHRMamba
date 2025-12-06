"""Mamba model for PyHealth 2.0 datasets.

This implementation mirrors the Transformer model structure but swaps the
self-attention blocks for Mamba state-space mixers. PyHealth's shared
:class:`EmbeddingModel` is used for all feature streams so the input contract
matches :mod:`pyhealth.models.transformer`.
"""

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.models.embedding import EmbeddingModel
from pyhealth.processors import (
    MultiHotProcessor,
    SequenceProcessor,
    StageNetProcessor,
    StageNetTensorProcessor,
    TensorProcessor,
    TimeseriesProcessor,
)

try:
    from mamba_ssm.modules.mamba_simple import Mamba as MambaBlock 
except Exception as exc:
    MambaBlock = None
    _mamba_import_error = exc
else:
    _mamba_import_error = None


class MambaEncoderBlock(nn.Module):
    """Pre-norm residual block wrapping a Mamba mixer."""

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.5,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        version: str = "v1"
    ) -> None:
        super().__init__()
        if MambaBlock is None:
            raise ImportError(
                "mamba-ssm is required for the Mamba model; install mamba-ssm first."
            ) from _mamba_import_error

        self.norm = nn.LayerNorm(hidden_size)
        self.mixer = MambaBlock(
            d_model=hidden_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is not None:
            mask = mask.to(x.device).unsqueeze(-1).float()
            x = x * mask
        residual = x
        mixed = self.mixer(self.norm(x))
        mixed = self.dropout(mixed)
        if mask is not None:
            mixed = mixed * mask
        return residual + mixed


class MambaLayer(nn.Module):
    """Stacked Mamba encoder blocks returning sequence and CLS embeddings."""

    def __init__(
        self,
        feature_size: int,
        num_layers: int = 1,
        dropout: float = 0.5,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MambaEncoderBlock(
                    hidden_size=feature_size,
                    dropout=dropout,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        for layer in self.layers:
            x = layer(x, mask=mask)
        emb = x
        cls_emb = x[:, 0, :]
        return emb, cls_emb


class Mamba(BaseModel):
    """Mamba model for PyHealth datasets using shared embeddings.

    Each feature stream is embedded with :class:`EmbeddingModel` and encoded by
    a stack of Mamba blocks. The [CLS]-style embedding from each stream is
    concatenated and passed through a linear classification head.
    """

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
        if MambaBlock is None:
            raise ImportError(
                "mamba-ssm is required for the Mamba model; install mamba-ssm first."
            ) from _mamba_import_error

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        assert (
            len(self.label_keys) == 1
        ), "Only one label key is supported if Mamba is initialized"
        self.label_key = self.label_keys[0]
        self.mode = self.dataset.output_schema[self.label_key]

        self.embedding_model = EmbeddingModel(dataset, embedding_dim)
        self.feature_processors = {
            feature_key: self.dataset.input_processors[feature_key]
            for feature_key in self.feature_keys
        }

        self.mamba_layers = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.mamba_layers[feature_key] = MambaLayer(
                feature_size=embedding_dim,
                num_layers=num_layers,
                dropout=dropout,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )

        output_size = self.get_output_size()
        self.fc = nn.Linear(len(self.feature_keys) * embedding_dim, output_size)

    @staticmethod
    def _split_temporal(feature: Any) -> Tuple[Optional[torch.Tensor], Any]:
        if isinstance(feature, tuple) and len(feature) == 2:
            return feature
        return None, feature

    def _ensure_tensor(self, feature_key: str, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        processor = self.feature_processors[feature_key]
        if isinstance(processor, (SequenceProcessor, StageNetProcessor)):
            return torch.tensor(value, dtype=torch.long)
        return torch.tensor(value, dtype=torch.float)

    def _create_mask(self, feature_key: str, value: torch.Tensor) -> torch.Tensor:
        processor = self.feature_processors[feature_key]
        if isinstance(processor, SequenceProcessor):
            mask = value != 0
        elif isinstance(processor, StageNetProcessor):
            if value.dim() >= 3:
                mask = torch.any(value != 0, dim=-1)
            else:
                mask = value != 0
        elif isinstance(processor, (TimeseriesProcessor, StageNetTensorProcessor)):
            if value.dim() >= 3:
                mask = torch.any(torch.abs(value) > 0, dim=-1)
            elif value.dim() == 2:
                mask = torch.any(torch.abs(value) > 0, dim=-1, keepdim=True)
            else:
                mask = torch.ones(
                    value.size(0),
                    1,
                    dtype=torch.bool,
                    device=value.device,
                )
        elif isinstance(processor, (TensorProcessor, MultiHotProcessor)):
            mask = torch.ones(
                value.size(0),
                1,
                dtype=torch.bool,
                device=value.device,
            )
        else:
            if value.dim() >= 2:
                mask = torch.any(value != 0, dim=-1)
            else:
                mask = torch.ones(
                    value.size(0),
                    1,
                    dtype=torch.bool,
                    device=value.device,
                )

        if mask.dim() == 1:
            mask = mask.unsqueeze(1)
        mask = mask.bool()
        if mask.dim() == 2:
            invalid_rows = ~mask.any(dim=1)
            if invalid_rows.any():
                mask[invalid_rows, 0] = True
        return mask

    @staticmethod
    def _pool_embedding(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.sum(dim=2)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return x

    @staticmethod
    def _mask_from_embeddings(x: torch.Tensor) -> torch.Tensor:
        mask = torch.any(torch.abs(x) > 0, dim=-1)
        if mask.dim() == 1:
            mask = mask.unsqueeze(1)
        invalid_rows = ~mask.any(dim=1)
        if invalid_rows.any():
            mask[invalid_rows, 0] = True
        return mask.bool()

    def forward_from_embedding(
        self,
        feature_embeddings: Dict[str, torch.Tensor],
        time_info: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        del time_info
        patient_emb = []

        for feature_key in self.feature_keys:
            x = feature_embeddings[feature_key].to(self.device)
            x = self._pool_embedding(x)
            mask = self._mask_from_embeddings(x).to(self.device)
            _, cls_emb = self.mamba_layers[feature_key](x, mask)
            patient_emb.append(cls_emb)

        patient_emb = torch.cat(patient_emb, dim=1)
        logits = self.fc(patient_emb)

        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        results = {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        return results

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        patient_emb = []
        embedding_inputs: Dict[str, torch.Tensor] = {}
        masks: Dict[str, torch.Tensor] = {}

        for feature_key in self.feature_keys:
            _, value = self._split_temporal(kwargs[feature_key])
            value_tensor = self._ensure_tensor(feature_key, value)
            embedding_inputs[feature_key] = value_tensor
            masks[feature_key] = self._create_mask(feature_key, value_tensor)

        embedded = self.embedding_model(embedding_inputs)

        for feature_key in self.feature_keys:
            x = embedded[feature_key]
            mask = masks[feature_key].to(self.device)
            x = self._pool_embedding(x)
            _, cls_emb = self.mamba_layers[feature_key](x, mask)
            patient_emb.append(cls_emb)

        patient_emb = torch.cat(patient_emb, dim=1)
        logits = self.fc(patient_emb)
        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        results = {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        return results
