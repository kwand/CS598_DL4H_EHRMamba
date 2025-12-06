"""Jamba model for PyHealth 2.0 datasets.

This implementation mirrors the Transformer model structure but swaps the
self-attention blocks for Jamba hybrid (Mamba + attention + optional MoE)
blocks. PyHealth's shared :class:`EmbeddingModel` is used for all feature
streams so the input contract matches :mod:`pyhealth.models.transformer`.
"""

from typing import Any, Dict, Optional, Tuple, List

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
    from mambapy.jamba import Jamba as MambaPyJamba  # type: ignore
    from mambapy.jamba import JambaLMConfig, load_balancing_loss  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    MambaPyJamba = None
    JambaLMConfig = None
    load_balancing_loss = None
    _mambapy_import_error = exc
else:
    _mambapy_import_error = None


class JambaLayers(nn.Module):
    """Stacked mambapy Jamba layers returning sequence and CLS embeddings."""

    def __init__(
        self,
        feature_size: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        mlp_size: Optional[int] = None,
        num_attention_heads: Optional[int] = None,
        num_key_value_heads: Optional[int] = None,
        attention_dropout: float = 0.0,
        num_experts: int = 1,
        num_experts_per_tok: int = 1,
        attn_layer_offset: int = 4,
        attn_layer_period: int = 8,
        expert_layer_offset: int = 1,
        expert_layer_period: int = 2,
        rms_norm_eps: float = 1e-5,
        use_cuda: Optional[bool] = None,
        pscan: bool = True,
    ) -> None:
        super().__init__()
        if MambaPyJamba is None or JambaLMConfig is None:
            raise ImportError(
                "mambapy is required for the Jamba model; install mambapy first."
            ) from _mambapy_import_error

        self.feature_size = feature_size
        self.num_attention_heads = self._infer_heads(feature_size, num_attention_heads)
        self.num_key_value_heads = (
            num_key_value_heads
            if num_key_value_heads is not None
            else max(1, self.num_attention_heads // 4)
        )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )

        config = JambaLMConfig(
            d_model=feature_size,
            n_layers=num_layers,
            mlp_size=mlp_size if mlp_size is not None else feature_size * expand,
            d_state=d_state,
            expand_factor=expand,
            d_conv=d_conv,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            attention_dropout=attention_dropout,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            attn_layer_offset=attn_layer_offset,
            attn_layer_period=attn_layer_period,
            expert_layer_offset=expert_layer_offset,
            expert_layer_period=expert_layer_period,
            rms_norm_eps=rms_norm_eps,
            use_cuda=torch.cuda.is_available() if use_cuda is None else use_cuda,
            pscan=pscan,
        )

        self.model = MambaPyJamba(config)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _infer_heads(feature_size: int, desired: Optional[int]) -> int:
        if desired is not None:
            if feature_size % desired != 0:
                raise ValueError(
                    "feature_size must be divisible by num_attention_heads"
                )
            return desired
        for candidate in range(min(8, feature_size), 0, -1):
            if feature_size % candidate == 0:
                return candidate
        return 1

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
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        x = self._apply_mask(x, mask)
        x, router_logits = self.model(x)
        x = self.dropout(x)
        x = self._apply_mask(x, mask)
        emb = x
        cls_emb = x[:, 0, :]
        return emb, cls_emb, router_logits


class Jamba(BaseModel):
    """Jamba encoder for PyHealth datasets using shared embeddings.

    Each feature stream is embedded with :class:`EmbeddingModel` and encoded by
    a stack of mambapy Jamba layers. The [CLS]-style embedding from each stream
    is concatenated and passed through a linear classification head.
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
        mlp_size: Optional[int] = None,
        num_attention_heads: Optional[int] = None,
        num_key_value_heads: Optional[int] = None,
        attention_dropout: float = 0.0,
        num_experts: int = 1,
        num_experts_per_tok: int = 1,
        attn_layer_offset: int = 4,
        attn_layer_period: int = 8,
        expert_layer_offset: int = 1,
        expert_layer_period: int = 2,
        rms_norm_eps: float = 1e-5,
        moe_aux_loss_weight: float = 0.0,
        use_cuda: Optional[bool] = None,
        pscan: bool = True,
    ) -> None:
        super().__init__(dataset=dataset)
        if MambaPyJamba is None or JambaLMConfig is None:
            raise ImportError(
                "mambapy is required for the Jamba model; install mambapy first."
            ) from _mambapy_import_error

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.mlp_size = mlp_size if mlp_size is not None else embedding_dim * expand
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.attention_dropout = attention_dropout
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.attn_layer_offset = attn_layer_offset
        self.attn_layer_period = attn_layer_period
        self.expert_layer_offset = expert_layer_offset
        self.expert_layer_period = expert_layer_period
        self.rms_norm_eps = rms_norm_eps
        self.moe_aux_loss_weight = moe_aux_loss_weight
        self.use_cuda = use_cuda
        self.pscan = pscan

        assert (
            len(self.label_keys) == 1
        ), "Only one label key is supported if Jamba is initialized"
        self.label_key = self.label_keys[0]
        self.mode = self.dataset.output_schema[self.label_key]

        self.embedding_model = EmbeddingModel(dataset, embedding_dim)
        self.feature_processors = {
            feature_key: self.dataset.input_processors[feature_key]
            for feature_key in self.feature_keys
        }

        self.jamba_layers = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.jamba_layers[feature_key] = JambaLayers(
                feature_size=embedding_dim,
                num_layers=num_layers,
                dropout=dropout,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                mlp_size=self.mlp_size,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
                attention_dropout=attention_dropout,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                attn_layer_offset=attn_layer_offset,
                attn_layer_period=attn_layer_period,
                expert_layer_offset=expert_layer_offset,
                expert_layer_period=expert_layer_period,
                rms_norm_eps=rms_norm_eps,
                use_cuda=use_cuda,
                pscan=pscan,
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

    def _maybe_add_moe_aux_loss(
        self, router_logits: List[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if (
            self.moe_aux_loss_weight <= 0
            or load_balancing_loss is None
            or self.num_experts <= 1
            or not router_logits
        ):
            return None
        return load_balancing_loss(
            router_logits, self.num_experts, self.num_experts_per_tok
        )

    def forward_from_embedding(
        self,
        feature_embeddings: Dict[str, torch.Tensor],
        time_info: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        del time_info
        patient_emb = []
        router_logits: List[torch.Tensor] = []

        for feature_key in self.feature_keys:
            x = feature_embeddings[feature_key].to(self.device)
            x = self._pool_embedding(x)
            mask = self._mask_from_embeddings(x).to(self.device)
            _, cls_emb, layer_router_logits = self.jamba_layers[feature_key](x, mask)
            patient_emb.append(cls_emb)
            router_logits.extend(layer_router_logits)

        patient_emb = torch.cat(patient_emb, dim=1)
        logits = self.fc(patient_emb)

        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        moe_aux_loss = self._maybe_add_moe_aux_loss(router_logits)
        if moe_aux_loss is not None:
            loss = loss + self.moe_aux_loss_weight * moe_aux_loss

        y_prob = self.prepare_y_prob(logits)
        results = {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}
        if moe_aux_loss is not None:
            results["moe_aux_loss"] = moe_aux_loss
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        return results

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        patient_emb = []
        embedding_inputs: Dict[str, torch.Tensor] = {}
        masks: Dict[str, torch.Tensor] = {}
        router_logits: List[torch.Tensor] = []

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
            _, cls_emb, layer_router_logits = self.jamba_layers[feature_key](x, mask)
            patient_emb.append(cls_emb)
            router_logits.extend(layer_router_logits)

        patient_emb = torch.cat(patient_emb, dim=1)
        logits = self.fc(patient_emb)
        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)

        moe_aux_loss = self._maybe_add_moe_aux_loss(router_logits)
        if moe_aux_loss is not None:
            loss = loss + self.moe_aux_loss_weight * moe_aux_loss

        y_prob = self.prepare_y_prob(logits)
        results = {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}
        if moe_aux_loss is not None:
            results["moe_aux_loss"] = moe_aux_loss
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        return results
