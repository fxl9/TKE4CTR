"""
CTR predictor module for Text2KG CTR task.
Support multiple predictor heads and embedding fusion with configurable multi‑head attention.
Relative project path configuration.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append("./")

from configs.config import ModelConfig, SwitchConfig, EmbeddingConfig
from src.models.fusion_layer import EmbeddingFusion


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class FeatureGating(nn.Module):
    def __init__(self, in_dim, reduction=4):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_dim, in_dim // reduction),
            Swish(),
            nn.Linear(in_dim // reduction, in_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        global_avg = x.mean(dim=0, keepdim=True)
        weight = self.gate(global_avg)
        return x * weight


class SimpleCrossLayer(nn.Module):
    def __init__(self, in_dim, dropout=0.1):
        super().__init__()
        self.cross_proj = nn.Linear(in_dim, in_dim)
        self.norm = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.cross_proj(x)
        x = residual * x
        return self.norm(residual + self.dropout(x))


class MLPFinalPredictor(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        cfg = ModelConfig.PREDICTOR_HYPERPARAMS["mlp_final"]
        hidden_dims = cfg["hidden_dims"]
        dropout = cfg["dropout"]
        activation = cfg["activation"]
        reduction = cfg.get("reduction", 4)

        self.feature_gating = FeatureGating(in_dim, reduction=reduction)

        layers = []
        prev_dim = in_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU() if activation == "relu" else Swish())
            layers.append(nn.LayerNorm(dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = dim

        self.feature_proj = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_gating(x)
        features = self.feature_proj(x)
        return self.output(features).squeeze(-1)


class DCNv3FMPredictor(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        cfg = ModelConfig.PREDICTOR_HYPERPARAMS["dcnv3_fm"]
        self.cross_layers = cfg["num_cross_layers"]
        self.hidden_dim = cfg["hidden_dim"]
        self.embedding_dim = cfg["embedding_dim"]
        dropout = cfg["dropout"]

        self.dcnv3_input_proj = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            Swish(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(dropout)
        )
        self.cross_blocks = nn.ModuleList([
            SimpleCrossLayer(self.hidden_dim, dropout=dropout)
            for _ in range(self.cross_layers)
        ])
        self.dcnv3_norm = nn.LayerNorm(self.hidden_dim)

        self.fm_in_dim = min(in_dim, 256)
        self.fm_proj = nn.Sequential(
            nn.Linear(in_dim, self.fm_in_dim),
            Swish(),
            nn.LayerNorm(self.fm_in_dim),
            nn.Dropout(dropout)
        )
        self.fm_first_order = nn.Linear(self.fm_in_dim, 1)
        self.fm_emb = nn.Parameter(torch.randn(self.fm_in_dim, self.embedding_dim))
        nn.init.xavier_uniform_(self.fm_emb)

        self.modal_gate = nn.Sequential(
            nn.Linear(self.fm_in_dim, self.fm_in_dim),
            nn.Sigmoid()
        )

        self.fusion_proj = nn.Linear(self.hidden_dim + 1, self.hidden_dim)
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dcnv3_x = self.dcnv3_input_proj(x)
        for block in self.cross_blocks:
            dcnv3_x = block(dcnv3_x)
        dcnv3_x = self.dcnv3_norm(dcnv3_x)

        fm_x = self.fm_proj(x)
        gate = self.modal_gate(fm_x)
        fm_x = fm_x * gate

        fm_emb = self.fm_emb.unsqueeze(0).repeat(fm_x.shape[0], 1, 1)
        vx = fm_emb * fm_x.unsqueeze(-1)

        sum_square = torch.sum(vx, dim=1) ** 2
        square_sum = torch.sum(vx ** 2, dim=1)
        fm_second = 0.5 * torch.sum(sum_square - square_sum, dim=1, keepdim=True)
        fm_out = self.fm_first_order(fm_x) + fm_second

        fusion_input = torch.cat([dcnv3_x, fm_out], dim=-1)
        fusion_x = self.fusion_proj(fusion_input)
        fusion_x = Swish()(fusion_x)
        fusion_x = self.fusion_norm(fusion_x)

        return self.output(fusion_x).squeeze(-1)


class CrossModalFMPredictor(nn.Module):
    def __init__(self, in_dim: int, kg_dim: int = 128, is_fusion_mode: bool = True):
        super().__init__()
        cfg = ModelConfig.PREDICTOR_HYPERPARAMS["cross_modal_fm"]
        self.embedding_dim = cfg["embedding_dim"]
        dropout = cfg["dropout"]
        self.kg_dim = kg_dim
        self.is_fusion_mode = is_fusion_mode

        if self.is_fusion_mode:
            self.text_dim = in_dim - self.kg_dim
            if self.text_dim <= 0:
                self.text_dim = max(64, self.kg_dim // 2)
                self.kg_dim = in_dim - self.text_dim
        else:
            self.text_dim = in_dim
            self.kg_dim = in_dim

        self.kg_proj = nn.Linear(in_dim, self.kg_dim)
        self.text_proj = nn.Linear(in_dim, self.text_dim)

        self.kg_emb = nn.Parameter(torch.randn(self.kg_dim, self.embedding_dim))
        self.text_emb = nn.Parameter(torch.randn(self.text_dim, self.embedding_dim))
        nn.init.xavier_uniform_(self.kg_emb)
        nn.init.xavier_uniform_(self.text_emb)

        self.first_order = nn.Linear(in_dim, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kg_x = self.kg_proj(x)
        text_x = self.text_proj(x)

        first_out = self.first_order(x)

        kg_vx = self.kg_emb.unsqueeze(0).repeat(kg_x.shape[0], 1, 1) * kg_x.unsqueeze(-1)
        text_vx = self.text_emb.unsqueeze(0).repeat(text_x.shape[0], 1, 1) * text_x.unsqueeze(-1)

        min_seq_len = min(kg_vx.shape[1], text_vx.shape[1])
        kg_vx = kg_vx[:, :min_seq_len, :]
        text_vx = text_vx[:, :min_seq_len, :]

        cross_interaction = torch.sum(kg_vx * text_vx, dim=[1, 2])
        cross_interaction = cross_interaction.unsqueeze(-1)

        out = first_out + self.dropout(cross_interaction)
        return out.squeeze(-1)


class CTRPredictor(nn.Module):
    def __init__(self, kg_dim: int = 128, text_emb_version: str = None, fusion_method: str = None,
                 predictor_type: str = None, text_dim: int = None, is_fusion_mode: bool = True, num_heads: int = 4):
        super().__init__()
        self.kg_dim = kg_dim
        self.text_emb_version = text_emb_version if (text_emb_version is not None or is_fusion_mode) else None
        self.fusion_method = fusion_method or ModelConfig.DEFAULT_FUSION_METHOD
        self.predictor_type = predictor_type or ModelConfig.DEFAULT_PREDICTOR
        self.text_dim = text_dim
        self.is_fusion_mode = is_fusion_mode
        self.num_heads = num_heads

        if self.is_fusion_mode and SwitchConfig.use_fusion_emb:
            self.fusion_layer = EmbeddingFusion(
                kg_dim=kg_dim,
                text_emb_version=self.text_emb_version,
                fusion_method=self.fusion_method,
                text_dim=self.text_dim,
                num_heads=self.num_heads
            )
            self.in_dim = self.fusion_layer.out_dim
            print(f"CTRPredictor | fusion:{self.fusion_method} | num_heads:{self.num_heads} | predictor_in_dim:{self.in_dim}")
        else:
            if SwitchConfig.use_text_emb and not SwitchConfig.use_kg_emb:
                self.in_dim = self.text_dim or EmbeddingConfig.LLM_DIM_MAP.get(self.text_emb_version, 768)
            elif SwitchConfig.use_kg_emb and not SwitchConfig.use_text_emb:
                self.in_dim = self.kg_dim
            else:
                raise ValueError("must enable only kg or text embedding in non fusion mode")
            self.fusion_layer = None
            print(f"CTRPredictor | mode:{'kg_only' if SwitchConfig.use_kg_emb else 'text_only'} | predictor_in_dim:{self.in_dim}")

        predictor_map = {
            "mlp_final": MLPFinalPredictor(self.in_dim),
            "dcnv3_fm": DCNv3FMPredictor(self.in_dim),
            "cross_modal_fm": CrossModalFMPredictor(self.in_dim, kg_dim=kg_dim, is_fusion_mode=self.is_fusion_mode)
        }

        if self.predictor_type not in predictor_map:
            raise ValueError(f"unsupported predictor type:{self.predictor_type}, select from {list(predictor_map.keys())}")

        self.predictor = predictor_map[self.predictor_type]

    def forward(self, kg_emb: torch.Tensor = None, text_emb: torch.Tensor = None) -> torch.Tensor:
        if self.is_fusion_mode and SwitchConfig.use_fusion_emb:
            if kg_emb is None or text_emb is None:
                raise ValueError("kg_emb and text_emb should not be none in fusion mode")
            x = self.fusion_layer(kg_emb, text_emb)
        else:
            if SwitchConfig.use_kg_emb and not SwitchConfig.use_text_emb:
                x = kg_emb
                if x is None:
                    raise ValueError("kg_emb should not be none in kg only mode")
            elif SwitchConfig.use_text_emb and not SwitchConfig.use_kg_emb:
                x = text_emb
                if x is None:
                    raise ValueError("text_emb should not be none in text only mode")
            else:
                raise ValueError("must enable only kg or text embedding in non fusion mode")

        logits = self.predictor(x)
        logits = torch.clamp(logits, min=-5.0, max=5.0)
        return logits
