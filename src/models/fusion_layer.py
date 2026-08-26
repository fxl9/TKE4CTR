# -*- coding: utf-8 -*-
"""
Embedding fusion module for KG and text feature.
Support concat, cross attention gated, cross modal interaction fusion methods.
Dynamic text dimension and configurable multi‑head attention.
"""
import torch
import torch.nn as nn
import sys

ROOT_DIR = "./"
sys.path.insert(0, ROOT_DIR)

from configs.config import ModelConfig, SwitchConfig, EmbeddingConfig


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * torch.sigmoid(gate)


class PositionwiseFFN(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim * 2),
            GLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim),
            nn.LayerNorm(in_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.ffn(x)
        return residual + self.dropout(x)


class ConcatFusion(nn.Module):
    def __init__(self, kg_dim: int, text_dim: int):
        super().__init__()
        dropout = ModelConfig.FUSION_HYPERPARAMS["concat"]["dropout"]
        hidden_dim = ModelConfig.FUSION_HYPERPARAMS["concat"]["hidden_dim"]

        self.kg_proj = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout)
        )

        self.fusion_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        self.out_dim = hidden_dim

    def forward(self, kg_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        kg_proj = self.kg_proj(kg_emb)
        text_proj = self.text_proj(text_emb)
        fused = torch.cat([kg_proj, text_proj], dim=-1)
        fused = self.fusion_layers(fused)
        return fused


class CrossAttentionGatedFusion(nn.Module):
    def __init__(self, kg_dim: int, text_dim: int, num_heads: int = None):
        super().__init__()
        self.kg_dim = kg_dim
        self.num_heads = num_heads if num_heads is not None else ModelConfig.FUSION_HYPERPARAMS["cross_attention_gated"]["num_heads"]
        self.hidden_dim = ModelConfig.FUSION_HYPERPARAMS["cross_attention_gated"]["hidden_dim"]
        dropout = ModelConfig.FUSION_HYPERPARAMS["cross_attention_gated"]["dropout"]

        assert self.hidden_dim % self.num_heads == 0, f"hidden_dim({self.hidden_dim}) must be divisible by num_heads({self.num_heads})"

        self.kg_proj = nn.Sequential(
            nn.Linear(kg_dim, self.hidden_dim),
            Swish(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(dropout)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, self.hidden_dim),
            Swish(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(dropout)
        )

        self.cross_attn_kg2text = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=self.num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_text2kg = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=self.num_heads, dropout=dropout, batch_first=True
        )

        self.gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            Swish(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid()
        )

        self.residual_proj = nn.Sequential(
            nn.Linear(kg_dim + text_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        self.final_fusion = PositionwiseFFN(self.hidden_dim, self.hidden_dim // 2, dropout)

        self.out_dim = self.hidden_dim

    def forward(self, kg_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        kg_proj = self.kg_proj(kg_emb).unsqueeze(1)
        text_proj = self.text_proj(text_emb).unsqueeze(1)

        kg_attn, _ = self.cross_attn_kg2text(kg_proj, text_proj, text_proj)
        text_attn, _ = self.cross_attn_text2kg(text_proj, kg_proj, kg_proj)

        kg_attn = kg_attn.squeeze(1)
        text_attn = text_attn.squeeze(1)
        kg_proj = kg_proj.squeeze(1)
        text_proj = text_proj.squeeze(1)

        gate_input = torch.cat([kg_proj, kg_attn, text_proj, text_attn], dim=-1)
        gate_weight = self.gate(gate_input)

        kg_fused = gate_weight * kg_attn + (1 - gate_weight) * kg_proj
        text_fused = gate_weight * text_attn + (1 - gate_weight) * text_proj

        residual = self.residual_proj(torch.cat([kg_emb, text_emb], dim=-1))
        fused_emb = (kg_fused + text_fused) / 2 + residual
        fused_emb = self.final_fusion(fused_emb)

        return fused_emb


class CrossModalInteractionLayer(nn.Module):
    def __init__(self, kg_dim: int = 128, text_dim: int = 2560, num_heads: int = None):
        super().__init__()

        cfg = ModelConfig.FUSION_HYPERPARAMS["cross_modal_interaction"]
        self.hidden_dim = cfg["hidden_dim"]
        self.num_heads = num_heads if num_heads is not None else cfg["num_heads"]
        self.dropout = cfg["dropout"]

        assert self.hidden_dim % self.num_heads == 0, f"hidden_dim({self.hidden_dim}) must be divisible by num_heads({self.num_heads})"

        self.kg_proj = nn.Sequential(
            nn.Linear(kg_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout)
        )

        self.cross_attn_kg2text = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=self.num_heads, dropout=self.dropout, batch_first=True
        )
        self.cross_attn_text2kg = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=self.num_heads, dropout=self.dropout, batch_first=True
        )

        self.fusion = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.residual_kg = nn.Linear(kg_dim, self.hidden_dim)
        self.dropout_layer = nn.Dropout(self.dropout)

        self.out_dim = self.hidden_dim

    def forward(self, kg_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        kg_hidden = self.kg_proj(kg_emb).unsqueeze(1)
        text_hidden = self.text_proj(text_emb).unsqueeze(1)

        kg_attended, _ = self.cross_attn_kg2text(kg_hidden, text_hidden, text_hidden)
        text_attended, _ = self.cross_attn_text2kg(text_hidden, kg_hidden, kg_hidden)

        kg_attended = kg_attended.squeeze(1)
        text_attended = text_attended.squeeze(1)
        fused = torch.cat([kg_attended, text_attended], dim=-1)
        fused = self.dropout_layer(self.fusion(fused))

        residual = self.residual_kg(kg_emb)
        output = fused + residual

        return output


class EmbeddingFusion(nn.Module):
    def __init__(self, kg_dim: int = 128, text_emb_version: str = None, fusion_method: str = None, text_dim: int = None, num_heads: int = None):
        super().__init__()
        self.text_emb_version = text_emb_version or EmbeddingConfig.DEFAULT_TEXT_EMB_VERSION
        self.text_dim = text_dim if text_dim is not None else EmbeddingConfig.LLM_DIM_MAP[self.text_emb_version]
        self.fusion_method = fusion_method or ModelConfig.DEFAULT_FUSION_METHOD
        self.num_heads = num_heads

        fusion_map = {
            "concat": ConcatFusion(kg_dim, self.text_dim),
            "cross_attention_gated": CrossAttentionGatedFusion(kg_dim, self.text_dim, num_heads=self.num_heads),
            "cross_modal_interaction": CrossModalInteractionLayer(kg_dim, self.text_dim, num_heads=self.num_heads)
        }

        if self.fusion_method not in fusion_map:
            raise ValueError(f"Unsupported fusion method:{self.fusion_method}, select from {list(fusion_map.keys())}")

        self.fusion = fusion_map[self.fusion_method]
        self.out_dim = self.fusion.out_dim

        self.dropout_layers = []
        for module in self.fusion.modules():
            if isinstance(module, nn.Dropout):
                self.dropout_layers.append(module)

        head_info = self.num_heads if self.num_heads is not None else f"config_default({self.fusion.num_heads if hasattr(self.fusion, 'num_heads') else 'N/A'})"
        print(f"EmbeddingFusion init done | fusion:{self.fusion_method} | out_dim:{self.out_dim} | num_heads:{head_info} | dropout_count:{len(self.dropout_layers)}")

    def set_dropout_prob(self, prob: float):
        for layer in self.dropout_layers:
            layer.p = prob

    def forward(self, kg_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        if not SwitchConfig.use_fusion_emb:
            raise RuntimeError("use_fusion_emb flag is disabled when running fusion forward")

        kg_emb = torch.clamp(kg_emb, min=-10.0, max=10.0)
        text_emb = torch.clamp(text_emb, min=-10.0, max=10.0)

        return self.fusion(kg_emb, text_emb)



