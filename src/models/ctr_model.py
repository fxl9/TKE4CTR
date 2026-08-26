"""
Main CTR model wrapper for Text2KG CTR task.
Integrate embedding reduction, predictor module, loss calculation, save and load utilities.
Support fusion mode, kg‑only mode and text‑only mode with configurable multi‑head attention.
"""
import torch
import torch.nn as nn
import torch.nn.init as init
import sys
import numpy as np

sys.path.insert(0, "./")

from configs.config import ModelConfig, EmbeddingConfig, SwitchConfig, TrainConfig, GPUConfig
from src.models.predictor import CTRPredictor


class CTRModel(nn.Module):
    def __init__(self, kg_dim: int = 128, text_emb_version: str = None, fusion_method: str = None,
                 predictor_type: str = None, num_heads: int = 4):
        super().__init__()
        self.kg_dim = kg_dim
        self.text_emb_version = text_emb_version if text_emb_version is not None else (
            None if fusion_method == "none" else EmbeddingConfig.DEFAULT_TEXT_EMB_VERSION
        )
        self.fusion_method = fusion_method or ModelConfig.DEFAULT_FUSION_METHOD
        self.predictor_type = predictor_type or ModelConfig.DEFAULT_PREDICTOR
        self.num_heads = num_heads

        self.is_fusion_mode = self.fusion_method != "none" and self.fusion_method in ModelConfig.AVAILABLE_FUSION_METHODS
        self.effective_fusion_method = self.fusion_method if self.is_fusion_mode else None

        self.is_text_mode = self.is_fusion_mode is False and self.text_emb_version is not None
        self.is_kg_mode = self.is_fusion_mode is False and self.text_emb_version is None

        if self.is_fusion_mode:
            SwitchConfig.use_kg_emb = True
            SwitchConfig.use_text_emb = True
            SwitchConfig.use_fusion_emb = True
        elif self.is_text_mode:
            SwitchConfig.use_kg_emb = False
            SwitchConfig.use_text_emb = True
            SwitchConfig.use_fusion_emb = False
        elif self.is_kg_mode:
            SwitchConfig.use_kg_emb = True
            SwitchConfig.use_text_emb = False
            SwitchConfig.use_fusion_emb = False

        self.original_text_dim = EmbeddingConfig.LLM_DIM_MAP.get(self.text_emb_version,
                                                                 768) if self.text_emb_version else self.kg_dim
        self.reduced_text_dim = self.original_text_dim // 2 if self.original_text_dim > 512 else self.original_text_dim
        self.text_reduce = nn.Linear(self.original_text_dim,
                                     self.reduced_text_dim) if self.original_text_dim > 512 else nn.Identity()

        if self.is_fusion_mode:
            supported_fusions = ModelConfig.AVAILABLE_FUSION_METHODS
            if self.fusion_method not in supported_fusions:
                raise ValueError(f"unsupported fusion method:{self.fusion_method}, select from {supported_fusions}")

        supported_predictors = ModelConfig.AVAILABLE_PREDICTORS
        if self.predictor_type not in supported_predictors:
            raise ValueError(f"unsupported predictor type:{self.predictor_type}, select from {supported_predictors}")

        self.fusion_hyperparams = ModelConfig.FUSION_HYPERPARAMS.get(self.fusion_method,
                                                                     {}) if self.is_fusion_mode else {}
        if self.is_fusion_mode and "num_heads" in self.fusion_hyperparams:
            self.fusion_hyperparams["num_heads"] = self.num_heads
            print(f"fusion num_heads updated: config {ModelConfig.FUSION_HYPERPARAMS[self.fusion_method]['num_heads']} -> custom {self.num_heads}")

        self.predictor_hyperparams = ModelConfig.PREDICTOR_HYPERPARAMS.get(self.predictor_type, {})

        self.predictor_input_dim = self.reduced_text_dim if self.is_text_mode else (
            self.kg_dim if self.is_kg_mode else ModelConfig.FUSION_OUTPUT_DIM
        )

        self.predictor = CTRPredictor(
            kg_dim=kg_dim,
            text_emb_version=self.text_emb_version,
            fusion_method=self.effective_fusion_method,
            predictor_type=self.predictor_type,
            text_dim=self.reduced_text_dim,
            is_fusion_mode=self.is_fusion_mode,
            num_heads=self.num_heads
        )

        self._print_fusion_config()
        self._init_weights()
        self._init_loss_fn()

    def _print_fusion_config(self):
        try:
            if self.is_fusion_mode:
                if hasattr(self.predictor, 'fusion_layer'):
                    fusion_layer = self.predictor.fusion_layer
                    fusion_type = fusion_layer.fusion_method if hasattr(fusion_layer,
                                                                        'fusion_method') else self.fusion_method
                    fusion_out_dim = fusion_layer.out_dim if hasattr(fusion_layer,
                                                                     'out_dim') else ModelConfig.FUSION_OUTPUT_DIM
                    print(f"CTRModel init done | fusion:{fusion_type} | fusion_out_dim:{fusion_out_dim} | num_heads:{self.num_heads} | predictor:{self.predictor_type}")
                else:
                    print(f"fusion_layer not found in CTRPredictor | specified fusion:{self.fusion_method} | num_heads:{self.num_heads}")
            else:
                modal_type = "text_only" if self.is_text_mode else "kg_only"
                print(f"CTRModel init done | mode:{modal_type} | predictor_in_dim:{self.predictor_input_dim} | predictor:{self.predictor_type}")
        except Exception as e:
            print(f"fusion config print failed:{str(e)[:80]}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.MultiheadAttention):
                for name, param in m.named_parameters():
                    if param.dim() > 1 and not name.endswith('bias'):
                        init.xavier_uniform_(param)
            elif isinstance(m, nn.Embedding):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.LayerNorm):
                init.constant_(m.weight, 1.0)
                init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Sequential):
                continue

    def _init_loss_fn(self):
        if TrainConfig.loss_fn == "bce_with_logits":
            pos_weight = torch.tensor(TrainConfig.pos_weight) if hasattr(TrainConfig, 'pos_weight') else None
            if pos_weight is not None:
                self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                self.loss_fn = nn.BCEWithLogitsLoss()
        elif TrainConfig.loss_fn == "focal":
            self.loss_fn = self._focal_loss
        elif TrainConfig.loss_fn == "smooth_l1":
            self.loss_fn = nn.SmoothL1Loss()
        elif TrainConfig.loss_fn == "mse":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"unsupported loss function:{TrainConfig.loss_fn}")

    def _focal_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        gamma = TrainConfig.focal_gamma if hasattr(TrainConfig, 'focal_gamma') else 2.0
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = (1 - pt) ** gamma * BCE_loss
        return focal_loss.mean()

    def forward(self, kg_emb: torch.Tensor = None, text_emb: torch.Tensor = None) -> torch.Tensor:
        if self.is_fusion_mode:
            if kg_emb is None or text_emb is None:
                raise ValueError("kg_emb and text_emb should not be none in fusion mode")
        else:
            if self.is_text_mode:
                if text_emb is None:
                    raise ValueError("text_emb should not be none in text only mode")
                kg_emb = None
            elif self.is_kg_mode:
                if kg_emb is None:
                    raise ValueError("kg_emb should not be none in kg only mode")
                text_emb = None

        if kg_emb is not None:
            kg_emb = kg_emb.to(dtype=torch.float32)
            kg_emb = torch.clamp(kg_emb, min=-10.0, max=10.0)

        if text_emb is not None:
            text_emb = text_emb.to(dtype=torch.float32)
            text_emb = self.text_reduce(text_emb)
            text_emb = torch.clamp(text_emb, min=-10.0, max=10.0)

        logits = self.predictor(kg_emb, text_emb)
        logits = torch.clamp(logits, min=-5.0, max=5.0)
        return logits

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.to(dtype=torch.float32)
        return self.loss_fn(logits, labels)

    def predict_proba(self, kg_emb: torch.Tensor = None, text_emb: torch.Tensor = None) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            logits = self.forward(kg_emb, text_emb)
            proba = torch.sigmoid(logits)
        return proba.cpu().numpy()

    def predict(self, kg_emb: torch.Tensor = None, text_emb: torch.Tensor = None, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(kg_emb, text_emb)
        return (proba >= threshold).astype(int)

    def save(self, path: str):
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": {
                "kg_dim": self.kg_dim,
                "text_emb_version": self.text_emb_version,
                "fusion_method": self.fusion_method,
                "predictor_type": self.predictor_type,
                "num_heads": self.num_heads,
                "reduced_text_dim": self.reduced_text_dim,
                "is_fusion_mode": self.is_fusion_mode,
                "is_text_mode": self.is_text_mode,
                "is_kg_mode": self.is_kg_mode,
                "model_config": {
                    "available_fusions": ModelConfig.AVAILABLE_FUSION_METHODS,
                    "available_predictors": ModelConfig.AVAILABLE_PREDICTORS,
                    "fusion_hyperparams": self.fusion_hyperparams,
                    "predictor_hyperparams": self.predictor_hyperparams
                },
                "train_config": {
                    "loss_fn": TrainConfig.loss_fn,
                    "focal_gamma": TrainConfig.focal_gamma if hasattr(TrainConfig, 'focal_gamma') else 2.0,
                    "pos_weight": TrainConfig.pos_weight if hasattr(TrainConfig, 'pos_weight') else 1.0,
                    "weight_decay": TrainConfig.weight_decay if hasattr(TrainConfig, 'weight_decay') else 1e-4
                },
                "switch_config": {
                    "use_kg_emb": SwitchConfig.use_kg_emb,
                    "use_text_emb": SwitchConfig.use_text_emb,
                    "use_fusion_emb": SwitchConfig.use_fusion_emb
                }
            }
        }, path)
        print(f"model saved to {path}")

    @classmethod
    def load(cls, path: str):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=device)

        switch_config = checkpoint["config"].get("switch_config", {})
        if hasattr(SwitchConfig, 'update'):
            SwitchConfig.update(
                use_kg_emb=switch_config.get("use_kg_emb", True),
                use_text_emb=switch_config.get("use_text_emb", True),
                use_fusion_emb=switch_config.get("use_fusion_emb", True)
            )
        else:
            SwitchConfig.use_kg_emb = switch_config.get("use_kg_emb", True)
            SwitchConfig.use_text_emb = switch_config.get("use_text_emb", True)
            SwitchConfig.use_fusion_emb = switch_config.get("use_fusion_emb", True)

        train_config = checkpoint["config"].get("train_config", {})
        if hasattr(TrainConfig, 'update'):
            TrainConfig.update(
                loss_fn=train_config.get("loss_fn", "bce_with_logits"),
                focal_gamma=train_config.get("focal_gamma", 2.0),
                pos_weight=train_config.get("pos_weight", 1.0),
                weight_decay=train_config.get("weight_decay", 1e-4)
            )
        else:
            TrainConfig.loss_fn = train_config.get("loss_fn", "bce_with_logits")
            TrainConfig.focal_gamma = train_config.get("focal_gamma", 2.0)
            TrainConfig.pos_weight = train_config.get("pos_weight", 1.0)
            TrainConfig.weight_decay = train_config.get("weight_decay", 1e-4)

        num_heads = checkpoint["config"].get("num_heads", 4)
        model = cls(
            kg_dim=checkpoint["config"]["kg_dim"],
            text_emb_version=checkpoint["config"]["text_emb_version"],
            fusion_method=checkpoint["config"]["fusion_method"],
            predictor_type=checkpoint["config"]["predictor_type"],
            num_heads=num_heads
        )

        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.eval()
        print(f"model loaded from {path} | restored num_heads:{num_heads}")
        return model

    def train_step(self, kg_emb: torch.Tensor, text_emb: torch.Tensor, labels: torch.Tensor,
                   optimizer: torch.optim.Optimizer) -> float:
        self.train()
        optimizer.zero_grad()

        logits = self.forward(kg_emb, text_emb)
        loss = self.compute_loss(logits, labels)

        loss.backward()
        grad_clip_norm = TrainConfig.grad_clip_norm if hasattr(TrainConfig, 'grad_clip_norm') else 1.0
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        return loss.item()
