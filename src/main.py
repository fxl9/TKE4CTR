# -*- coding: utf-8 -*-
"""
CTR model training pipeline for KG‑text fusion recommendation task
"""
import os
import sys
import time
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import csv
from datetime import datetime
import argparse

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from configs.config import ModelConfig, EmbeddingConfig, SwitchConfig, TrainConfig, GPUConfig, DatasetConfig, output_cfg
from src.utils.data_loader import load_ctr_data
from src.models.ctr_model import CTRModel


class CTRTrainer:
    def __init__(self, domain: str, text_emb_version: str = None, fusion_method: str = None, predictor_type: str = None, gpu_id: int = 0, train_mode: str = "kg_text_fusion", num_heads: int = 4):
        self.domain = domain
        self.train_mode = train_mode
        self.gpu_id = gpu_id
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.num_heads = num_heads

        if self.train_mode == "only_kg":
            self.text_emb_version = None
            self.fusion_method = "none"
            self.display_text_emb = "none"
        elif self.train_mode == "only_text":
            self.text_emb_version = text_emb_version or EmbeddingConfig.DEFAULT_TEXT_EMB_VERSION
            self.fusion_method = "none"
            self.display_text_emb = self.text_emb_version
        else:
            self.text_emb_version = text_emb_version or EmbeddingConfig.DEFAULT_TEXT_EMB_VERSION
            self.fusion_method = fusion_method or ModelConfig.DEFAULT_FUSION_METHOD
            self.display_text_emb = self.text_emb_version

        self.predictor_type = predictor_type or ModelConfig.DEFAULT_PREDICTOR
        self._init_output_paths()
        self._setup_device()

        self.epochs = getattr(TrainConfig, 'epochs', 20)
        self.batch_size = getattr(TrainConfig, 'batch_size', 256)
        self.eval_batch_size = getattr(TrainConfig, 'eval_batch_size', 512)
        self.learning_rate = getattr(TrainConfig, 'learning_rate', 1e-5)
        self.weight_decay = getattr(TrainConfig, 'weight_decay', 1e-4)
        self.patience = getattr(TrainConfig, 'patience', 3)
        self.seed = getattr(TrainConfig, 'seed', 42)
        self.loss_fn_name = getattr(TrainConfig, 'loss_fn', 'bce_with_logits')

        self.eval_metrics = ["auc"]
        self.early_stopping_mode = "max"
        self.early_stopping_metric = "auc"

        self.grad_clip_norm = getattr(TrainConfig, 'grad_clip_norm', 1.0)
        self.grad_accum_steps = getattr(TrainConfig, 'grad_accum_steps', 2)

        self._set_seed()
        self._load_data()
        self._check_data_distribution()
        self.class_weights = self._compute_class_weights()
        self._log(f"SwitchConfig status | use_kg_emb: {SwitchConfig.use_kg_emb} | use_text_emb: {SwitchConfig.use_text_emb}")
        self.model = self._init_model()
        self.loss_fn = self._get_loss_fn()
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()

        self.best_metric = 0.0
        self.early_stop_count = 0
        self.train_metrics_history = {"auc": [], "loss": []}
        self.valid_metrics_history = {"auc": [], "loss": []}

        self._init_log_file()

    def _setup_device(self):
        if not isinstance(self.gpu_id, int) or self.gpu_id < 0 or self.gpu_id > 3:
            raise ValueError(f"GPU ID must be integer between 0‑3, input: {self.gpu_id}")

        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:0")
            try:
                torch.cuda.empty_cache()
                torch.tensor([1.0]).to(self.device)
                self._log(f"Use GPU {self.gpu_id}: {torch.cuda.get_device_name(self.device)}")
            except Exception as e:
                self._log(f"GPU {self.gpu_id} unavailable, fallback to CPU: {e}")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
            self._log("CUDA not available, use CPU for training, speed will be slow")

    def _init_output_paths(self):
        num_heads_suffix = f"_heads{self.num_heads}" if self.fusion_method == "cross_attention_gated" else ""

        if self.train_mode == "only_kg":
            model_name_prefix = f"best_only_kg_{self.predictor_type}{num_heads_suffix}"
            log_name = f"{self.domain}_only_kg_{self.predictor_type}{num_heads_suffix}_{self.timestamp}.log"
        elif self.train_mode == "only_text":
            model_name_prefix = f"best_only_text_{self.display_text_emb}_{self.predictor_type}{num_heads_suffix}"
            log_name = f"{self.domain}_only_text_{self.display_text_emb}_{self.predictor_type}{num_heads_suffix}_{self.timestamp}.log"
        else:
            model_name_prefix = f"best_{self.display_text_emb}_kg_{self.fusion_method}_{self.predictor_type}{num_heads_suffix}"
            log_name = f"{self.domain}_kg_text_fusion_{self.display_text_emb}_{self.fusion_method}_{self.predictor_type}{num_heads_suffix}_{self.timestamp}.log"

        model_dir = output_cfg.get_model_dir(self.domain)
        log_dir = output_cfg.get_log_dir(self.domain)
        result_dir = output_cfg.get_result_dir(self.domain)
        for dir_path in [model_dir, log_dir, result_dir]:
            os.makedirs(dir_path, exist_ok=True)

        self.best_model_path = os.path.join(model_dir, f"{model_name_prefix}_gpu{self.gpu_id}_{self.timestamp}.pth")
        self.log_path = os.path.join(log_dir, log_name)
        self.result_csv_path = os.path.join(result_dir, f"{self.domain}_results1.csv")

    def _load_data(self):
        self._log("Start loading dataset")
        try:
            self.dataloaders = load_ctr_data(
                domain=self.domain,
                text_emb_version=self.text_emb_version
            )
        except Exception as e:
            self._log(f"Data loading failed: {str(e)[:100]}")
            raise RuntimeError(f"Data loading failed: {e}")

        self.train_loader = self.dataloaders["train"]
        self.valid_loader = self.dataloaders["valid"]
        self.test_loader = self.dataloaders["test"]
        self._log(
            f"Dataset loaded | train: {len(self.train_loader.dataset)} | valid: {len(self.valid_loader.dataset)} | test: {len(self.test_loader.dataset)}")

    def _check_data_distribution(self):
        train_labels = []
        max_samples = 1000 * self.batch_size
        for i, batch in enumerate(self.train_loader):
            if len(train_labels) >= max_samples:
                break
            labels = batch["labels"].cpu().numpy()
            train_labels.extend(labels)

        if not train_labels:
            raise RuntimeError("Train set is empty, cannot analyze label distribution")

        train_labels = np.array(train_labels)
        self.pos_ratio = np.sum(train_labels == 1) / len(train_labels)
        if self.pos_ratio > 0.9 or self.pos_ratio < 0.1:
            self.classification_threshold = 0.2
            self._log(f"Skewed label distribution (positive ratio: {self.pos_ratio:.4f}), threshold: {self.classification_threshold}")
        else:
            self.classification_threshold = 0.5
            self._log(f"Normal label distribution, threshold: {self.classification_threshold}")

    def _set_seed(self):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self._log(f"Random seed set: {self.seed}")

    def _compute_class_weights(self):
        all_labels = []
        max_samples = 1000 * self.batch_size
        for i, batch in enumerate(self.train_loader):
            if len(all_labels) >= max_samples:
                break
            labels = batch["labels"].cpu().numpy()
            all_labels.extend(labels)

        all_labels = np.array(all_labels)
        classes = np.unique(all_labels)

        if len(classes) == 1:
            self._log("Train set contains only one class, use equal weight")
            return torch.tensor([1.0]).to(self.device)

        weights = compute_class_weight('balanced', classes=classes, y=all_labels)
        max_weight = 20.0
        weights = np.clip(weights, 0.1, max_weight)
        class_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        return class_weights

    def _init_model(self):
        self._log(
            f"Initialize model | fusion_method: {self.fusion_method} | predictor_type: {self.predictor_type} | train_mode: {self.train_mode} | num_heads: {self.num_heads}")
        model = CTRModel(
            kg_dim=128,
            text_emb_version=self.text_emb_version,
            fusion_method=self.fusion_method,
            predictor_type=self.predictor_type,
            num_heads=self.num_heads
        ).to(self.device)

        try:
            fusion_type = "none(non‑fusion mode)"
            if self.train_mode == "kg_text_fusion":
                if hasattr(model, 'predictor'):
                    ctr_predictor = model.predictor
                    if hasattr(ctr_predictor, 'fusion_layer'):
                        fusion_layer = ctr_predictor.fusion_layer
                        if hasattr(fusion_layer, 'fusion'):
                            fusion_type = type(fusion_layer.fusion).__name__
                        else:
                            fusion_type = type(fusion_layer).__name__
                        fusion_type = fusion_type.replace("Fusion", "").replace("Embedding", "").strip()

            predictor_type = "unknown"
            if hasattr(model, 'predictor'):
                ctr_predictor = model.predictor
                if hasattr(ctr_predictor, 'predictor'):
                    predictor_type = type(ctr_predictor.predictor).__name__.replace("Predictor", "").strip()
                else:
                    predictor_type = type(ctr_predictor).__name__.replace("Predictor", "").strip()

            self._log(
                f"Model structure check | specified fusion: {self.fusion_method} | actual fusion layer: {fusion_type} | predictor: {predictor_type} | num_heads: {self.num_heads}")
        except Exception as e:
            self._log(f"Parse model structure failed: {str(e)[:80]} | fusion_method: {self.fusion_method} | num_heads: {self.num_heads}")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._log(f"Model params | total: {total_params / 1e6:.2f}M | trainable: {trainable_params / 1e6:.2f}M")

        param_count = {}
        for name, param in model.named_parameters():
            layer_type = name.split('.')[0] if '.' in name else 'other'
            param_count[layer_type] = param_count.get(layer_type, 0) + param.numel()
        for layer, cnt in param_count.items():
            self._log(f"  - {layer}: {cnt / 1e6:.2f}M")

        return model

    def _get_loss_fn(self):
        self._log(f"Init loss function: {self.loss_fn_name}")
        if self.loss_fn_name == "bce_with_logits":
            if len(self.class_weights) > 1 and self.pos_ratio < 0.2:
                pos_weight = self.class_weights[1] / self.class_weights[0]
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(self.device)
                self._log(f"Use weighted BCE | pos_weight: {pos_weight:.2f}")
            else:
                loss_fn = nn.BCEWithLogitsLoss().to(self.device)
        elif self.loss_fn_name == "focal":
            def focal_loss(logits, labels):
                gamma = 2.0
                alpha = self.class_weights[1] if len(self.class_weights) > 1 else 0.25
                bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none')
                pt = torch.exp(-bce_loss)
                focal_loss_val = alpha * (1 - pt) ** gamma * bce_loss
                return focal_loss_val.mean()

            loss_fn = focal_loss
        elif self.loss_fn_name == "mse":
            loss_fn = nn.MSELoss().to(self.device)
        elif self.loss_fn_name == "smooth_l1":
            loss_fn = nn.SmoothL1Loss().to(self.device)
        else:
            self._log(f"Unknown loss function: {self.loss_fn_name}, fallback to BCEWithLogitsLoss")
            loss_fn = nn.BCEWithLogitsLoss().to(self.device)

        return loss_fn

    def _get_optimizer(self):
        optimizer_type = getattr(TrainConfig, 'optimizer', 'adamw').lower()
        if optimizer_type == "adamw":
            param_groups = [
                {'params': [p for n, p in self.model.named_parameters() if 'fusion' in n or 'predictor' in n],
                 'lr': self.learning_rate},
                {'params': [p for n, p in self.model.named_parameters() if 'fusion' not in n and 'predictor' not in n],
                 'lr': self.learning_rate * 0.1}
            ]
            optimizer = optim.AdamW(
                param_groups,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_type == "adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_type == "sgd":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
                nesterov=True
            )
        else:
            self._log(f"Unknown optimizer: {optimizer_type}, fallback to AdamW")
            optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        self._log(f"Optimizer initialized: {optimizer_type} | LR: {self.learning_rate} | weight_decay: {self.weight_decay}")
        return optimizer

    def _get_scheduler(self):
        scheduler_type = getattr(TrainConfig, 'lr_scheduler', 'cosine').lower()
        if scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=5,
                T_mult=2,
                eta_min=1e-6
            )
        elif scheduler_type == "step":
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=3,
                gamma=0.5
            )
        elif scheduler_type == "reduce_on_plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=self.early_stopping_mode,
                factor=0.5,
                patience=2,
                min_lr=1e-6,
            )
        elif scheduler_type == "none":
            scheduler = None
        else:
            self._log(f"Unknown scheduler: {scheduler_type}, fallback to cosine scheduler")
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=5, T_mult=2, eta_min=1e-6
            )
        self._log(f"LR scheduler: {scheduler_type if scheduler else 'none'}")
        return scheduler

    def _init_log_file(self):
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(f"Train start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"===== Train Config =====\n")
            f.write(f"dataset domain: {self.domain}\n")
            f.write(f"train mode: {self.train_mode}\n")
            f.write(f"text embedding version: {self.display_text_emb}\n")
            f.write(f"fusion method: {self.fusion_method}\n")
            f.write(f"num_heads: {self.num_heads}\n")
            f.write(f"predictor type: {self.predictor_type}\n")
            f.write(f"gpu id: {self.gpu_id}\n")
            f.write(f"epochs: {self.epochs}\n")
            f.write(f"batch size: {self.batch_size}\n")
            f.write(f"eval batch size: {self.eval_batch_size}\n")
            f.write(f"learning rate: {self.learning_rate}\n")
            f.write(f"weight decay: {self.weight_decay}\n")
            f.write(f"grad clip norm: {self.grad_clip_norm}\n")
            f.write(f"grad accum steps: {self.grad_accum_steps}\n")
            f.write(f"loss function: {self.loss_fn_name}\n")
            f.write(f"early stop metric: {self.early_stopping_metric}\n")
            f.write(f"early stop mode: {self.early_stopping_mode}\n")
            f.write(f"early stop patience: {self.patience}\n")
            f.write(f"random seed: {self.seed}\n")
            f.write(f"result csv path: {self.result_csv_path}\n")
            f.write(f"best model path: {self.best_model_path}\n")
            f.write(f"device: {self.device}\n")
            f.write(f"=" * 80 + "\n\n")

    def _log(self, msg: str):
        print(msg)
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception as e:
            print(f"Log write failed: {e}")

    def _compute_metrics(self, logits: torch.Tensor, labels: torch.Tensor):
        proba = torch.sigmoid(logits).cpu().detach().numpy()
        labels = labels.cpu().numpy()
        metrics = {}
        try:
            if len(np.unique(labels)) <= 1:
                metrics["auc"] = 0.5
                self._log("AUC warning: only one class in labels, return 0.5")
            else:
                metrics["auc"] = roc_auc_score(labels, proba)
        except Exception as e:
            metrics["auc"] = 0.5
            self._log(f"AUC compute failed: {e}, return 0.5")
        return metrics

    def train_one_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        all_logits = []
        all_labels = []
        self.optimizer.zero_grad()
        clear_cache_interval = 100

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs} Train [GPU{self.gpu_id}]")
        for step, batch in enumerate(pbar):
            if step % clear_cache_interval == 0 and step > 0:
                torch.cuda.empty_cache()

            labels = batch["labels"].to(self.device, dtype=torch.float32)
            kg_emb = batch.get("kg_emb", None)
            text_emb = batch.get("text_emb", None)

            if kg_emb is not None:
                kg_emb = kg_emb.to(self.device, non_blocking=True)
            if text_emb is not None:
                text_emb = text_emb.to(self.device, non_blocking=True)

            logits = self.model(kg_emb, text_emb).squeeze()
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)
            if logits.shape != labels.shape:
                logits = logits.reshape(labels.shape)

            loss = self.loss_fn(logits, labels)
            loss = loss / self.grad_accum_steps
            loss.backward()

            if (step + 1) % self.grad_accum_steps == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * labels.shape[0]
            max_samples = 50000
            if len(all_logits) < max_samples:
                all_logits.append(logits.detach())
                all_labels.append(labels.detach())

            pbar.set_postfix(
                {"loss": f"{loss.item() * self.grad_accum_steps:.4f}",
                 "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}"})

        avg_loss = total_loss / len(self.train_loader.dataset)
        metrics = {"auc": 0.0}
        if len(all_logits) > 0:
            all_logits = torch.cat(all_logits)
            all_labels = torch.cat(all_labels)
            metrics = self._compute_metrics(all_logits, all_labels)

        self.train_metrics_history["loss"].append(avg_loss)
        self.train_metrics_history["auc"].append(metrics["auc"])

        self._log(f"Epoch {epoch + 1} Train | Loss: {avg_loss:.4f} | LR: {self.optimizer.param_groups[0]['lr']:.6f} | auc:{metrics['auc']:.4f}")
        torch.cuda.empty_cache()
        return avg_loss, metrics

    @torch.no_grad()
    def validate(self, epoch: int):
        self.model.eval()
        total_loss = 0.0
        all_logits = []
        all_labels = []
        torch.cuda.empty_cache()

        pbar = tqdm(self.valid_loader, desc=f"Epoch {epoch + 1}/{self.epochs} Valid [GPU{self.gpu_id}]")
        for batch in pbar:
            labels = batch["labels"].to(self.device, dtype=torch.float32)
            kg_emb = batch.get("kg_emb", None)
            text_emb = batch.get("text_emb", None)

            if kg_emb is not None:
                kg_emb = kg_emb.to(self.device, non_blocking=True)
            if text_emb is not None:
                text_emb = text_emb.to(self.device, non_blocking=True)

            logits = self.model(kg_emb, text_emb).squeeze()
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)
            if logits.shape != labels.shape:
                logits = logits.reshape(labels.shape)

            loss = self.loss_fn(logits, labels)
            total_loss += loss.item() * labels.shape[0]
            all_logits.append(logits)
            all_labels.append(labels)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.valid_loader.dataset)
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        metrics = self._compute_metrics(all_logits, all_labels)

        self.valid_metrics_history["loss"].append(avg_loss)
        self.valid_metrics_history["auc"].append(metrics["auc"])
        self._log(f"Epoch {epoch + 1} Valid | Loss: {avg_loss:.4f} | auc:{metrics['auc']:.4f}")

        torch.cuda.empty_cache()
        return avg_loss, metrics

    def _check_early_stop(self, current_metric):
        eps = 1e-4
        if current_metric > self.best_metric + eps:
            self.best_metric = current_metric
            self.early_stop_count = 0
            self._log(f"Validation {self.early_stopping_metric} improved: {current_metric:.4f} (best: {self.best_metric:.4f})")
            return True
        else:
            self.early_stop_count += 1
            return False

    def _append_result_to_csv(self, test_metrics):
        result_row = {
            "timestamp": self.timestamp,
            "gpu_id": self.gpu_id,
            "domain": self.domain,
            "train_mode": self.train_mode,
            "text_emb_version": self.display_text_emb,
            "fusion_method": self.fusion_method,
            "num_heads": self.num_heads,
            "predictor_type": self.predictor_type,
            "best_valid_auc": round(self.best_metric, 4),
            "test_loss": round(test_metrics.get("loss", 0), 4),
            "test_auc": round(test_metrics.get("auc", 0), 4),
            "epochs_trained": len(self.train_metrics_history["loss"]),
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "pos_ratio": round(self.pos_ratio, 4),
            "classification_threshold": self.classification_threshold,
            "optimizer": getattr(TrainConfig, 'optimizer', 'adamw'),
            "lr_scheduler": getattr(TrainConfig, 'lr_scheduler', 'cosine'),
            "loss_function": self.loss_fn_name,
            "weight_decay": self.weight_decay,
            "patience": self.patience,
            "seed": self.seed,
            "device": str(self.device),
            "best_model_path": self.best_model_path,
            "total_train_samples": len(self.train_loader.dataset),
            "grad_accum_steps": self.grad_accum_steps
        }

        file_exists = os.path.exists(self.result_csv_path)
        try:
            with open(self.result_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=result_row.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(result_row)
            self._log(f"Experiment result saved to: {self.result_csv_path}")
        except Exception as e:
            self._log(f"CSV write failed: {e}")
            self._log(f"Result content: {result_row}")

    def train(self):
        self._log("\n" + "=" * 80)
        self._log(f"Start training dataset {self.domain}")
        self._log(
            f"Config: train_mode={self.train_mode} | LLM={self.display_text_emb} | fusion={self.fusion_method} | num_heads={self.num_heads} | predictor={self.predictor_type}")
        self._log(f"GPU: {self.gpu_id} | device: {self.device} | train samples: {len(self.train_loader.dataset)}")
        self._log(
            f"Hyper‑params: Epochs={self.epochs} | BatchSize={self.batch_size} | LR={self.learning_rate} | weight_decay={self.weight_decay}")
        self._log("=" * 80 + "\n")

        start_time = time.time()
        try:
            for epoch in range(self.epochs):
                self.train_one_epoch(epoch)
                val_loss, val_metrics = self.validate(epoch)

                current_metric = val_metrics.get(self.early_stopping_metric, 0.0)
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(current_metric)
                    else:
                        self.scheduler.step()

                if self._check_early_stop(current_metric):
                    save_dict = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_metric': self.best_metric,
                        'config': {
                            'domain': self.domain,
                            'train_mode': self.train_mode,
                            'text_emb_version': self.text_emb_version,
                            'fusion_method': self.fusion_method,
                            'num_heads': self.num_heads,
                            'predictor_type': self.predictor_type,
                            'pos_ratio': self.pos_ratio,
                            'threshold': self.classification_threshold,
                            'hyper_params': {
                                'lr': self.learning_rate,
                                'batch_size': self.batch_size,
                                'weight_decay': self.weight_decay
                            }
                        }
                    }
                    try:
                        torch.save(save_dict, self.best_model_path)
                        self._log(f"Best model saved to: {self.best_model_path}")
                    except Exception as e:
                        self._log(f"Model save failed: {e}")
                else:
                    self._log(
                        f"No improvement on validation {self.early_stopping_metric} | current: {current_metric:.4f} | best: {self.best_metric:.4f} | early‑stop count: {self.early_stop_count}/{self.patience}")
                    if self.early_stop_count >= self.patience:
                        self._log(f"Early‑stop triggered, no improvement for {self.patience} consecutive epochs")
                        break

        except KeyboardInterrupt:
            self._log("Training interrupted manually")
        except Exception as e:
            self._log(f"Error during training: {str(e)[:100]}")
            raise

        total_time = time.time() - start_time
        self._log("\n" + "=" * 80)
        self._log(f"Training finished | total time: {total_time / 60:.2f} min | best valid {self.early_stopping_metric}: {self.best_metric:.4f}")
        self._log("=" * 80 + "\n")

        test_metrics = self.test()
        self._append_result_to_csv(test_metrics)

    @torch.no_grad()
    def test(self):
        self._log("\nStart test set evaluation, load best checkpoint")
        try:
            checkpoint = torch.load(self.best_model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self._log(f"Best model loaded (Epoch {checkpoint['epoch']} | num_heads: {checkpoint['config']['num_heads']})")
        except Exception as e:
            self._log(f"Load best model failed: {e}, use current model weights")

        self.model.eval()
        total_loss = 0.0
        all_logits = []
        all_labels = []
        torch.cuda.empty_cache()

        pbar = tqdm(self.test_loader, desc=f"Test [GPU{self.gpu_id}]")
        for batch in pbar:
            labels = batch["labels"].to(self.device, dtype=torch.float32)
            kg_emb = batch.get("kg_emb", None)
            text_emb = batch.get("text_emb", None)

            if kg_emb is not None:
                kg_emb = kg_emb.to(self.device, non_blocking=True)
            if text_emb is not None:
                text_emb = text_emb.to(self.device, non_blocking=True)

            logits = self.model(kg_emb, text_emb).squeeze()
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)
            if logits.shape != labels.shape:
                logits = logits.reshape(labels.shape)

            loss = self.loss_fn(logits, labels)
            total_loss += loss.item() * labels.shape[0]
            all_logits.append(logits)
            all_labels.append(labels)

        avg_loss = total_loss / len(self.test_loader.dataset)
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        metrics = self._compute_metrics(all_logits, all_labels)
        metrics["loss"] = avg_loss

        self._log("\n" + "=" * 80)
        self._log(f"Test result | Loss: {avg_loss:.4f} | auc:{metrics['auc']:.4f}")
        self._log("=" * 80 + "\n")
        return metrics


def main_sig():
    parser = argparse.ArgumentParser(description="CTR prediction model training main entry")
    parser.add_argument("--gpu_id", type=int, default=3, help="GPU ID (0‑3)")
    parser.add_argument("--domain", type=str, default="All_Beauty", help="target dataset domain")
    parser.add_argument("--text_emb_version", type=str, default="Qwen3‑4b", help="text embedding version")
    parser.add_argument("--fusion_method", type=str, default="cross_attention_gated", help="fusion method")
    parser.add_argument("--predictor_type", type=str, default="mlp_final", help="predictor head type")
    parser.add_argument("--num_heads", type=int, default=4, help="attention heads for cross_attention_gated, suggest power‑of‑two:2,4,8,16")
    parser.add_argument("--epochs", type=int, default=50, help="max training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="train batch size")
    parser.add_argument("--eval_batch_size", type=int, default=512, help="valid/test batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay for regularization")
    parser.add_argument("--patience", type=int, default=25, help="early‑stop patience")
    parser.add_argument("--optimizer", type=str, default="adamw", help="optimizer type")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", help="lr scheduler type")
    parser.add_argument("--loss_fn", type=str, default="bce_with_logits", help="loss function")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="gradient clip max norm")
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="gradient accumulation steps")
    parser.add_argument("--train_mode", type=str, default="kg_text_fusion", choices=["only_kg", "only_text", "kg_text_fusion"], help="training mode")

    args = parser.parse_args()

    if args.num_heads <= 0 or not isinstance(args.num_heads, int):
        raise ValueError(f"num_heads must be positive integer, input: {args.num_heads}")
    if (args.num_heads & (args.num_heads - 1)) != 0 and args.fusion_method == "cross_attention_gated":
        print(f"Warning: num_heads recommend power‑of‑two(2,4,8,16), current {args.num_heads}, may reduce attention efficiency")

    supported_predictors = ModelConfig.AVAILABLE_PREDICTORS
    if args.predictor_type not in supported_predictors:
        raise ValueError(f"predictor_type only support {supported_predictors}, input: {args.predictor_type}")

    supported_fusions = ModelConfig.AVAILABLE_FUSION_METHODS
    if args.fusion_method not in supported_fusions and args.train_mode == "kg_text_fusion":
        raise ValueError(f"fusion_method only support {supported_fusions}, input: {args.fusion_method}")

    if args.train_mode == "kg_text_fusion":
        SwitchConfig.use_kg_emb = True
        SwitchConfig.use_text_emb = True
        SwitchConfig.use_fusion_emb = True
    elif args.train_mode == "only_kg":
        SwitchConfig.use_kg_emb = True
        SwitchConfig.use_text_emb = False
        SwitchConfig.use_fusion_emb = False
    elif args.train_mode == "only_text":
        SwitchConfig.use_kg_emb = False
        SwitchConfig.use_text_emb = True
        SwitchConfig.use_fusion_emb = False

    TrainConfig.epochs = args.epochs
    TrainConfig.batch_size = args.batch_size
    TrainConfig.eval_batch_size = args.eval_batch_size
    TrainConfig.learning_rate = args.lr
    TrainConfig.weight_decay = args.weight_decay
    TrainConfig.patience = args.patience
    TrainConfig.optimizer = args.optimizer
    TrainConfig.lr_scheduler = args.lr_scheduler
    TrainConfig.loss_fn = args.loss_fn
    TrainConfig.early_stopping_metric = "auc"
    TrainConfig.early_stopping_mode = "max"
    TrainConfig.seed = args.seed
    TrainConfig.eval_metrics = ["auc"]
    TrainConfig.grad_clip_norm = args.grad_clip_norm
    TrainConfig.grad_accum_steps = args.grad_accum_steps

    trainer = CTRTrainer(
        domain=args.domain,
        text_emb_version=args.text_emb_version,
        fusion_method=args.fusion_method,
        predictor_type=args.predictor_type,
        gpu_id=args.gpu_id,
        train_mode=args.train_mode,
        num_heads=args.num_heads
    )
    trainer.train()


if __name__ == "__main__":
    main_sig()
