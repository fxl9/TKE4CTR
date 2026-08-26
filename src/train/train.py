# -*- coding: utf-8 -*-
"""
CTR model trainer module.
Encapsulate full training pipeline including data loading, model initialization,
epoch training loop, validation, early‑stopping, checkpoint saving and test evaluation.
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

sys.path.insert(0, "./")

from configs.config import (
    ModelConfig, EmbeddingConfig, SwitchConfig, TrainConfig, GPUConfig,
    DatasetConfig, output_cfg
)
from src.utils.data_loader import load_ctr_data
from src.models.ctr_model import CTRModel


class CTRTrainer:
    def __init__(self, domain: str, text_emb_version: str = None, fusion_method: str = None,
                 predictor_type: str = None, gpu_id: int = 0):
        self.domain = domain
        self.text_emb_version = text_emb_version or EmbeddingConfig.DEFAULT_TEXT_EMB_VERSION
        self.fusion_method = fusion_method or ModelConfig.DEFAULT_FUSION_METHOD
        self.predictor_type = predictor_type or ModelConfig.DEFAULT_PREDICTOR
        self.gpu_id = gpu_id
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._init_output_paths()
        self._setup_device()

        self.epochs = getattr(TrainConfig, 'epochs', 20)
        self.batch_size = getattr(TrainConfig, 'batch_size', 256)
        self.eval_batch_size = getattr(TrainConfig, 'eval_batch_size', 512)
        self.learning_rate = getattr(TrainConfig, 'learning_rate', 5e-5)
        self.weight_decay = getattr(TrainConfig, 'weight_decay', 1e-6)
        self.patience = getattr(TrainConfig, 'patience', 5)
        self.seed = getattr(TrainConfig, 'seed', 42)
        self.loss_fn_name = getattr(TrainConfig, 'loss_fn', 'bce_with_logits')
        self.eval_metrics = ["auc"]
        self.early_stopping_mode = "max"
        self.early_stopping_metric = "auc"

        self._set_seed()
        self._load_data()
        self._check_data_distribution()
        self.class_weights = self._compute_class_weights()
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
        if self.gpu_id < 0 or self.gpu_id > 3:
            raise ValueError(f"GPU ID must be between 0‑3, input:{self.gpu_id}")

        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:0")
            self._log(f"Use GPU {self.gpu_id}: {torch.cuda.get_device_name(self.device)}")
        else:
            self.device = torch.device("cpu")
            self._log("CUDA not available, use CPU for training")

    def _init_output_paths(self):
        self.best_model_path = output_cfg.get_best_model_path(
            domain=self.domain,
            text_emb_version=self.text_emb_version,
            fusion_method=self.fusion_method,
            predictor=self.predictor_type
        )
        self.best_model_path = os.path.splitext(self.best_model_path)[0] + f"_gpu{self.gpu_id}_{self.timestamp}.pth"

        self.log_path = output_cfg.get_log_path(
            domain=self.domain,
            fusion_method=self.fusion_method,
            predictor=self.predictor_type,
            text_emb_version=self.text_emb_version,
            timestamp=self.timestamp
        )

        result_dir = output_cfg.get_result_dir(self.domain)
        self.result_csv_path = os.path.join(result_dir, f"{self.domain}_results.csv")

        for dir_path in [os.path.dirname(self.best_model_path), os.path.dirname(self.log_path), result_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def _load_data(self):
        self._log("Loading data...")
        try:
            self.dataloaders = load_ctr_data(
                domain=self.domain,
                text_emb_version=self.text_emb_version
            )
        except Exception as e:
            self._log(f"Load data failed: {e}")
            raise

        self.train_loader = self.dataloaders["train"]
        self.valid_loader = self.dataloaders["valid"]
        self.test_loader = self.dataloaders["test"]
        self._log(
            f"Data loaded | train:{len(self.train_loader.dataset)} | valid:{len(self.valid_loader.dataset)} | test:{len(self.test_loader.dataset)}")

    def _check_data_distribution(self):
        train_labels = []
        for batch in self.train_loader:
            labels = batch["labels"].cpu().numpy()
            train_labels.extend(labels)
        train_labels = np.array(train_labels)

        self.pos_ratio = np.sum(train_labels == 1) / len(train_labels)
        self._log(f"Train positive ratio: {self.pos_ratio:.4f}")

        if self.pos_ratio > 0.9 or self.pos_ratio < 0.1:
            self.classification_threshold = 0.2
        else:
            self.classification_threshold = 0.5

    def _set_seed(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed) if torch.cuda.is_available() else None
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _compute_class_weights(self):
        all_labels = []
        for batch in self.train_loader:
            labels = batch["labels"].cpu().numpy()
            all_labels.extend(labels)
        all_labels = np.array(all_labels)

        classes = np.unique(all_labels)
        if len(classes) == 1:
            return torch.tensor([1.0]).to(self.device)

        weights = compute_class_weight('balanced', classes=classes, y=all_labels)
        max_weight = 20.0
        weights = np.clip(weights, 0.1, max_weight)
        class_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        self._log(f"Class weights: {class_weights.cpu().numpy()}")
        return class_weights

    def _init_model(self):
        model = CTRModel(
            kg_dim=128,
            text_emb_version=self.text_emb_version,
            fusion_method=self.fusion_method,
            predictor_type=self.predictor_type
        ).to(self.device)

        try:
            fusion_type = "none"
            if SwitchConfig.use_fusion_emb:
                ctr_predictor = model.predictor
                if hasattr(ctr_predictor, 'fusion_layer'):
                    embedding_fusion = ctr_predictor.fusion_layer
                    if hasattr(embedding_fusion, 'fusion'):
                        fusion_type = type(embedding_fusion.fusion).__name__
                    else:
                        fusion_type = type(embedding_fusion).__name__

            predictor_type_name = "unknown"
            ctr_predictor = model.predictor
            if hasattr(ctr_predictor, 'predictor'):
                predictor_type_name = type(ctr_predictor.predictor).__name__

            self._log(f"Fusion layer type: {fusion_type:<25} | Predictor type: {predictor_type_name:<20}")
        except Exception as e:
            self._log(f"Get fusion/predictor type failed: {str(e)[:80]} (no impact on training)")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._log(f"Model params: total={total_params / 1e6:.2f}M | trainable={trainable_params / 1e6:.2f}M")

        return model

    def _get_loss_fn(self):
        if len(self.class_weights) == 1:
            return nn.BCEWithLogitsLoss()

        if self.loss_fn_name == "bce_with_logits":
            pos_weight = self.class_weights[1] if len(self.class_weights) > 1 else torch.tensor(1.0).to(self.device)
            if self.pos_ratio < 0.1:
                pos_weight = pos_weight * 2.0
            elif self.pos_ratio > 0.9:
                pos_weight = pos_weight * 0.5
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            return loss_fn
        elif self.loss_fn_name == "bce":
            return nn.BCELoss(weight=self.class_weights)
        else:
            return nn.BCEWithLogitsLoss()

    def _get_optimizer(self):
        optimizer_type = getattr(TrainConfig, 'optimizer', 'adamw')
        params = self.model.parameters()

        if optimizer_type == "adamw":
            return optim.AdamW(
                params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_type == "adam":
            return optim.Adam(
                params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_type == "sgd":
            return optim.SGD(
                params,
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
                nesterov=True
            )
        else:
            return optim.AdamW(params, lr=self.learning_rate)

    def _get_scheduler(self):
        scheduler_type = getattr(TrainConfig, 'lr_scheduler', 'reduce_on_plateau')

        if scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=5,
                T_mult=2,
                eta_min=1e-6
            )
        elif scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=5,
                gamma=0.5
            )
        elif scheduler_type == "reduce_on_plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        else:
            self._log(f"Unknown scheduler {scheduler_type}, use cosine")
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=5,
                T_mult=2,
                eta_min=1e-6
            )

    def _init_log_file(self):
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(f"Train start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"===== Train Config =====\n")
            f.write(f"domain: {self.domain}\n")
            f.write(f"text_emb_version: {self.text_emb_version}\n")
            f.write(f"fusion_method: {self.fusion_method}\n")
            f.write(f"predictor_type: {self.predictor_type}\n")
            f.write(f"GPU ID: {self.gpu_id}\n")
            f.write(f"epochs: {self.epochs}\n")
            f.write(f"batch_size: {self.batch_size}\n")
            f.write(f"lr: {self.learning_rate}\n")
            f.write(f"weight_decay: {self.weight_decay}\n")
            f.write(f"loss: {self.loss_fn_name}\n")
            f.write(f"early_stop metric: auc\n")
            f.write(f"patience: {self.patience}\n")
            f.write(f"csv path: {self.result_csv_path}\n")
            f.write(f"=" * 80 + "\n\n")

    def _log(self, msg: str):
        print(msg)
        try:
            if not hasattr(self, 'log_path') or not self.log_path:
                self.log_path = os.path.join("./", "temp_train.log")
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception as e:
            print(f"Log write failed: {e}")

    def _compute_metrics(self, logits: torch.Tensor, labels: torch.Tensor):
        proba = torch.sigmoid(logits).cpu().detach().numpy()
        labels = labels.cpu().numpy()
        metrics = {}

        if len(np.unique(labels)) <= 1:
            metrics["auc"] = 0.5
        else:
            try:
                metrics["auc"] = roc_auc_score(labels, proba)
            except Exception:
                metrics["auc"] = 0.5
        return metrics

    def train_one_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        all_logits = []
        all_labels = []

        grad_accum_steps = 2
        self.optimizer.zero_grad()
        clear_cache_interval = 100

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs} Train [GPU{self.gpu_id}]")
        for step, batch in enumerate(pbar):
            if step % clear_cache_interval == 0 and step > 0:
                torch.cuda.empty_cache()

            labels = batch["labels"].to(self.device).float()
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
            loss = loss / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * grad_accum_steps * labels.shape[0]
            if len(all_logits) < 10000:
                all_logits.append(logits.detach())
                all_labels.append(labels.detach())

            pbar.set_postfix({"loss": f"{loss.item() * grad_accum_steps:.4f}",
                              "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}"})

        avg_loss = total_loss / len(self.train_loader.dataset)
        if len(all_logits) > 0:
            all_logits = torch.cat(all_logits)
            all_labels = torch.cat(all_labels)
            metrics = self._compute_metrics(all_logits, all_labels)
        else:
            metrics = {"auc": 0.0}

        self.train_metrics_history["loss"].append(avg_loss)
        self.train_metrics_history["auc"].append(metrics["auc"])

        self._log(f"Epoch {epoch+1} Train | Loss:{avg_loss:.4f} | LR:{self.optimizer.param_groups[0]['lr']:.6f} | auc:{metrics['auc']:.4f}")
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
            labels = batch["labels"].to(self.device).float()
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
            if len(all_logits) < 5000:
                all_logits.append(logits)
                all_labels.append(labels)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.valid_loader.dataset)
        if len(all_logits) > 0:
            all_logits = torch.cat(all_logits)
            all_labels = torch.cat(all_labels)
            metrics = self._compute_metrics(all_logits, all_labels)
        else:
            metrics = {"auc":0.0}

        self.valid_metrics_history["loss"].append(avg_loss)
        self.valid_metrics_history["auc"].append(metrics["auc"])
        self._log(f"Epoch {epoch+1} Valid | Loss:{avg_loss:.4f} | auc:{metrics['auc']:.4f}")

        torch.cuda.empty_cache()
        return avg_loss, metrics

    def _check_early_stop(self, current_metric):
        if current_metric > self.best_metric + 1e-4:
            self.best_metric = current_metric
            self.early_stop_count = 0
            return True
        else:
            self.early_stop_count += 1
            return False

    def _append_result_to_csv(self, test_metrics):
        result_row = {
            "timestamp": self.timestamp,
            "gpu_id": self.gpu_id,
            "domain": self.domain,
            "text_emb_version": self.text_emb_version,
            "fusion_method": self.fusion_method,
            "predictor_type": self.predictor_type,
            "best_valid_auc": round(self.best_metric, 4),
            "test_loss": round(test_metrics.get("loss",0),4),
            "test_auc": round(test_metrics.get("auc",0),4),
            "epochs_trained": len(self.train_metrics_history["loss"]),
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "pos_ratio": round(self.pos_ratio,4),
            "classification_threshold": self.classification_threshold,
            "optimizer": getattr(TrainConfig, 'optimizer', 'adamw'),
            "lr_scheduler": getattr(TrainConfig, 'lr_scheduler', 'reduce_on_plateau'),
            "loss_function": self.loss_fn_name,
            "weight_decay": self.weight_decay,
            "patience": self.patience,
            "seed": self.seed
        }

        file_exists = os.path.exists(self.result_csv_path)
        try:
            with open(self.result_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=result_row.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(result_row)
            self._log(f"Result appended to {self.result_csv_path}")
        except Exception as e:
            self._log(f"CSV write failed: {e}")
            self._log(f"Result row: {result_row}")

    def train(self):
        self._log(f"===== Start training domain {self.domain} =====")
        self._log(f"config: LLM={self.text_emb_version} | fusion={self.fusion_method} | predictor={self.predictor_type}")
        self._log(f"GPU:{self.gpu_id} | device:{self.device} | train samples:{len(self.train_loader.dataset)}")
        self._log(f"hyper: epochs={self.epochs} | batch={self.batch_size} | lr={self.learning_rate} | grad_acc=2")

        start_time = time.time()
        for epoch in range(self.epochs):
            self.train_one_epoch(epoch)
            val_loss, val_metrics = self.validate(epoch)

            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["auc"])
                else:
                    self.scheduler.step()

            current_metric = val_metrics["auc"]
            if self._check_early_stop(current_metric):
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_metric': self.best_metric,
                        'config': {
                            'domain': self.domain,
                            'text_emb_version': self.text_emb_version,
                            'fusion_method': self.fusion_method,
                            'predictor_type': self.predictor_type,
                            'pos_ratio': self.pos_ratio,
                            'threshold': self.classification_threshold
                        }
                    }, self.best_model_path)
                    self._log(f"Save best model (auc={current_metric:.4f}) -> {self.best_model_path}")
                except Exception as e:
                    self._log(f"Save model failed: {e}")
            else:
                self._log(f"AUC not improved (current:{current_metric:.4f}, best:{self.best_metric:.4f}), early‑stop count:{self.early_stop_count}/{self.patience}")
                if self.early_stop_count >= self.patience:
                    self._log(f"Early‑stop triggered, patience {self.patience} exhausted")
                    break

        total_time = time.time() - start_time
        self._log(f"\n===== Training finished =====")
        self._log(f"total time: {total_time/60:.2f} min | best valid auc: {self.best_metric:.4f}")

        test_metrics = self.test()
        self._append_result_to_csv(test_metrics)

    @torch.no_grad()
    def test(self):
        self._log(f"\n===== Test evaluation =====")
        try:
            import numpy
            torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])
            checkpoint = torch.load(self.best_model_path, map_location=self.device, weights_only=False)

            model_state_dict = checkpoint['model_state_dict']
            current_state_dict = self.model.state_dict()
            filtered_state_dict = {}
            for k, v in model_state_dict.items():
                if k in current_state_dict and current_state_dict[k].shape == v.shape:
                    filtered_state_dict[k] = v
                else:
                    self._log(f"Skip unmatched param {k}: ckpt {v.shape}, current {current_state_dict.get(k, None).shape if k in current_state_dict else 'none'}")
            self.model.load_state_dict(filtered_state_dict, strict=False)
            self._log(f"Loaded {len(filtered_state_dict)}/{len(model_state_dict)} params from epoch {checkpoint['epoch']}")
        except Exception as e:
            self._log(f"Load checkpoint failed: {e}, use current weights")

        self.model.eval()
        total_loss = 0.0
        all_logits = []
        all_labels = []
        torch.cuda.empty_cache()

        pbar = tqdm(self.test_loader, desc=f"Test [GPU{self.gpu_id}]")
        for batch in pbar:
            labels = batch["labels"].to(self.device).float()
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

        self._log(f"Test | Loss:{avg_loss:.4f} | auc:{metrics['auc']:.4f}")
        return metrics
