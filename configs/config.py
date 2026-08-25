import os
import sys
import time
from typing import Dict, List, Optional, Union

TIMESTAMP = time.strftime("%Y%m%d_%H%M%S", time.localtime())

class DatasetConfig:
    AVAILABLE_DOMAINS: List[str] = ["All_Beauty", "Amazon_Fashion", "Digital_Music", "Gift_Cards", "Musical_Instruments"]
    DATASET_TYPES: List[str] = ["train", "valid", "test"]
    DEFAULT_DOMAIN: str = "All_Beauty"
    CTR_SAMPLE_PATH_TEMPLATE: str = os.path.join(
        "datasets", "amazon_review_data", "data_splits",
        "{domain}", "{domain}_{data_type}.csv"
    )

    @classmethod
    def update(cls,** kwargs):
        for k, v in kwargs.items():
            if hasattr(cls, k):
                setattr(cls, k, v)
            else:
                raise ValueError(f"DatasetConfig has no config item: {k}")

class EmbeddingConfig:
    AVAILABLE_TEXT_EMB_VERSIONS: List[str] = [
        "bert-base-uncased",
        "deberta-v3-large",
        "gpt2",
        "Qwen3-4b",
        "Sheared-LLaMA-1.3B",
        "Hunyuan-4B-Instruct"
    ]
    LLM_DIM_MAP: Dict[str, int] = {
        "bert-base-uncased": 768,
        "deberta-v3-large": 1024,
        "gpt2": 768,
        "Qwen3-4b": 2560,
        "Sheared-LLaMA-1.3B": 2048,
        "Hunyuan-4B-Instruct": 3072
    }
    DEFAULT_TEXT_EMB_VERSION: str = "bert-base-uncased"

    KG_GAT_EMB_PATH_TEMPLATE: str = os.path.join(
        "datasets", "amazon_review_data", "gnn_embedding",
        "{domain}", "{data_type}_gat_embeddings.npy"
    )
    TEXT_EMB_PATH_TEMPLATE: str = os.path.join(
        "datasets", "amazon_review_data", "text_embedding",
        "{domain}", "{data_type}", "{emb_version}", "embeddings.npy"
    )

    @classmethod
    def update(cls,** kwargs):
        for k, v in kwargs.items():
            if hasattr(cls, k):
                setattr(cls, k, v)
            else:
                raise ValueError(f"EmbeddingConfig has no config item: {k}")

class GPUConfig:
    use_gpu: bool = True
    gpu_ids: Optional[List[int]] = [0]
    default_gpu_id: int = 0
    use_amp: bool = True
    max_gpu_memory: Optional[int] = 32
    parallel_strategy: str = "ddp"

    @property
    def device(self) -> str:
        if not self.use_gpu:
            return "cpu"
        if self.gpu_ids is None or len(self.gpu_ids) == 0:
            return "cpu"
        return f"cuda:{self.gpu_ids[0]}" if len(self.gpu_ids) == 1 else "cuda"

    @classmethod
    def update(cls,** kwargs):
        for k, v in kwargs.items():
            if hasattr(cls, k):
                setattr(cls, k, v)
            else:
                raise ValueError(f"GPUConfig has no config item: {k}")

class ModelConfig:
    AVAILABLE_FUSION_METHODS: List[str] = [
        "concat",
        "cross_modal_interaction",
        "cross_attention_gated"
    ]
    DEFAULT_FUSION_METHOD: str = "cross_attention_gated"

    GLOBAL_DROPOUT_RATE: float = 0.3
    FUSION_OUTPUT_DIM: int = 512
    KG_EMB_DIM: int = 128
    TEXT_EMB_DIM: Dict[str, int] = EmbeddingConfig.LLM_DIM_MAP

    FUSION_HYPERPARAMS: Dict[str, Dict] = {
        "concat": {
            "dropout": GLOBAL_DROPOUT_RATE,
            "hidden_dim": FUSION_OUTPUT_DIM
        },
        "cross_modal_interaction": {
            "hidden_dim": 256,
            "num_heads": 4,
            "dropout": GLOBAL_DROPOUT_RATE
        },
        "cross_attention_gated": {
            "num_heads": 4,
            "hidden_dim": FUSION_OUTPUT_DIM,
            "dropout": GLOBAL_DROPOUT_RATE
        }
    }

    AVAILABLE_PREDICTORS: List[str] = [
        "mlp_final",
        "dcnv3_fm",
        "cross_modal_fm"
    ]
    DEFAULT_PREDICTOR: str = "mlp_final"

    PREDICTOR_HYPERPARAMS: Dict[str, Dict] = {
        "mlp_final": {
            "hidden_dims": [256],
            "dropout": GLOBAL_DROPOUT_RATE,
            "activation": "relu"
        },
        "dcnv3_fm": {
            "num_cross_layers": 3,
            "hidden_dim": FUSION_OUTPUT_DIM,
            "embedding_dim": 64,
            "dropout": GLOBAL_DROPOUT_RATE
        },
        "cross_modal_fm": {
            "embedding_dim": 64,
            "dropout": GLOBAL_DROPOUT_RATE
        }
    }

    @classmethod
    def update(cls,** kwargs):
        for k, v in kwargs.items():
            if hasattr(cls, k):
                setattr(cls, k, v)
                if k == "GLOBAL_DROPOUT_RATE":
                    for fusion_type in cls.FUSION_HYPERPARAMS:
                        cls.FUSION_HYPERPARAMS[fusion_type]["dropout"] = v
                    for predictor_type in cls.PREDICTOR_HYPERPARAMS:
                        if "dropout" in cls.PREDICTOR_HYPERPARAMS[predictor_type]:
                            cls.PREDICTOR_HYPERPARAMS[predictor_type]["dropout"] = v
            else:
                raise ValueError(f"ModelConfig has no config item: {k}")

class SwitchConfig:
    use_kg_emb: bool = True
    use_text_emb: bool = True
    use_fusion_emb: bool = True

    save_best_model: bool = True
    save_checkpoint: bool = False
    log_training_metrics: bool = True
    save_test_results: bool = True

    validate_during_train: bool = True
    early_stopping: bool = True

    @classmethod
    def update(cls,** kwargs):
        for k, v in kwargs.items():
            if hasattr(cls, k):
                setattr(cls, k, v)
            else:
                raise ValueError(f"SwitchConfig has no config item: {k}")

class OutputConfig:
    BASE_OUTPUT_DIR: str = "output"
    MODEL_BASE_DIR: str = os.path.join(BASE_OUTPUT_DIR, "ctr_models")
    BEST_MODEL_NAME_TEMPLATE: str = "best_{text_emb_version}_{kg_flag}_gat_{fusion_method}_{predictor}.pt"
    CHECKPOINT_NAME_TEMPLATE: str = "checkpoint_{epoch}.pt"

    LOG_BASE_DIR: str = os.path.join(BASE_OUTPUT_DIR, "ctr_logs")
    LOG_NAME_TEMPLATE: str = "{domain}_{fusion_method}_{predictor}_{text_emb_version}_{timestamp}.log"

    RESULT_BASE_DIR: str = os.path.join(BASE_OUTPUT_DIR, "ctr_results")
    RESULT_NAME_TEMPLATE: str = "{domain}_results_{fusion_method}_{predictor}_{timestamp}.csv"

    @classmethod
    def update(cls,** kwargs):
        for k, v in kwargs.items():
            if hasattr(cls, k):
                setattr(cls, k, v)
            else:
                raise ValueError(f"OutputConfig has no config item: {k}")

    def get_model_dir(self, domain: str) -> str:
        dir_path = os.path.join(self.MODEL_BASE_DIR, domain)
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    def get_log_dir(self, domain: str) -> str:
        dir_path = os.path.join(self.LOG_BASE_DIR, domain)
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    def get_result_dir(self, domain: str) -> str:
        dir_path = os.path.join(self.RESULT_BASE_DIR, domain)
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    def get_best_model_path(self, domain: str = None, text_emb_version: str = None, fusion_method: str = None,
                            predictor: str = None) -> str:
        domain = domain or DatasetConfig.DEFAULT_DOMAIN
        text_emb_version = text_emb_version or EmbeddingConfig.DEFAULT_TEXT_EMB_VERSION
        fusion_method = fusion_method or ModelConfig.DEFAULT_FUSION_METHOD
        predictor = predictor or ModelConfig.DEFAULT_PREDICTOR

        kg_flag = "kg" if SwitchConfig.use_kg_emb else "no_kg"
        model_name = self.BEST_MODEL_NAME_TEMPLATE.format(
            text_emb_version=text_emb_version,
            kg_flag=kg_flag,
            fusion_method=fusion_method,
            predictor=predictor
        )
        return os.path.join(self.get_model_dir(domain), model_name)

    def get_log_path(self, domain: str = None, fusion_method: str = None, predictor: str = None,
                     text_emb_version: str = None, timestamp: str = None) -> str:
        domain = domain or DatasetConfig.DEFAULT_DOMAIN
        fusion_method = fusion_method or ModelConfig.DEFAULT_FUSION_METHOD
        predictor = predictor or ModelConfig.DEFAULT_PREDICTOR
        text_emb_version = text_emb_version or EmbeddingConfig.DEFAULT_TEXT_EMB_VERSION
        timestamp = timestamp or TIMESTAMP

        log_name = self.LOG_NAME_TEMPLATE.format(
            domain=domain,
            fusion_method=fusion_method,
            predictor=predictor,
            text_emb_version=text_emb_version,
            timestamp=timestamp
        )
        return os.path.join(self.get_log_dir(domain), log_name)

    def get_result_path(self, domain: str = None, fusion_method: str = None, predictor: str = None,
                        timestamp: str = None) -> str:
        domain = domain or DatasetConfig.DEFAULT_DOMAIN
        fusion_method = fusion_method or ModelConfig.DEFAULT_FUSION_METHOD
        predictor = predictor or ModelConfig.DEFAULT_PREDICTOR
        timestamp = timestamp or TIMESTAMP

        result_name = self.RESULT_NAME_TEMPLATE.format(
            domain=domain,
            fusion_method=fusion_method,
            predictor=predictor,
            timestamp=timestamp
        )
        return os.path.join(self.get_result_dir(domain), result_name)

class TrainConfig:
    optimizer: str = "adamw"
    lr: float = 1e-5
    weight_decay: float = 1e-4
    lr_scheduler: str = "cosine"
    lr_step_size: int = 10
    lr_gamma: float = 0.1
    warmup_epochs: int = 2
    min_lr: float = 1e-6

    epochs: int = 50
    batch_size: int = 256
    eval_batch_size: int = 512
    num_workers: int = 4
    pin_memory: bool = True

    seed: int = 42

    loss_fn: str = "bce_with_logits"
    focal_gamma: float = 2.0
    pos_weight: float = 1.0
    eval_metrics: List[str] = ["auc"]

    early_stopping_metric: str = "auc"
    early_stopping_mode: str = "max"
    early_stopping_patience: int = 25
    save_dir: str = "./checkpoints"

    grad_clip_norm: float = 1.0
    grad_accum_steps: int = 2

    @classmethod
    def update(cls,** kwargs):
        for k, v in kwargs.items():
            if hasattr(cls, k):
                setattr(cls, k, v)
            else:
                raise ValueError(f"TrainConfig has no config item: {k}")

output_cfg = OutputConfig()

def validate_config() -> None:
    if DatasetConfig.DEFAULT_DOMAIN not in DatasetConfig.AVAILABLE_DOMAINS:
        raise ValueError(f"Default dataset domain {DatasetConfig.DEFAULT_DOMAIN} is not in available list")
    if EmbeddingConfig.DEFAULT_TEXT_EMB_VERSION not in EmbeddingConfig.AVAILABLE_TEXT_EMB_VERSIONS:
        raise ValueError(f"Default text embedding version {EmbeddingConfig.DEFAULT_TEXT_EMB_VERSION} is not in available list")
    if ModelConfig.DEFAULT_FUSION_METHOD not in ModelConfig.AVAILABLE_FUSION_METHODS:
        raise ValueError(f"Default fusion method {ModelConfig.DEFAULT_FUSION_METHOD} is not in available list")
    if ModelConfig.DEFAULT_PREDICTOR not in ModelConfig.AVAILABLE_PREDICTORS:
        raise ValueError(f"Default predictor {ModelConfig.DEFAULT_PREDICTOR} is not in available list")
    if GPUConfig.use_gpu and GPUConfig.gpu_ids is not None:
        for gpu_id in GPUConfig.gpu_ids:
            if not isinstance(gpu_id, int) or gpu_id < 0:
                raise ValueError(f"Invalid gpu id: {gpu_id}, must be non‑negative integer")
    if SwitchConfig.use_fusion_emb and not (SwitchConfig.use_kg_emb or SwitchConfig.use_text_emb):
        raise ValueError("When fusion embedding is enabled, at least one of KG embedding or text embedding must be enabled")

def validate_all_dataset_paths(
        check_label: bool = True,
        check_kg_emb: bool = True,
        check_text_emb: bool = True,
        text_emb_versions: List[str] = None
) -> Dict[str, Dict[str, List[str]]]:
    text_emb_versions = text_emb_versions or EmbeddingConfig.AVAILABLE_TEXT_EMB_VERSIONS
    result = {domain: {"missing": []} for domain in DatasetConfig.AVAILABLE_DOMAINS}

    print("\n===== Start validating all dataset paths =====")
    for domain in DatasetConfig.AVAILABLE_DOMAINS:
        print(f"\n📌 Validating dataset domain: {domain}")

        if check_label:
            for data_type in DatasetConfig.DATASET_TYPES:
                label_path = DatasetConfig.CTR_SAMPLE_PATH_TEMPLATE.format(domain=domain, data_type=data_type)
                if not os.path.exists(label_path):
                    result[domain]["missing"].append(f"Label file missing: {label_path}")
                    print(f"❌ {label_path}")
                else:
                    print(f"✅ {label_path}")

        if check_kg_emb:
            for data_type in DatasetConfig.DATASET_TYPES:
                kg_emb_path = EmbeddingConfig.KG_GAT_EMB_PATH_TEMPLATE.format(domain=domain, data_type=data_type)
                if not os.path.exists(kg_emb_path):
                    result[domain]["missing"].append(f"KG‑GAT embedding file missing: {kg_emb_path}")
                    print(f"❌ {kg_emb_path}")
                else:
                    print(f"✅ {kg_emb_path}")

        if check_text_emb:
            for data_type in DatasetConfig.DATASET_TYPES:
                for emb_version in text_emb_versions:
                    text_emb_path = EmbeddingConfig.TEXT_EMB_PATH_TEMPLATE.format(
                        domain=domain, data_type=data_type, emb_version=emb_version
                    )
                    if not os.path.exists(text_emb_path):
                        result[domain]["missing"].append(f"Text embedding file missing: {text_emb_path}")
                        print(f"❌ {text_emb_path}")
                    else:
                        print(f"✅ {text_emb_path}")

    total_missing = sum(len(v["missing"]) for v in result.values())
    print(f"\n===== Path validation finished =====\nTotal missing files: {total_missing}")
    if total_missing > 0:
        print("Missing file details:")
        for domain, res in result.items():
            if res["missing"]:
                print(f"  {domain}: {res['missing']}")
    else:
        print("✅ All dataset paths exist!")

    return result

def validate_output_paths() -> None:
    print("\n===== Validating output directories =====")
    for domain in DatasetConfig.AVAILABLE_DOMAINS:
        for dir_func in [output_cfg.get_model_dir, output_cfg.get_log_dir, output_cfg.get_result_dir]:
            dir_path = dir_func(domain)
            try:
                os.makedirs(dir_path, exist_ok=True)
                print(f"✅ Output directory is accessible: {dir_path}")
            except PermissionError:
                raise PermissionError(f"❌ Permission denied when creating output directory: {dir_path}")
