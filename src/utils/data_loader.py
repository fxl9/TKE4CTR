# -*- coding: utf-8 -*-
"""
CTR dataset module.
Implement custom PyTorch Dataset, data loading pipeline for kg embedding and text embedding.
Produce train/val/test dataloaders according to global config.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union

sys.path.insert(0, "./")

from configs.config import (
    DatasetConfig, EmbeddingConfig, SwitchConfig, TrainConfig,
    validate_config
)


class CTRDataset(Dataset):
    def __init__(self, label_data: pd.DataFrame, kg_emb: Optional[np.ndarray] = None,
                 text_emb: Optional[np.ndarray] = None):
        self.labels = torch.tensor(label_data["label"].values, dtype=torch.float32)
        self.kg_emb = torch.tensor(kg_emb, dtype=torch.float32) if kg_emb is not None else None
        self.text_emb = torch.tensor(text_emb, dtype=torch.float32) if text_emb is not None else None

        if self.kg_emb is not None:
            assert len(self.labels) == len(self.kg_emb), f"KG embedding count({len(self.kg_emb)}) mismatch with label count({len(self.labels)})"
        if self.text_emb is not None:
            assert len(self.labels) == len(self.text_emb), f"Text embedding count({len(self.text_emb)}) mismatch with label count({len(self.labels)})"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {"labels": self.labels[idx]}
        if self.kg_emb is not None:
            item["kg_emb"] = self.kg_emb[idx]
        if self.text_emb is not None:
            item["text_emb"] = self.text_emb[idx]
        return item


def load_single_type_data(domain: str, data_type: str, text_emb_version: str = None) -> Tuple[
    pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray]]:
    text_emb_version = text_emb_version or EmbeddingConfig.DEFAULT_TEXT_EMB_VERSION

    label_path = DatasetConfig.CTR_SAMPLE_PATH_TEMPLATE.format(domain=domain, data_type=data_type)
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file missing: {label_path}")
    label_data = pd.read_csv(label_path)

    kg_emb = None
    if SwitchConfig.use_kg_emb:
        kg_emb_path = EmbeddingConfig.KG_GAT_EMB_PATH_TEMPLATE.format(domain=domain, data_type=data_type)
        if not os.path.exists(kg_emb_path):
            raise FileNotFoundError(f"KG embedding file missing: {kg_emb_path}")
        kg_emb = np.load(kg_emb_path)
        if len(kg_emb) != len(label_data):
            raise ValueError(f"{domain}-{data_type} KG embedding({len(kg_emb)}) mismatch with label({len(label_data)})")

    text_emb = None
    if SwitchConfig.use_text_emb:
        text_emb_path = EmbeddingConfig.TEXT_EMB_PATH_TEMPLATE.format(domain=domain, data_type=data_type,
                                                                      emb_version=text_emb_version)
        if not os.path.exists(text_emb_path):
            raise FileNotFoundError(f"Text embedding file missing: {text_emb_path}")
        text_emb = np.load(text_emb_path)
        if len(text_emb) != len(label_data):
            raise ValueError(f"{domain}-{data_type} text embedding({len(text_emb)}) mismatch with label({len(label_data)})")

    return label_data, kg_emb, text_emb


def load_ctr_data(domain: str = None, text_emb_version: str = None) -> Dict[str, DataLoader]:
    validate_config()
    domain = domain or DatasetConfig.DEFAULT_DOMAIN
    text_emb_version = text_emb_version or EmbeddingConfig.DEFAULT_TEXT_EMB_VERSION

    if SwitchConfig.use_fusion_emb and not (SwitchConfig.use_kg_emb and SwitchConfig.use_text_emb):
        raise ValueError("Fusion mode requires both kg embedding and text embedding enabled")
    if not SwitchConfig.use_kg_emb and not SwitchConfig.use_text_emb:
        raise ValueError("At least one of kg embedding or text embedding must be enabled")

    dataloaders = {}
    for data_type in DatasetConfig.DATASET_TYPES:
        label_data, kg_emb, text_emb = load_single_type_data(domain, data_type, text_emb_version)

        dataset = CTRDataset(label_data, kg_emb, text_emb)

        is_train = (data_type == "train")
        dataloader = DataLoader(
            dataset,
            batch_size=TrainConfig.batch_size if is_train else TrainConfig.eval_batch_size,
            shuffle=is_train,
            num_workers=TrainConfig.num_workers,
            pin_memory=TrainConfig.pin_memory
        )
        dataloaders[data_type] = dataloader

        emb_info = []
        if kg_emb is not None:
            emb_info.append(f"KG({kg_emb.shape[-1]} dim)")
        if text_emb is not None:
            emb_info.append(f"Text({text_emb.shape[-1]} dim)")
        print(f"  {data_type} | sample count: {len(dataset)} | embedding: {', '.join(emb_info)}")

    return dataloaders
