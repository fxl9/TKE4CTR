"""
GAT for CTR with per-sample knowledge graph.
One text block maps to one subgraph and one embedding, strict one-to-one index mapping.
"""
import json
import os
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from tqdm import tqdm
import argparse
import random
import pandas as pd
from transformers import BertTokenizer, BertModel
from torch.cuda.amp import autocast, GradScaler
import warnings

warnings.filterwarnings('ignore')


class GATConfig:
    KG_ROOT = "./datasets/amazon_review_data/kg_triples"
    DATASETS = ["All_Beauty", "Amazon_Fashion", "Digital_Music","Gift_Cards", "Musical_Instruments"]
    SPLITS = ["train", "valid", "test"]
    EMB_SAVE_ROOT = "./datasets/amazon_review_data/gnn_embedding"
    CTR_SAMPLE_PATH_TEMPLATE = os.path.join(
        "./datasets/amazon_review_data/data_splits",
        "{domain}", "{domain}_{data_type}.csv"
    )
    BERT_PATH = "./pretrained_models/bert-base-uncased/"

    ENTITY_TYPE_DIM = 32
    ENTITY_ID_BASE_DIM = 32
    TEXT_EMB_DIM = 32
    STRUCT_FEAT_DIM = 4
    NODE_FEAT_DIM = 32 + 64 + 4

    GAT_HEADS = 8
    GAT_HIDDEN_DIM = 16
    EMBEDDING_DIM = 128
    PREDICTION_DIM = 1
    DROPOUT_RATE = 0.15
    ALPHA = 0.2
    WEIGHT_DECAY = 1e-4

    CONTRAST_PRETRAIN_EPOCHS = 5
    CTR_TRAIN_EPOCHS = 10
    TRAIN_BATCH_SIZE = 512
    INFER_BATCH_SIZE = 512
    LEARNING_RATE = 1e-4
    WARMUP_RATIO = 0.1
    SAVE_INTERVAL = 5
    SEED = 42


def seed_everything(seed=GATConfig.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def init_gpu_device(gpu_id):
    if not torch.cuda.is_available():
        print("CUDA not available, use CPU")
        return torch.device("cpu")

    num_gpus = torch.cuda.device_count()
    print(f"\nAvailable GPU count: {num_gpus}")
    for i in range(num_gpus):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")

    if gpu_id < 0 or gpu_id >= num_gpus:
        print(f"\nInvalid GPU id {gpu_id}, switch to GPU 1")
        gpu_id = 1

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    try:
        test_tensor = torch.tensor([1.0]).to(device)
        print(f"\nSuccess init GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        print(f"   Device: cuda:{gpu_id} | Total memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024 / 1024 / 1024:.1f}GB")
        return device
    except Exception as e:
        print(f"\nGPU {gpu_id} init failed: {e}, switch to GPU 2")
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        torch.cuda.set_device(2)
        return torch.device("cuda:2")


def get_entity_type(ent):
    ent = str(ent)
    if ent.startswith("user_"):
        return "user"
    elif ent.startswith("product_"):
        return "product"
    return "other"


def hash_entity_id(ent, dim):
    try:
        id_part = str(ent).split("_")[-1] if "_" in str(ent) else str(ent)
        hash_obj = hashlib.sha256(id_part.encode())
        hash_hex = hash_obj.hexdigest()
        hash_vals = np.array([int(hash_hex[i:i + 2], 16) / 255.0 for i in range(0, min(len(hash_hex), dim * 2), 2)])
        return hash_vals[:dim] if len(hash_vals) >= dim else np.zeros(dim)
    except:
        return np.zeros(dim)


def get_relation_category_and_weight(rel):
    rel_weights = {
        "click_history_id": 2.0,
        "has_price": 1.0, "has_brand": 1.0, "has_title": 1.0, "belongs_to_category": 1.0, "has_style": 1.0,
        "has_rank": 1.0, "has_verified_status": 1.0,
        "also_bought_with": 1.5, "also_viewed_with": 1.5, "has_similar_item": 1.5,
        "has_username": 0.5, "belongs_to_scenario": 0.5
    }
    return rel_weights.get(str(rel).lower(), 0.5)


class BertTextEmbedding:
    def __init__(self, device):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(GATConfig.BERT_PATH)
        self.bert_model = BertModel.from_pretrained(GATConfig.BERT_PATH).to(device)
        self.bert_model.eval()
        self.projection = nn.Linear(768, GATConfig.TEXT_EMB_DIM).to(device)
        if device.type == "cuda":
            self.bert_model = self.bert_model.half()

    @torch.no_grad()
    def extract_embedding_batch(self, texts):
        texts = [t.strip() if t else "" for t in texts]
        inputs = self.tokenizer(
            texts, max_length=128, truncation=True, padding="max_length", return_tensors="pt"
        ).to(self.device)

        inputs["input_ids"] = inputs["input_ids"].long()
        inputs["attention_mask"] = inputs["attention_mask"].long()

        if self.device.type == "cuda":
            with autocast():
                outputs = self.bert_model(**inputs)
                cls_emb = outputs.last_hidden_state[:, 0, :].float()
        else:
            outputs = self.bert_model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :]

        emb = self.projection(cls_emb)
        return F.normalize(emb, p=2, dim=1).cpu().numpy()


def process_graph_batch(triples_batch, texts_batch, bert_emb, device):
    text_embs = bert_emb.extract_embedding_batch(texts_batch)
    batch_graph_data = []

    for idx, (triples, text_emb) in enumerate(zip(triples_batch, text_embs)):
        entities = list({t["head"] for t in triples} | {t["tail"] for t in triples})
        is_empty_graph = False
        if not entities:
            entities = [f"placeholder_{idx}"]
            is_empty_graph = True

        node2idx = {ent: i for i, ent in enumerate(entities)}
        num_nodes = len(entities)

        core_node_idx = 0
        for i, ent in enumerate(entities):
            if get_entity_type(ent) == "product":
                core_node_idx = i
                break

        node_feats = np.zeros((num_nodes, GATConfig.NODE_FEAT_DIM), dtype=np.float32)
        for i, ent in enumerate(entities):
            ent_type = get_entity_type(ent)
            if ent_type == "user":
                node_feats[i, :16] = 1.0
            elif ent_type == "product":
                node_feats[i, 16:32] = 1.0

            id_feat = hash_entity_id(ent, GATConfig.ENTITY_ID_BASE_DIM)
            node_feats[i, 32:64] = id_feat
            if node2idx[ent] == core_node_idx:
                node_feats[i, 64:96] = text_emb

        edges = []
        edge_weights = []
        for t in triples:
            if t["head"] in node2idx and t["tail"] in node2idx:
                src = node2idx[t["head"]]
                dst = node2idx[t["tail"]]
                edges.append([src, dst])
                edge_weights.append(get_relation_category_and_weight(t["relation"]))

        if len(edges) > 0:
            G = nx.DiGraph()
            G.add_nodes_from(range(num_nodes))
            G.add_edges_from(edges)

            max_degree = max(dict(G.degree()).values()) if G.degree() else 1
            degree_centrality = nx.degree_centrality(G)
            clustering = nx.clustering(G.to_undirected())

            for i in range(num_nodes):
                node_feats[i, 96] = G.in_degree(i) / max_degree
                node_feats[i, 97] = G.out_degree(i) / max_degree
                node_feats[i, 98] = clustering.get(i, 0.0)
                node_feats[i, 99] = degree_centrality.get(i, 0.0)

        node_feats = torch.from_numpy(node_feats)
        edges = torch.LongTensor(edges).t() if edges else torch.LongTensor([]).reshape(2, 0)
        edge_weights = torch.FloatTensor(edge_weights) if edge_weights else torch.FloatTensor([])

        batch_graph_data.append({
            "node_feats": node_feats,
            "edges": edges,
            "edge_weights": edge_weights,
            "core_node_idx": core_node_idx,
            "num_nodes": num_nodes,
            "is_empty_graph": is_empty_graph
        })

    batch_node_feats = []
    batch_edges = []
    batch_edge_weights = []
    batch_core_idxs = []
    batch_empty_mask = []
    node_offset = 0

    for data in batch_graph_data:
        batch_node_feats.append(data["node_feats"])
        if data["edges"].shape[1] > 0:
            batch_edges.append(data["edges"] + node_offset)
        batch_edge_weights.append(data["edge_weights"])
        batch_core_idxs.append(data["core_node_idx"] + node_offset)
        batch_empty_mask.append(data["is_empty_graph"])
        node_offset += data["num_nodes"]

    if device.type == "cuda":
        batch_node_feats = torch.cat(batch_node_feats, dim=0).to(device, non_blocking=True)
        batch_edges = torch.cat(batch_edges, dim=1).to(device, non_blocking=True) if batch_edges else torch.LongTensor([[]]).reshape(2, 0).to(device, non_blocking=True)
        batch_edge_weights = torch.cat(batch_edge_weights, dim=0).to(device, non_blocking=True) if batch_edge_weights else torch.FloatTensor([]).to(device, non_blocking=True)
        batch_core_idxs = torch.LongTensor(batch_core_idxs).to(device, non_blocking=True)
    else:
        batch_node_feats = torch.cat(batch_node_feats, dim=0).to(device)
        batch_edges = torch.cat(batch_edges, dim=1).to(device) if batch_edges else torch.LongTensor([[]]).reshape(2, 0).to(device)
        batch_edge_weights = torch.cat(batch_edge_weights, dim=0).to(device) if batch_edge_weights else torch.FloatTensor([]).to(device)
        batch_core_idxs = torch.LongTensor(batch_core_idxs).to(device)

    return {
        "node_feats": batch_node_feats,
        "edges": batch_edges,
        "edge_weights": batch_edge_weights,
        "core_idxs": batch_core_idxs,
        "empty_mask": batch_empty_mask
    }


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, heads=8, alpha=0.2, dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads

        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features * heads))
        self.a = nn.Parameter(torch.FloatTensor(1, heads, 2 * out_features))
        self.residual = nn.Linear(in_features, out_features * heads) if in_features != out_features * heads else nn.Identity()

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, node_feats, edges, edge_weights=None):
        N = node_feats.size(0)
        residual = self.residual(node_feats)

        h = torch.mm(node_feats, self.W)
        h = h.view(N, self.heads, self.out_features)

        if edges.size(1) == 0:
            out = h.mean(dim=1) + residual.view(N, self.heads, self.out_features).mean(dim=1)
            return out

        edge_src = edges[0]
        edge_dst = edges[1]
        h_src = h[edge_src]
        h_dst = h[edge_dst]

        a_input = torch.cat([h_src, h_dst], dim=-1)
        e = self.leakyrelu(torch.sum(self.a * a_input, dim=-1))

        if edge_weights is not None and edge_weights.numel() > 0:
            e = e * edge_weights.unsqueeze(1)

        out = h.new_zeros((N, self.heads, self.out_features))
        for head in range(self.heads):
            e_h = e[:, head]
            dst_nodes = edge_dst
            exp_e = torch.exp(e_h)
            sum_exp = torch.zeros(N, device=h.device).scatter_add_(0, dst_nodes, exp_e)
            alpha_h = exp_e / (sum_exp[dst_nodes] + 1e-8)
            alpha_h = self.dropout(alpha_h)
            out[:, head, :] = out[:, head, :].scatter_add(0, dst_nodes.unsqueeze(-1).expand(-1, self.out_features),
                                                           alpha_h.unsqueeze(-1) * h_src[:, head, :])

        out = out.view(N, -1) + residual
        return out


class GATModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gat1 = GraphAttentionLayer(
            GATConfig.NODE_FEAT_DIM,
            GATConfig.GAT_HIDDEN_DIM,
            GATConfig.GAT_HEADS,
            GATConfig.ALPHA,
            GATConfig.DROPOUT_RATE
        )
        self.gat2 = GraphAttentionLayer(
            GATConfig.GAT_HIDDEN_DIM * GATConfig.GAT_HEADS,
            GATConfig.EMBEDDING_DIM,
            1,
            GATConfig.ALPHA,
            GATConfig.DROPOUT_RATE
        )

        self.bn1 = nn.BatchNorm1d(GATConfig.NODE_FEAT_DIM)
        self.bn2 = nn.BatchNorm1d(GATConfig.GAT_HIDDEN_DIM * GATConfig.GAT_HEADS)
        self.dropout = nn.Dropout(GATConfig.DROPOUT_RATE)
        self.relu = nn.ReLU()

        self.predictor = nn.Sequential(
            nn.Linear(GATConfig.EMBEDDING_DIM, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, batch_data, return_embedding=False):
        node_feats = batch_data["node_feats"]
        edges = batch_data["edges"]
        edge_weights = batch_data["edge_weights"]
        core_idxs = batch_data["core_idxs"]

        x = self.bn1(node_feats)
        x = self.dropout(x)

        x = self.gat1(x, edges, edge_weights)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.gat2(x, edges, edge_weights)
        x = self.relu(x)

        core_feats = x[core_idxs]
        core_feats = F.normalize(core_feats, p=2, dim=1)

        if return_embedding:
            return core_feats
        else:
            return self.predictor(core_feats)


class GATTrainer:
    def __init__(self, device):
        self.device = device
        self.model = GATModel().to(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=GATConfig.LEARNING_RATE,
            weight_decay=GATConfig.WEIGHT_DECAY
        )
        self.ctr_criterion = nn.BCEWithLogitsLoss()
        self.contrast_criterion = nn.CosineEmbeddingLoss(margin=0.3)
        self.scaler = GradScaler() if device.type == "cuda" else None
        self.bert_emb = BertTextEmbedding(device)
        self.scheduler = None

    def train_epoch(self, triples_list, texts_list, labels):
        self.model.train()
        total_loss = 0.0
        num_samples = len(labels)
        indices = np.random.permutation(num_samples)

        pbar = tqdm(range(0, num_samples, GATConfig.TRAIN_BATCH_SIZE), desc="Training")
        for start_idx in pbar:
            end_idx = min(start_idx + GATConfig.TRAIN_BATCH_SIZE, num_samples)
            batch_idx = indices[start_idx:end_idx]

            batch_triples = [triples_list[i] for i in batch_idx]
            batch_texts = [texts_list[i] for i in batch_idx]
            batch_labels = torch.FloatTensor(np.array(labels)[batch_idx]).unsqueeze(1).to(self.device)

            batch_data = process_graph_batch(batch_triples, batch_texts, self.bert_emb, self.device)

            if self.device.type == "cuda":
                with autocast():
                    pred = self.model(batch_data)
                    loss = self.ctr_criterion(pred, batch_labels)
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(batch_data)
                loss = self.ctr_criterion(pred, batch_labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            batch_loss = loss.item() * (end_idx - start_idx)
            total_loss += batch_loss
            avg_loss = total_loss / (start_idx + end_idx - start_idx)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{avg_loss:.4f}"})

        return total_loss / num_samples

    @torch.no_grad()
    def generate_embeddings(self, triples_list, texts_list):
        self.model.eval()
        all_embeddings = []
        num_samples = len(triples_list)

        pbar = tqdm(range(0, num_samples, GATConfig.INFER_BATCH_SIZE), desc="Generating Embeddings")
        for start_idx in pbar:
            end_idx = min(start_idx + GATConfig.INFER_BATCH_SIZE, num_samples)
            batch_triples = triples_list[start_idx:end_idx]
            batch_texts = texts_list[start_idx:end_idx]

            batch_data = process_graph_batch(batch_triples, batch_texts, self.bert_emb, self.device)

            if self.device.type == "cuda":
                with autocast():
                    embeddings = self.model(batch_data, return_embedding=True)
            else:
                embeddings = self.model(batch_data, return_embedding=True)

            all_embeddings.extend(embeddings.cpu().numpy())
            pbar.set_postfix({"processed": f"{end_idx}/{num_samples}"})

        embeddings_np = np.array(all_embeddings)
        assert len(triples_list) == embeddings_np.shape[0], "sample embedding count mismatch"
        print(f"sample count:{len(triples_list)}, embedding shape:{embeddings_np.shape}, one-to-one check pass")
        return embeddings_np


def main(gpu_id=1):
    device = init_gpu_device(gpu_id)
    seed_everything()

    trainer = GATTrainer(device)

    for dataset in GATConfig.DATASETS:
        print(f"\nProcess dataset: {dataset}")
        save_dir = os.path.join(GATConfig.EMB_SAVE_ROOT, dataset)
        os.makedirs(save_dir, exist_ok=True)

        try:
            train_kg_path = os.path.join(GATConfig.KG_ROOT, dataset, "train_triples.json")
            with open(train_kg_path, "r", encoding="utf-8") as f:
                train_kg_data = json.load(f)

            train_triples = [item["triples"] for item in train_kg_data["triples_list"]]
            train_texts = [item.get("text_content", "") for item in train_kg_data["triples_list"]]

            train_ctr_path = GATConfig.CTR_SAMPLE_PATH_TEMPLATE.format(domain=dataset, data_type="train")
            train_df = pd.read_csv(train_ctr_path)
            train_labels = train_df["label"].tolist()

            max_len = min(len(train_triples), len(train_labels))
            train_triples = train_triples[:max_len]
            train_texts = train_texts[:max_len]
            train_labels = train_labels[:max_len]

            print(f"Load complete: {max_len} samples")

            print(f"\nStart training for {GATConfig.CTR_TRAIN_EPOCHS} epochs")
            for epoch in range(GATConfig.CTR_TRAIN_EPOCHS):
                avg_loss = trainer.train_epoch(train_triples, train_texts, train_labels)
                print(f"Epoch {epoch + 1}/{GATConfig.CTR_TRAIN_EPOCHS} | Avg loss: {avg_loss:.4f}")

                if (epoch + 1) % GATConfig.SAVE_INTERVAL == 0:
                    model_path = os.path.join(save_dir, f"gat_model_epoch{epoch + 1}.pth")
                    torch.save(trainer.model.state_dict(), model_path)
                    print(f"Model saved to: {model_path}")

            for split in GATConfig.SPLITS:
                print(f"\nGenerate {split} set embeddings")
                kg_path = os.path.join(GATConfig.KG_ROOT, dataset, f"{split}_triples.json")
                with open(kg_path, "r", encoding="utf-8") as f:
                    kg_data = json.load(f)

                triples = [item["triples"] for item in kg_data["triples_list"]]
                texts = [item.get("text_content", "") for item in kg_data["triples_list"]]

                embeddings = trainer.generate_embeddings(triples, texts)

                save_path = os.path.join(save_dir, f"{split}_gat_embeddings.npy")
                np.save(save_path, embeddings)
                print(f"{split} embedding saved: {save_path} | shape: {embeddings.shape}")

                if device.type == "cuda":
                    torch.cuda.empty_cache()

        except Exception as e:
            print(f"Process {dataset} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nAll datasets finished")
    if device.type == "cuda":
        used_mem = torch.cuda.memory_allocated(device) / 1024 / 1024 / 1024
        total_mem = torch.cuda.get_device_properties(device).total_memory / 1024 / 1024 / 1024
        print(f"GPU memory usage: {used_mem:.2f}GB / {total_mem:.2f}GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1, help='GPU ID')
    args = parser.parse_args()
    main(args.gpu)
