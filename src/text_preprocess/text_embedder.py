# -*- coding: utf-8 -*-
"""
Generate PLM‑encoded text embeddings.
Supported models: bert-base-uncased, deberta-v3-large, gpt2, Qwen3-4b, Sheared-LLaMA-1.3B, Hunyuan-4B-Instruct.
"""
import os
import re
import torch
import numpy as np
import pickle
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from functools import partial


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class PromptDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, tokenizer, max_seq_len):
    outputs = tokenizer(
        batch,
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt"
    )

    if outputs["attention_mask"].dim() == 1:
        outputs["attention_mask"] = outputs["attention_mask"].unsqueeze(0)
    elif outputs["attention_mask"].dim() > 2:
        outputs["attention_mask"] = outputs["attention_mask"].view(-1, max_seq_len)

    return outputs


class PromptEmbeddingGenerator:
    def __init__(self, model_rel_name_list, data_rel_dir, output_rel_dir, max_seq_len=256, use_distributed=False):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        self.base_dir = os.path.abspath(os.path.dirname(__file__))
        self.project_root = os.path.abspath(os.path.join(self.base_dir, "../.."))

        self.model_rel_name_list = model_rel_name_list
        self.data_dir = os.path.abspath(os.path.join(self.project_root, data_rel_dir))
        self.output_dir = os.path.abspath(os.path.join(self.project_root, output_rel_dir))
        self.max_seq_len = max_seq_len
        self.use_distributed = use_distributed

        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.process_index = self.accelerator.process_index
        self.num_processes = self.accelerator.num_processes

        self.init_distributed()

        if self.process_index == 0:
            print(f"Mode: {'Single GPU' if self.num_processes == 1 else 'Distributed'}, Device: {self.device}")

        pretrained_root = os.path.join(self.project_root, "pretrained_models")
        for model_name in self.model_rel_name_list:
            full_model_path = os.path.join(pretrained_root, model_name)
            if not os.path.exists(full_model_path) and self.process_index == 0:
                raise FileNotFoundError(f"Model not found: {full_model_path}")
        if not os.path.exists(self.data_dir) and self.process_index == 0:
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        self.current_model = None
        self.current_tokenizer = None

    def init_distributed(self):
        if self.use_distributed and dist.is_available() and not dist.is_initialized() and self.num_processes > 1:
            try:
                dist.init_process_group(
                    backend='nccl',
                    rank=self.process_index,
                    world_size=self.num_processes
                )
            except Exception as e:
                print(f"Distributed init warning: {e}")
                self.use_distributed = False

    def broadcast_object(self, obj):
        if not self.use_distributed or self.num_processes == 1:
            return obj
        try:
            if dist.is_available() and dist.is_initialized():
                if self.process_index == 0:
                    buffer_data = pickle.dumps(obj)
                    buffer = torch.ByteTensor(torch.frombuffer(buffer_data, dtype=torch.uint8)).to(self.device)
                    size_tensor = torch.tensor([buffer.numel()], dtype=torch.int64, device=self.device)
                    dist.broadcast(size_tensor, src=0)
                    dist.broadcast(buffer, src=0)
                    return obj
                else:
                    size_tensor = torch.tensor([0], dtype=torch.int64, device=self.device)
                    dist.broadcast(size_tensor, src=0)
                    buffer = torch.empty(size_tensor.item(), dtype=torch.uint8, device=self.device)
                    dist.broadcast(buffer, src=0)
                    return pickle.loads(buffer.cpu().numpy().tobytes())
        except:
            pass
        return obj

    def _load_single_model(self, model_short_name):
        if self.current_model is not None:
            del self.current_model
            del self.current_tokenizer
            torch.cuda.empty_cache()

        abs_model_path = os.path.join(self.project_root, "pretrained_models", model_short_name)
        try:
            tokenizer = AutoTokenizer.from_pretrained(abs_model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"

            if model_short_name in ["bert-base-uncased", "deberta-v3-large"]:
                model = AutoModel.from_pretrained(
                    abs_model_path,
                    device_map={"": self.device},
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    abs_model_path,
                    device_map={"": self.device},
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
                model = model.model if hasattr(model, 'model') else model

            model.eval()
            model, tokenizer = self.accelerator.prepare(model, tokenizer)

            self.current_model = model
            self.current_tokenizer = tokenizer

            if self.process_index == 0:
                print(f"Loaded model: {abs_model_path}")
                print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

            return True

        except Exception as e:
            if self.process_index == 0:
                print(f"Failed to load {abs_model_path}: {str(e)[:150]}...")
            return False

    def split_prompts(self, raw_text, domain_name, stage):
        lines = re.split(r'\r?\n', raw_text)
        chunks = [line.strip() for line in lines if line.strip()]

        if self.process_index == 0:
            print(f"Domain: {domain_name}, Stage: {stage}, valid prompt count: {len(chunks)}")

        if len(chunks) == 0:
            chunks = [raw_text.strip()]
        return chunks

    def load_and_split(self, domain, stage):
        prompt_file_name = f"{domain}_{stage}_text.txt"
        prompt_file_path = os.path.join(self.data_dir, domain, prompt_file_name)

        if not os.path.exists(prompt_file_path):
            raise FileNotFoundError(f"Prompt file not found: {prompt_file_path}")

        encodings = ['utf-8', 'latin-1', 'utf-16']
        raw_text = None
        for encoding in encodings:
            try:
                with open(prompt_file_path, "r", encoding=encoding) as f:
                    raw_text = f.read()
                break
            except UnicodeDecodeError:
                continue
        else:
            raw_text = None

        if raw_text is None:
            split_samples = []
        else:
            split_samples = self.split_prompts(raw_text, domain, stage)

        split_samples = self.broadcast_object(split_samples)
        return split_samples

    def generate_embeddings(self, samples):
        if self.current_model is None or self.current_tokenizer is None:
            return None

        model = self.current_model
        tokenizer = self.current_tokenizer
        all_embeddings = []

        batch_size = 16

        dataset = PromptDataset(samples)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=partial(collate_fn, tokenizer=tokenizer, max_seq_len=self.max_seq_len),
            shuffle=False
        )
        dataloader = self.accelerator.prepare(dataloader)

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generate embedding", disable=self.process_index != 0)):
                torch.cuda.empty_cache()
                mask = batch["attention_mask"]

                if mask.dim() == 1:
                    batch["attention_mask"] = mask.unsqueeze(0)
                elif mask.dim() > 2:
                    batch["attention_mask"] = mask.view(-1, self.max_seq_len)
                final_mask = batch["attention_mask"]

                outputs = model(**batch)

                def get_last_non_pad(mask_tensor):
                    bsz = mask_tensor.size(0)
                    last_non_pad_pos = []
                    for b in range(bsz):
                        non_pad_idx = (mask_tensor[b] == 1).nonzero().squeeze()
                        if non_pad_idx.numel() == 0:
                            last_non_pad_pos.append(self.max_seq_len - 1)
                        else:
                            last_non_pad_pos.append(non_pad_idx[-1].item() if non_pad_idx.dim() > 0 else non_pad_idx.item())
                    return torch.tensor(last_non_pad_pos, device=self.device, dtype=torch.long)

                last_non_pad = get_last_non_pad(final_mask)
                vec = outputs.last_hidden_state[range(len(last_non_pad)), last_non_pad, :]

                if self.process_index == 0:
                    batch_embeds = vec.cpu().float().numpy()
                    batch_embeds = batch_embeds / (np.linalg.norm(batch_embeds, axis=1, keepdims=True) + 1e-8)
                    all_embeddings.append(batch_embeds)

        if self.process_index == 0:
            return np.concatenate(all_embeddings, axis=0) if all_embeddings else None
        return None

    def process_domain_stage(self, domain, stage, model_short_name):
        try:
            split_samples = self.load_and_split(domain, stage)
        except Exception as e:
            if self.process_index == 0:
                print(f"Skip {domain}-{stage}: {e}")
            return

        if not split_samples:
            return

        embeddings = self.generate_embeddings(split_samples)

        if self.process_index == 0 and embeddings is not None:
            save_dir = os.path.join(self.output_dir, domain, stage, model_short_name)
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, "embeddings.npy"), embeddings)
            print(f"Saved embedding: {save_dir} | shape={embeddings.shape}")
            del embeddings
            torch.cuda.empty_cache()

    def process_all(self, domains, stages):
        for model_short in self.model_rel_name_list:
            if self.process_index == 0:
                print(f"\n===== Process model: {model_short} =====")
            if not self._load_single_model(model_short):
                continue

            for domain in domains:
                for stage in stages:
                    if self.process_index == 0:
                        print(f"\n----- {domain} - {stage} -----")
                    self.process_domain_stage(domain, stage, model_short)

            self.current_model = None
            self.current_tokenizer = None
            torch.cuda.empty_cache()


def main():
    model_rel_name_list = [
        "bert-base-uncased",
        "deberta-v3-large",
        "gpt2",
        "Qwen3-4b",
        "Sheared-LLaMA-1.3B",
        "Hunyuan-4B-Instruct"
    ]

    data_rel_dir = r"datasets/amazon_review_data/data_text"
    output_rel_dir = r"datasets/amazon_review_data/text_embedding"

    embedding_generator = PromptEmbeddingGenerator(
        model_rel_name_list=model_rel_name_list,
        data_rel_dir=data_rel_dir,
        output_rel_dir=output_rel_dir,
        max_seq_len=256,
        use_distributed=False
    )

    domains = ["All_Beauty", "Amazon_Fashion", "Digital_Music", "Gift_Cards", "Musical_Instruments"]
    stages = ["train", "valid", "test"]

    embedding_generator.process_all(domains, stages)


if __name__ == "__main__":
    main()
