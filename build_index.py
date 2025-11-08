import os
import json
import argparse
from typing import Any, Dict, Optional

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import yaml
import numpy as np
from tqdm import tqdm

from transformers import CLIPModel, CLIPProcessor


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_and_processor(
    repo_id: str,
    subfolder: Optional[str],
    processor_repo: str,
    fine_tuned_path: Optional[str] = None,
):
    if fine_tuned_path and os.path.exists(fine_tuned_path):
        print(f"Loading fine-tuned model from: {fine_tuned_path}")
        model = CLIPModel.from_pretrained(fine_tuned_path)
        try:
            processor = CLIPProcessor.from_pretrained(fine_tuned_path)
        except Exception:
            try:
                processor = CLIPProcessor.from_pretrained(repo_id)
            except Exception:
                processor = CLIPProcessor.from_pretrained(processor_repo)
    elif not subfolder:
        model = CLIPModel.from_pretrained(repo_id)
        try:
            processor = CLIPProcessor.from_pretrained(repo_id)
        except Exception:
            processor = CLIPProcessor.from_pretrained(processor_repo)
    else:
        _ = hf_hub_download(
            repo_id=repo_id, filename="config.json", subfolder=subfolder
        )
        model_path = hf_hub_download(
            repo_id=repo_id, filename="pytorch_model.bin", subfolder=subfolder
        )
        model_dir = os.path.dirname(model_path)
        model = CLIPModel.from_pretrained(model_dir)
        try:
            processor = CLIPProcessor.from_pretrained(repo_id)
        except Exception:
            processor = CLIPProcessor.from_pretrained(processor_repo)
    return model, processor


def get_dataset(cfg: Dict[str, Any]):
    ds_cfg = cfg["dataset"]
    if ds_cfg.get("local_path"):
        dataset = load_dataset(ds_cfg["local_path"], split=ds_cfg["split"])
    else:
        dataset = load_dataset(ds_cfg["name"], split=ds_cfg["split"])
    return dataset


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norm


def main(cfg_path: str) -> None:
    import faiss  # type: ignore

    cfg = load_config(cfg_path)
    ds_cfg = cfg["dataset"]
    idx_cfg = cfg["index"]
    mdl_cfg = cfg["model"]

    os.makedirs(idx_cfg["out_dir"], exist_ok=True)

    model, processor = load_model_and_processor(
        repo_id=mdl_cfg["repo_id"],
        subfolder=mdl_cfg.get("subfolder"),
        processor_repo=mdl_cfg.get("processor_repo"),
        fine_tuned_path=mdl_cfg.get("fine_tuned_path"),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU доступен: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU память: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )
    model.to(device)
    model.eval()

    ds = get_dataset(cfg)
    image_key = ds_cfg["image_column"]

    # Батчинг для более эффективной обработки
    batch_size = idx_cfg.get("batch_size", 32)

    all_embs = []
    meta_path = os.path.join(idx_cfg["out_dir"], idx_cfg["meta_file"])

    print(f"Обработка {len(ds)} изображений с батч-размером {batch_size}...")

    with open(meta_path, "w", encoding="utf-8") as meta_f:
        # Обработка батчами для экономии памяти и ускорения
        for batch_start in tqdm(
            range(0, len(ds), batch_size), desc="Обработка изображений"
        ):
            batch_end = min(batch_start + batch_size, len(ds))
            batch_rows = [ds[i] for i in range(batch_start, batch_end)]

            # Подготовка батча изображений
            batch_images = [row[image_key] for row in batch_rows]
            inputs = processor(images=batch_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Получение эмбеддингов
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                image_features = torch.nn.functional.normalize(
                    image_features, p=2, dim=-1
                )

            # Сохранение эмбеддингов и метаданных
            batch_embs = image_features.cpu().numpy().astype("float32")
            all_embs.append(batch_embs)

            for i, row_idx in enumerate(range(batch_start, batch_end)):
                meta = {"row": row_idx}
                meta_f.write(json.dumps(meta, ensure_ascii=False) + "\n")

            # Периодическая очистка памяти
            if device.type == "cuda" and batch_start % (batch_size * 10) == 0:
                torch.cuda.empty_cache()

    embs = np.concatenate(all_embs, axis=0)
    # cosine sim via inner product since embeddings are L2-normalized
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    index_path = os.path.join(idx_cfg["out_dir"], idx_cfg["index_file"])
    faiss.write_index(index, index_path)
    print(f"Index built: {index_path} with {embs.shape[0]} vectors")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build FAISS index of image embeddings")
    p.add_argument("--config", type=str, default="config.yaml")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
