import os
import math
import random
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from functools import partial

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import yaml
from tqdm import tqdm

from transformers import (
    CLIPModel,
    CLIPProcessor,
    get_cosine_schedule_with_warmup,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(mixed_precision: Optional[str]) -> torch.device:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return device


def load_model_and_processor(
    repo_id: str, subfolder: Optional[str], processor_repo: Optional[str]
) -> (CLIPModel, CLIPProcessor, str):
    # If subfolder is None or empty, load directly from repo_id
    if not subfolder:
        model = CLIPModel.from_pretrained(repo_id)
        model_dir = None
    else:
        _ = hf_hub_download(
            repo_id=repo_id, filename="config.json", subfolder=subfolder
        )
        model_path = hf_hub_download(
            repo_id=repo_id, filename="pytorch_model.bin", subfolder=subfolder
        )
        model_dir = os.path.dirname(model_path)
        model = CLIPModel.from_pretrained(model_dir)

    processor: Optional[CLIPProcessor] = None
    try:
        processor = CLIPProcessor.from_pretrained(repo_id)
    except Exception:
        if processor_repo:
            processor = CLIPProcessor.from_pretrained(processor_repo)
        else:
            raise

    return model, processor, model_dir if model_dir else repo_id


def get_dataset(cfg: Dict[str, Any]):
    ds_cfg = cfg["dataset"]
    if ds_cfg.get("local_path"):
        dataset = load_dataset(ds_cfg["local_path"], split=ds_cfg["split"])
    else:
        dataset = load_dataset(ds_cfg["name"], split=ds_cfg["split"])
    return dataset


class CollateFn:
    """Collate function that can be pickled for multiprocessing on Windows."""

    def __init__(self, processor: CLIPProcessor, image_key: str, text_key: str):
        self.processor = processor
        self.image_key = image_key
        self.text_key = text_key

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = [e[self.image_key] for e in examples]
        texts = [e[self.text_key] for e in examples]
        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return batch


@dataclass
class TrainState:
    epoch: int
    global_step: int


def train(cfg_path: str) -> None:
    cfg = load_config(cfg_path)

    ds_cfg = cfg["dataset"]
    tr_cfg = cfg["training"]
    model_cfg = cfg["model"]

    os.makedirs(tr_cfg["output_dir"], exist_ok=True)

    set_seed(tr_cfg.get("seed", 42))
    device = resolve_device(tr_cfg.get("mixed_precision"))

    model, processor, _ = load_model_and_processor(
        repo_id=model_cfg["repo_id"],
        subfolder=model_cfg["subfolder"],
        processor_repo=model_cfg.get("processor_repo"),
    )
    model.to(device)

    if device.type == "cuda" and tr_cfg.get("gradient_checkpointing", False):
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing включен для экономии памяти")

    model.train()

    dataset = get_dataset(cfg)

    collate_func = CollateFn(processor, ds_cfg["image_column"], ds_cfg["text_column"])

    num_workers = tr_cfg.get("num_workers", 0)
    if num_workers is None:
        num_workers = 2 if device.type == "cuda" else 0

    dl = DataLoader(
        dataset,
        batch_size=tr_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_func,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tr_cfg["learning_rate"],
        weight_decay=tr_cfg["weight_decay"],
    )
    total_steps = (
        tr_cfg["epochs"]
        * math.ceil(len(dataset) / tr_cfg["batch_size"])
        // max(1, tr_cfg.get("grad_accum_steps", 1))
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=tr_cfg["warmup_steps"],
        num_training_steps=total_steps,
    )

    scaler: Optional[torch.cuda.amp.GradScaler] = None
    use_fp16 = tr_cfg.get("mixed_precision") == "fp16" and device.type == "cuda"
    use_bf16 = (
        tr_cfg.get("mixed_precision") == "bf16"
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    if use_fp16:
        scaler = torch.cuda.amp.GradScaler()

    state = TrainState(epoch=0, global_step=0)
    best_loss = float("inf")

    print(f"\nНачало обучения:")
    print(f"  - Эпох: {tr_cfg['epochs']}")
    print(f"  - Размер батча: {tr_cfg['batch_size']}")
    print(f"  - Gradient accumulation: {tr_cfg.get('grad_accum_steps', 1)}")
    print(f"  - Всего шагов: {total_steps}")
    print(f"  - Mixed precision: {tr_cfg.get('mixed_precision', 'none')}")
    print(f"  - Устройство: {device}\n")

    for epoch in range(tr_cfg["epochs"]):
        state.epoch = epoch
        epoch_losses = []
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{tr_cfg['epochs']}")

        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(
                enabled=use_fp16, dtype=torch.bfloat16 if use_bf16 else None
            ):
                outputs = model(**batch, return_loss=True)
                loss = outputs.loss / max(1, tr_cfg.get("grad_accum_steps", 1))

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % max(1, tr_cfg.get("grad_accum_steps", 1)) == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                state.global_step += 1

            # Сохраняем loss для эпохи
            loss_value = loss.item() * max(1, tr_cfg.get("grad_accum_steps", 1))
            epoch_losses.append(loss_value)

            # Обновляем прогресс-бар
            pbar.set_postfix(
                {
                    "loss": f"{loss_value:.4f}",
                    "step": state.global_step,
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

            # Периодическая очистка кэша CUDA для экономии памяти
            if device.type == "cuda" and step % 100 == 0:
                torch.cuda.empty_cache()

        # Calculate average loss for epoch
        epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        pbar.close()

        # Очистка памяти перед сохранением
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Save checkpoint at epoch end
        save_dir = os.path.join(tr_cfg["output_dir"], f"epoch-{epoch+1}")
        os.makedirs(save_dir, exist_ok=True)
        print(f"\nСохранение чекпоинта в {save_dir}...")
        model.save_pretrained(save_dir)
        try:
            processor.save_pretrained(save_dir)
        except Exception:
            pass
        print(f"Чекпоинт сохранен (средний loss: {epoch_loss:.4f})")

        # Save best model if enabled
        if tr_cfg.get("save_best", False) and epoch_loss < best_loss:
            best_loss = epoch_loss
            best_dir = os.path.join(tr_cfg["output_dir"], "best")
            os.makedirs(best_dir, exist_ok=True)
            print(f"Сохранение лучшей модели в {best_dir}...")
            model.save_pretrained(best_dir)
            try:
                processor.save_pretrained(best_dir)
            except Exception:
                pass
            print(f"Лучшая модель сохранена (loss: {best_loss:.4f})")

        print(f"Эпоха {epoch+1} завершена. Средний loss: {epoch_loss:.4f}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finetune CLIP for meme retrieval")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to YAML config"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config)
