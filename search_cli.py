import os
import json
import argparse
from typing import Any, Dict, List, Optional

import yaml
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

try:
    from IPython.display import display as ipy_display  # type: ignore

    IN_NOTEBOOK = True
except Exception:
    IN_NOTEBOOK = False


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_and_processor(
    repo_id: str,
    subfolder: Optional[str],
    processor_repo: str,
    fine_tuned_path: Optional[str] = None,
):
    # If fine_tuned_path is provided, use it (this is the trained model)
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
    # If subfolder is None or empty, load directly from repo_id
    elif not subfolder:
        model = CLIPModel.from_pretrained(repo_id)
        try:
            processor = CLIPProcessor.from_pretrained(repo_id)
        except Exception:
            processor = CLIPProcessor.from_pretrained(processor_repo)
    else:
        # Download model weights and config from the subfolder, then load locally
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


def load_index(idx_path: str):
    import faiss  # type: ignore

    return faiss.read_index(idx_path)


def query(
    model: CLIPModel, processor: CLIPProcessor, text: str, device: torch.device
) -> np.ndarray:
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
    return text_features.cpu().numpy().astype("float32")


def show_results(
    ds,
    indices: List[int],
    scores: List[float],
    image_col: str,
    text_col: str,
    save_dir: Optional[str] = None,
) -> None:
    """Show search results with images and text. Optionally save images to directory."""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for rank, (idx, score) in enumerate(zip(indices, scores), 1):
        row = ds[int(idx)]
        img = row[image_col]
        txt = row.get(text_col, "")

        print(f"\n{'='*60}")
        print(f"#{rank} | Score: {score:.4f} | Row: {idx}")
        print(f"Text: {txt[:200]}{'...' if len(txt) > 200 else ''}")

        # Show image (inline in notebooks if available, otherwise via default viewer)
        if isinstance(img, Image.Image):
            if IN_NOTEBOOK:
                ipy_display(img)
            else:
                img.show()
            if save_dir:
                save_path = os.path.join(save_dir, f"result_{rank}_row_{idx}.jpg")
                img.save(save_path)
                print(f"Saved to: {save_path}")
        else:
            print(f"Image type: {type(img)}")


def main(
    cfg_path: str, prompt: str, top_k: int, save_dir: Optional[str] = None
) -> None:
    cfg = load_config(cfg_path)

    mdl_cfg = cfg["model"]
    idx_cfg = cfg["index"]
    ds_cfg = cfg["dataset"]

    # Load model
    print("Loading model...")
    model, processor = load_model_and_processor(
        mdl_cfg["repo_id"],
        mdl_cfg.get("subfolder"),
        mdl_cfg.get("processor_repo"),
        mdl_cfg.get("fine_tuned_path"),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load index and metadata
    print("Loading index...")
    idx_path = os.path.join(idx_cfg["out_dir"], idx_cfg["index_file"])
    meta_path = os.path.join(idx_cfg["out_dir"], idx_cfg["meta_file"])
    index = load_index(idx_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_lines = [json.loads(line) for line in f]

    # Load dataset
    print("Loading dataset...")
    if ds_cfg.get("local_path"):
        ds = load_dataset(
            ds_cfg["local_path"], split=ds_cfg["split"]
        )  # supports local HF datasets
    else:
        ds = load_dataset(ds_cfg["name"], split=ds_cfg["split"])

    # Query
    print(f"Searching for: '{prompt}'...")
    qvec = query(model, processor, prompt, device)
    scores, idxs = index.search(qvec, top_k)
    idxs = idxs[0].tolist()
    scores_list = scores[0].tolist()

    print(f"\n{'='*60}")
    print(f"Top-{top_k} results for: '{prompt}'")
    print(f"{'='*60}")
    show_results(
        ds, idxs, scores_list, ds_cfg["image_column"], ds_cfg["text_column"], save_dir
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Search memes by text query")
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--query", type=str, required=True, help="Text query to search for")
    p.add_argument("--top_k", type=int, default=10, help="Number of results to return")
    p.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save result images (optional)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.query, args.top_k, args.save_dir)
