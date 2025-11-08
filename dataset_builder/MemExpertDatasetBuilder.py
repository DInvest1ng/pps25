"""
Usage:
    python build_dataset.py \
        --images_dir ./images \
        --json_dir ./json_data \
        --out dataset.csv

Creates CSV with columns:
    image (base64), description, width, height
"""

import argparse
import base64
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import re

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]


class DatasetBuilder:
    def __init__(self, images_dir: Path, json_dir: Path, out_path: Path):
        self.images_dir = images_dir
        self.json_dir = json_dir
        self.out_path = out_path

    # --- Helpers ---
    def encode_image(self, path: Path) -> Optional[str]:
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return None

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"[\n\r\t]+", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    def get_image_size(self, path: Path) -> tuple[Optional[int], Optional[int]]:
        if not PIL_AVAILABLE:
            return None, None
        try:
            with Image.open(path) as im:
                return im.width, im.height
        except Exception:
            return None, None

    def find_image_for_json(self, name: str) -> Optional[Path]:
        stem = Path(name).stem
        for ext in IMAGE_EXTS:
            candidate = self.images_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        return None

    # --- Main logic ---
    def process_json_file(self, json_path: Path) -> Optional[Dict[str, Any]]:
        """Парсинг одного JSON и связывание с изображением"""
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            return None

        image_path = self.find_image_for_json(json_path.name)
        if not image_path:
            return None

        image_b64 = self.encode_image(image_path)
        width, height = self.get_image_size(image_path)
        description = self.clean_text(
            str(data.get("description") or data.get("text") or "")
        )

        return {
            "image": image_b64 or "",
            "description": description,
            "width": width or "",
            "height": height or "",
        }

    def build(self):
        rows: List[Dict[str, Any]] = []
        json_files = sorted(self.json_dir.glob("*.json"))

        for jp in json_files:
            item = self.process_json_file(jp)
            if item:
                rows.append(item)

        df = pd.DataFrame(rows)
        df.to_csv(self.out_path, index=False, encoding="utf-8")
        print(f"✅ Dataset saved: {self.out_path} ({len(df)} rows)")
        return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True, help="Path to images directory")
    parser.add_argument("--json_dir", required=True, help="Path to JSON directory")
    parser.add_argument("--out", default="dataset.csv", help="Output CSV path")
    args = parser.parse_args()

    builder = DatasetBuilder(Path(args.images_dir), Path(args.json_dir), Path(args.out))
    builder.build()


if __name__ == "__main__":
    main()
