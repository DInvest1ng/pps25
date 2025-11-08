"""
Usage:
    python build_dataset_with_images_clean.py \
        --images_dir /path/to/images_pub \
        --jsons_dir /path/to/jsons_pub \
        --out dataset.csv

Produces CSV with columns:
    image (base64), description, width, height, image_link
"""

import json
import base64
import re
from pathlib import Path
import argparse
from typing import Optional, Dict, Any, List
import pandas as pd

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".gif"]


class VKDatasetBuilder:
    def __init__(self, config):
        self.config = config
        self.images_dir = Path(config.images_dir)
        self.jsons_dir = Path(config.jsons_dir)
        self.out_path = Path(config.out)

    def encode_image_to_base64(self, image_path: Path) -> Optional[str]:
        """Конвертирование изображение в base-64."""
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return None

    def clean_description(self, text: str) -> str:
        """Удаление лишних символов в описании."""
        if not text:
            return ""
        text = re.sub(r"[\n\r\t]+", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    def find_image_file(self, image_name: Optional[str], post_id: Optional[str]) -> Optional[Path]:
        """Поиск изображения в директории"""
        if image_name:
            p = self.images_dir / image_name
            if p.exists():
                return p
            if not Path(image_name).suffix:
                for ext in IMAGE_EXTS:
                    p2 = self.images_dir / f"{image_name}{ext}"
                    if p2.exists():
                        return p2

        if post_id:
            for ext in IMAGE_EXTS:
                p = self.images_dir / f"{post_id}{ext}"
                if p.exists():
                    return p
            matches = list(self.images_dir.glob(f"{post_id}.*"))
            if matches:
                return matches[0]

        return None

    def get_size_from_image(self, path: Path):
        """Получение размеров изображения"""
        if not PIL_AVAILABLE:
            return None, None
        try:
            with Image.open(path) as im:
                return im.width, im.height
        except Exception:
            return None, None

    def normalize_item(self, item: Dict[str, Any], json_path_stem: Optional[str]) -> Dict[str, Any]:
        """Получение данных из json-файла, сбор данных."""
        post_id = str(item.get("post_id") or item.get("id") or json_path_stem or "")
        image_name = item.get("image_name") or item.get("image") or item.get("filename")
        description = self.clean_description((item.get("description") or item.get("text") or "").strip())
        width = item.get("width")
        height = item.get("height")
        image_link = item.get("image_link") or item.get("main_img") or ""

        image_file = self.find_image_file(image_name, post_id)
        image_b64 = ""
        if image_file and image_file.exists():
            image_b64 = self.encode_image_to_base64(image_file)
            if (width is None or height is None):
                w, h = self.get_size_from_image(image_file)
                if w and h:
                    width = width or w
                    height = height or h

        return {
            "image": image_b64,
            "description": description,
            "width": width if width is not None else "",
            "height": height if height is not None else "",
            "image_link": image_link
        }

    def process_json_file(self, json_path: Path) -> List[Dict[str, Any]]:
        """Получение данных из json-файла."""
        rows = []
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            return rows

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    rows.append(self.normalize_item(item, json_path.stem))
        elif isinstance(data, dict):
            if "post_id" in data or "image_name" in data:
                rows.append(self.normalize_item(data, json_path.stem))
            else:
                for k, v in data.items():
                    if isinstance(v, dict):
                        v["post_id"] = k
                        rows.append(self.normalize_item(v, k))
        return rows

    def build(self):
        """Основной метод для построения датасета"""
        all_rows = []

        json_paths = list(self.jsons_dir.glob("**/*.json"))
        for jp in json_paths:
            all_rows.extend(self.process_json_file(jp))

        df = pd.DataFrame(all_rows)
        df.to_csv(self.out_path, index=False, encoding="utf-8")
        print(f"Dataset saved to {self.out_path} — {len(df)} rows")
        return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True, help="Path to images_{pub_name}")
    parser.add_argument("--jsons_dir", required=True, help="Path to jsons_{pub_name}")
    parser.add_argument("--out", default="dataset.csv", help="Output CSV file")
    cfg = parser.parse_args()
    
    builder = VKDatasetBuilder(cfg)
    builder.build()


if __name__ == "__main__":
    main()