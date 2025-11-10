import torch
import numpy as np
from typing import List, Union
from transformers import CLIPModel, AutoTokenizer, AutoProcessor
from PIL.Image import Image

class CLIPVectorizer:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: Union[str, None] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

    def _batched(self, items, batch_size):
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]

    def encode_text(self, texts: Union[str, List[str]], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        features_list = []
        with torch.inference_mode():
            for batch in self._batched(texts, batch_size):
                inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
                feats = self.model.get_text_features(**inputs)
                if normalize:
                    feats = torch.nn.functional.normalize(feats, p=2, dim=1)
                features_list.append(feats.cpu().float().numpy())
        return np.vstack(features_list)

    def encode_image(self, images: Union[Image, List[Image]], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        single = False
        if isinstance(images, Image):
            images = [images]
            single = True
        features_list = []
        with torch.inference_mode():
            for batch in self._batched(images, batch_size):
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                feats = self.model.get_image_features(**inputs)
                if normalize:
                    feats = torch.nn.functional.normalize(feats, p=2, dim=1)
                features_list.append(feats.cpu().float().numpy())
        return np.vstack(features_list)

if __name__ == "__main__":
    from PIL import Image
    import io

    vec = CLIPVectorizer()
    texts = ["a photo of a cat", "a photo of a dog"]
    text_embeddings = vec.encode_text(texts)
    print("text embeddings shape:", text_embeddings.shape)