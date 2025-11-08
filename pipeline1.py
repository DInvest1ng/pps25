from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
from huggingface_hub import hf_hub_download
import torch
import os

# Загрузка датасета
print("Загрузка датасета foldl/rumeme-desc...")
ds = load_dataset("foldl/rumeme-desc", split="train")
print(f"Датасет загружен. Размер: {len(ds)} примеров")
print(f"Колонки датасета: {ds.column_names}")

# Загрузка предобученной модели ru-clip
# В репозитории есть несколько версий, используем ruclip-vit-base-patch32-v5
print("\nЗагрузка модели ai-forever/ru-clip...")
repo_id = "ai-forever/ru-clip"
subfolder = "ruclip-vit-base-patch32-v5"

# Скачиваем необходимые файлы модели
print("Скачивание файлов модели...")
config_path = hf_hub_download(
    repo_id=repo_id, filename="config.json", subfolder=subfolder
)
model_path = hf_hub_download(
    repo_id=repo_id, filename="pytorch_model.bin", subfolder=subfolder
)

# Загружаем модель из локального пути
model_dir = os.path.dirname(model_path)
model = CLIPModel.from_pretrained(model_dir)

# Для процессора используем совместимую модель OpenAI CLIP (архитектура совместима)
# или пытаемся загрузить из базового репозитория
from transformers import CLIPTokenizer, CLIPImageProcessor

try:
    # Пробуем загрузить из базового репозитория
    processor = CLIPProcessor.from_pretrained(repo_id)
except:
    # Если не получается, используем совместимый процессор от OpenAI
    print("Используется совместимый процессор от OpenAI CLIP...")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("Модель и процессор загружены успешно!")

# Информация о модели
print(f"\nИнформация о модели:")
print(f"  - Тип модели: {type(model).__name__}")
print(f"  - Устройство: {next(model.parameters()).device}")

# Проверка доступности GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nДоступное устройство: {device}")
if device == "cuda":
    model = model.to(device)
    print("Модель перемещена на GPU")

# Пример просмотра данных из датасета
print("\nПример данных из датасета:")
if len(ds) > 0:
    example = ds[0]
    print(f"  Первый пример: {example}")

print("\n[OK] Программа готова к работе!")
