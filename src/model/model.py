"""
Заглушка модели для разработки.

Предоставляет объект `model` и функцию `get_model()` с минимальным интерфейсом,
который ожидает `src/services/vector_search.VectorSearchEngine`:

- model.encode_texts(texts: List[str]) -> np.ndarray (n, dim)
- model.get_sentence_embedding_dimension() -> int

Эта заглушка детерминирована (одно и то же входное слово -> одинаковый вектор)
и НЕ предназначена для использования в продакшне.
"""

from typing import List
import hashlib
import numpy as np

DIM = 384

class DummyModel:
    def __init__(self, dim: int = DIM):
        self._dim = dim

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Детерминированно преобразует список строк в L2-нормализованные векторы.

        Метод использует SHA-256 хеширование каждой строки с индексом
        вектора для получения псевдослучайных чисел — гарантия детерминизма
        при разработке/тестировании.
        """
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            # формируем значения поэлементно из хеша
            for j in range(self._dim):
                data = hashlib.sha256(f"{t}||{j}".encode("utf-8")).digest()
                # берем первые 4 байта как uint32 и нормируем в [0,1)
                val = int.from_bytes(data[:4], "big") / 2**32
                out[i, j] = float(val)
        # L2-нормализация
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        out = out / norms
        return out

    def encode_images(self, images: List[object]) -> np.ndarray:
        """Опциональная реализация: возвращаем нулевые векторы.
        Если нужна реальная поддержка изображений — заменить на подходящую реализацию.
        """
        return np.zeros((len(images), self._dim), dtype=np.float32)


# Экземпляр модели по умолчанию (импортируется как `from src.model.model import model`)
model = DummyModel()


def get_model() -> DummyModel:
    """Возвращает модель (совместимо с импорта get_model в сервисе)."""
    return model
