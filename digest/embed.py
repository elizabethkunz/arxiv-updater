import hashlib
import re

import joblib
import numpy as np

from .data import STATE

_model = None


def _cache_path(model_name: str):
    d = STATE / "emb_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d / (re.sub(r"\W+", "_", model_name) + ".joblib")


def embed(texts: list[str], model_name: str) -> np.ndarray:
    """L2-normalised embeddings, cached on disk by text hash."""
    global _model
    path = _cache_path(model_name)
    cache = joblib.load(path) if path.exists() else {}
    keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
    missing = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in cache]
    if missing:
        if _model is None:
            from fastembed import TextEmbedding
            _model = TextEmbedding(model_name)
        print(f"  embedding {len(missing)} texts with {model_name} ...")
        vectors = _model.embed([t for _, t in missing], batch_size=16)
        for n, ((k, _), v) in enumerate(zip(missing, vectors), 1):
            cache[k] = (v / np.linalg.norm(v)).astype(np.float32)
            if n % 500 == 0 or n == len(missing):  # checkpoint so an interrupted run resumes
                joblib.dump(cache, path)
    return np.stack([cache[k] for k in keys])
