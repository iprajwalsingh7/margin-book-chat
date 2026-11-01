"""Local semantic indexing and book-scoped retrieval."""
from pathlib import Path
import os
import threading

import numpy as np

MODEL_NAME = 'BAAI/bge-small-en-v1.5'
CACHE_DIR = Path(__file__).resolve().parents[1] / 'data' / 'models'
_model = None
_model_lock = threading.Lock()


def get_embedder():
    global _model
    with _model_lock:
        if _model is None:
            os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
            from fastembed import TextEmbedding
            _model = TextEmbedding(model_name=MODEL_NAME, cache_dir=str(CACHE_DIR), threads=2)
        return _model


def normalize(vectors):
    array = np.asarray(vectors, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), 1e-10)


def build_index(book, progress=lambda **update: None):
    progress(stage='indexing', progress=91, detail='Loading the local embedding model')
    model = get_embedder()
    chunks = book['chunks']
    vectors = []
    for start in range(0, len(chunks), 32):
        batch = chunks[start:start + 32]
        vectors.extend(model.passage_embed([chunk['text'] for chunk in batch]))
        progress(stage='indexing', progress=91 + round(min(start + 32, len(chunks)) / len(chunks) * 8),
                 detail=f'Locally indexed {min(start + 32, len(chunks))} of {len(chunks)} passages')
    book['vectors'] = normalize(vectors)
    book['embeddingModel'] = MODEL_NAME
    return book


def retrieve(book, question, limit=5):
    vector = normalize(list(get_embedder().query_embed(question)))[0]
    scores = book['vectors'] @ vector
    candidates = np.argsort(-scores)
    selected = []
    for index in candidates:
        chunk = book['chunks'][int(index)]
        # Avoid sending almost identical overlapping passages to Gemini.
        duplicate = any(chunk.get('page') == c.get('page') and chunk.get('section') == c.get('section')
                        and max(chunk['start'], c['start']) < min(chunk['end'], c['end'])
                        and min(chunk['end'], c['end']) - max(chunk['start'], c['start']) > 0.6 * min(len(chunk['text']), len(c['text']))
                        for c in selected)
        if duplicate:
            continue
        if float(scores[index]) < 0.30:
            continue
        selected.append({**chunk, 'score': round(float(scores[index]), 4)})
        if len(selected) >= limit:
            break
    return selected


def sources_for(chunks):
    return [{key: chunk[key] for key in ('id', 'source', 'text', 'label', 'page', 'section', 'extraction', 'ocrConfidence', 'score') if key in chunk}
            for chunk in chunks]
