"""Local semantic indexing and book-scoped retrieval."""
from pathlib import Path
import os
import re
import threading

import numpy as np

MODEL_NAME = 'BAAI/bge-small-en-v1.5'
CACHE_DIR = Path(__file__).resolve().parents[1] / 'data' / 'models'
_model = None
_model_lock = threading.Lock()


def is_overview_question(question):
    text = re.sub(r'[^\w\s]', ' ', question.casefold())
    text = re.sub(r'\s+', ' ', text).strip()
    return bool(re.search(
        r'\bwhat (?:is|s) (?:this|the) (?:book|document|text) (?:all )?about\b'
        r'|\b(?:summari[sz]e|sum up) (?:this |the |entire |whole )*(?:book|document|text)\b'
        r'|\b(?:overview|summary|synopsis|main themes?|main topics?|central themes?) (?:of|in|for) (?:this |the |entire |whole )*(?:book|document|text)\b'
        r'|\bwhat (?:does|topics does) (?:this|the) (?:book|document) cover\b', text))


def retrieve_overview(book, limit=18):
    """Select bounded front matter and distributed passages, without a similarity cutoff."""
    chunks = book['chunks']
    if not chunks:
        return []
    selected, used = [], set()

    def add(chunk, role):
        if chunk['id'] not in used and len(selected) < limit:
            selected.append({**chunk, 'overviewRole': role})
            used.add(chunk['id'])

    # Prioritize front matter that explicitly describes the work, rather than only its cover.
    front = chunks[:max(1, min(80, len(chunks) // 4 + 1))]
    structure = re.compile(r'\b(?:table of contents|contents|introduction|preface|about this book|who this book is for)\b', re.I)
    structural = [c for c in front if structure.search(c.get('label', '') + ' ' + c['text'][:700])]
    for chunk in structural[:5]:
        add(chunk, 'introduction or contents')
    for chunk in chunks[:3]:
        add(chunk, 'opening passage')

    # Spread remaining context over distinct PDF pages / EPUB sections. Long TXT documents
    # have only one section, so sample their chunks instead.
    units, locations = [], set()
    for chunk in chunks:
        location = (chunk.get('page'), chunk.get('section'))
        if location not in locations:
            units.append(chunk)
            locations.add(location)
    if len(units) < min(4, len(chunks)):
        units = chunks
    slots = limit - len(selected)
    if slots:
        for index in np.linspace(0, len(units) - 1, slots, dtype=int):
            add(units[int(index)], 'representative passage')
    # Fill gaps left by duplicate opening samples, preserving distributed coverage first.
    for chunk in units:
        if len(selected) >= min(limit, len(units)):
            break
        add(chunk, 'representative passage')
    return selected


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
    if is_overview_question(question):
        return retrieve_overview(book)
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
    return [{key: chunk[key] for key in ('id', 'source', 'text', 'label', 'page', 'section', 'extraction', 'ocrConfidence', 'score', 'overviewRole') if key in chunk}
            for chunk in chunks]
