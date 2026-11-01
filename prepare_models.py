"""Download/cache the local embedding model without uploading user content."""
from backend.retrieval import get_embedder

if __name__ == '__main__':
    vector = next(get_embedder().query_embed('Model setup check'))
    print(f'Local embedding model ready: {len(vector)} dimensions')
