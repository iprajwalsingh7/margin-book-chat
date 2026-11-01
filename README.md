# Margin — chat with your book

A Python RAG project built in reviewable layers. The frontend is plain HTML, CSS and browser JavaScript. FastAPI serves the interface and API; extraction and chunking run entirely in Python.

## Run on Windows

Install uv first. Python 3.11+ is supported; this project selects Python 3.14. uv manages the interpreter and project environment automatically. No manual environment creation or activation is needed.

```powershell
uv sync --locked
uv run run.py
```

Open http://127.0.0.1:3000. To use another port, set `$env:PORT = '3001'` before starting.

Dependencies are declared in `pyproject.toml` and resolved in `uv.lock`. Use `uv add <package>` for runtime dependencies and `uv add --dev <package>` for development tools. uv uses its standard ignored `.venv` directory internally.

## Layer 1 — foundation

A responsive dark research workspace with electric blue accents, a source library, drag-and-drop file selection, validation and a disabled question composer.

## Layer 2 — ingestion

- Upload one PDF, EPUB or UTF-8 TXT file, up to 25 MB.
- **PDF:** pypdf extracts text page by page and preserves physical PDF page numbers.
- **EPUB:** Python's zipfile opens the archive; defusedxml reads the manifest and spine; Beautiful Soup extracts chapter text in reading order. Section titles and chapter paths are preserved. Page numbers are not invented.
- **TXT:** Python decodes UTF-8 text.
- **Chunking:** normalize text and create passages up to 1,600 characters with approximately 200 characters of overlap. Keep source and page/section metadata on every passage.
- **Background work:** separate Python processes extract documents while FastAPI serves status requests. Text extraction has a 60-second deadline. Jobs that need OCR get up to 15 minutes total. Deletion terminates the extraction process.
- Inspect the first three passages, text statistics and extraction warnings in the interface.
- Retry a failed upload, replace a book, remove it or reload the tab to restore its job.

Layer 3 now enables local semantic search and Gemini answers once a key is configured.

## Architecture

```text
run.py                 Local Uvicorn entry point
backend/app.py         FastAPI routes, temporary jobs, cookies and process lifecycle
backend/ingestion.py   PDF/EPUB/TXT extraction and source-aware chunking
backend/ocr.py         Local PDF rendering and OCR with bundled models
backend/retrieval.py   Local embeddings, indexing and retrieval
backend/answers.py     Gemini adapter and citation verification
prepare_models.py     Embedding model setup
public/                HTML, CSS and browser interactions
tests/                 Generated document fixtures and Python tests
pyproject.toml         Project metadata and dependency groups
uv.lock                Reproducible dependency resolution
.python-version        Selected Python version
```

## Layer 2.5 — local scanned-PDF OCR

PDF pages with fewer than 80 alphanumeric characters and a raster image are automatically sent through OCR. Textless pages with drawing content are also checked, supporting some outlined text and inline-image PDFs. Ordinary digital text pages use their existing text.

pypdfium2 renders candidate pages at up to 200 DPI with bounded image dimensions (4,000 pixels per side and 12 million pixels total). RapidOCR runs CPU inference with ONNX Runtime and explicitly selected models bundled with the pinned rapidocr package. No Tesseract installation, API key, runtime model download or external OCR service is required. Installing dependencies still requires Internet access initially.

Recovered passages retain physical page numbers, an OCR marker and average line recognition confidence. The interface shows the OCR page count and warns about recognition errors. Confidence below 85% triggers a page-specific warning; the score is an engine estimate, not a guarantee of accuracy. If OCR recovers less text than the existing text layer, the existing text is retained.

This implementation has been tested on clear printed English text. Handwriting, complex reading order, damaged scans and other languages need separate validation. Text inside images on text-rich pages is not automatically transcribed, and scanned images embedded in EPUBs are outside this OCR layer.

OCR jobs allow up to 500 candidate pages and 15 minutes total, including model startup. Larger scanned books should be split into smaller PDFs. Removing a processing book cancels its Python process. The original file and rendered page images are not written to disk.

## Layer 3 — local retrieval + Gemini answers

After extraction/OCR, the Python worker embeds every passage with FastEmbed's BAAI/bge-small-en-v1.5 model (384 dimensions). Vectors stay in server memory with the book. Search embeds the question locally, ranks normalized vectors by cosine similarity, and returns up to five passages above a 0.30 similarity floor, with near-duplicate overlaps suppressed. This prototype searches in memory; it does not need a hosted vector database. The model is intended for English retrieval.

The embedding model is downloaded once into the ignored data/models cache. Its weights were prepared locally for this workspace. On another machine, prepare them before uploading:

```powershell
uv run prepare_models.py
```

Use **Find passages locally** to inspect retrieval without sending book content or questions to Gemini. The model provider is not contacted by this action.

### Enable Gemini answers

The ignored .env file has been prepared with a blank key. Edit it locally:

```dotenv
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
```

Do not paste keys into chat or commit them. `.env.example` contains the shareable blank configuration. Environment variables take precedence. Refresh the browser after editing .env; the backend reads settings for each request and does not require a restart. The status label confirms only that a key is present, not that the provider has accepted it.

**Ask Gemini** sends the question and selected passages plus their source metadata (up to five for focused questions, up to 18 for book overviews) to Google's Gemini Developer API. Embeddings are not sent. Use a model available to your AI Studio project; availability and free-tier quotas can change. The app does not enable billing or guarantee that a key belongs to a free-tier project. Review Google's applicable data-use terms before asking about private books.

Gemini returns structured statements with source IDs and verbatim quotations. The server rejects unknown source IDs and quotes not present in the retrieved passages, then constructs numbered citations and page/section source cards. This verifies citation provenance; it does not prove that every generated claim follows logically from its quote. Review cited evidence. Missing evidence produces an explicit insufficient-evidence answer.

Questions are limited to 2,000 characters. Only one request per book runs at a time, with a five-second interval between Gemini attempts, a 45-second provider timeout and no automatic retries. Provider quota errors, missing keys, invalid configuration and unverifiable citations are surfaced clearly. Plain text rendering avoids executing markup from answers or books. Prior questions are not sent as conversation history in this layer. Book-overview questions use a separate route: opening passages, introduction/contents when found in the front matter, and passages sampled across distinct pages or sections. The response describes the subject, audience, purpose and themes supported by that selection, with verified quotations. This is a bounded overview, not an exhaustive chapter-by-chapter summary.

### Layer 3 verification

Tests use real local embeddings and a small paraphrased-query retrieval evaluation. Gemini calls are replaced with test providers, covering structured output, cited quotes, invalid source IDs, insufficient evidence and quota errors. API tests check local search without a key, book isolation, question limits and the generated-answer flow. Live Gemini generation has been verified on generated sample passages, including a broad book-overview question. User books are not included in the test fixtures.

## Temporary storage and limits

Uploads are not written to disk. Extracted passages live in server memory and expire after one hour without a status request or on server restart. Browser cookies isolate temporary jobs; production authentication comes in a later layer. The server listens on loopback only and blocks cross-origin mutations.

Limits: 25 MB per upload, 2,000 PDF pages, 5 million extracted characters, two concurrent uploads/extraction processes, and 12 retained jobs. EPUB archives are limited to 4,000 entries, 60 MB total declared expansion and 10 MB per entry.

Scanned PDFs are supported through automatic local OCR. Empty PDF pages produce warnings; entirely unreadable books fail clearly. Password-protected PDFs and encrypted EPUBs are rejected. Complex PDF layouts may need further extraction refinement. TXT currently requires UTF-8 encoding.

## Verify

```powershell
uv run --locked pytest -q
```

Tests cover passage coverage, PDF text and page references, empty-page warnings, password protection, EPUB reading order and script removal, invalid files, size limits, real upload-to-process extraction for all formats, session isolation and deletion. Tests generate their own documents and do not need a running server or model key. OCR tests use actual image-only PDFs and local inference, check mixed-page provenance and confidence warnings, prohibit HTTP model fetching during recognition, and verify deletion during processing.

## Next layers

3. **RAG (implemented):** local embeddings, book-scoped retrieval and Gemini answers with verified source quotations. Live provider validation requires your key.
4. **Conversation:** streaming, follow-up resolution, cancellation and recovery. Whole-book summaries need a separate chapter-level flow.
5. **Durability and access:** persistent storage, accounts, ownership, deletion and deployment preparation.

Project directory: `C:\Users\prajwal\Documents\Projects\book-chat`. No remote repository or deployment is configured. User books, secrets, extracted data and environments belong in ignored locations.




