"""Text extraction and source-aware passage preparation."""
from io import BytesIO
import posixpath
import re
from urllib.parse import unquote
from zipfile import ZipFile, BadZipFile

from bs4 import BeautifulSoup
from defusedxml import ElementTree as ET
from pypdf import PdfReader
from backend.ocr import PdfOCR, MAX_OCR_PAGES, LOW_CONFIDENCE, has_raster_image

MAX_BYTES = 25 * 1024 * 1024
MAX_TEXT = 5_000_000


class IngestionError(ValueError):
    pass


def chunk_sections(sections, source):
    chunks = []
    for section in sections:
        text = re.sub(r'\s+', ' ', section['text']).strip()
        start = 0
        while start < len(text):
            end = min(start + 1600, len(text))
            if end < len(text):
                boundary = text.rfind(' ', start, end + 1)
                if boundary > start + 1000:
                    end = boundary
            chunks.append({**section, 'id': f'chunk-{len(chunks) + 1}', 'source': source,
                           'text': text[start:end], 'start': start, 'end': end})
            if end == len(text):
                break
            start = max(start + 1, end - 200)
    return chunks


def _epub_sections(data, progress):
    try:
        archive = ZipFile(BytesIO(data))
    except BadZipFile as exc:
        raise IngestionError('This EPUB is not a valid ZIP archive.') from exc
    with archive:
        entries = archive.infolist()
        if (len(entries) > 4000 or sum(e.file_size for e in entries) > 60 * 1024 * 1024
                or any(e.file_size > 10 * 1024 * 1024 for e in entries)):
            raise IngestionError('EPUB exceeds the supported extraction limits.')
        paths = {entry.filename for entry in entries}
        if len(paths) != len(entries):
            raise IngestionError('EPUB contains duplicate archive paths.')
        if any(e.flag_bits & 1 for e in entries) or 'META-INF/encryption.xml' in paths:
            raise IngestionError('Encrypted EPUBs are not supported. Use an unencrypted book.')
        if 'mimetype' not in paths or archive.read('mimetype').strip() != b'application/epub+zip':
            raise IngestionError('The file is not an EPUB publication.')
        if 'META-INF/container.xml' not in paths:
            raise IngestionError('EPUB is missing its publication manifest.')
        container = ET.fromstring(archive.read('META-INF/container.xml'))
        rootfile = next((e for e in container.iter() if e.tag.split('}')[-1] == 'rootfile'), None)
        opf_path = rootfile.get('full-path') if rootfile is not None else None
        if opf_path not in paths:
            raise IngestionError('EPUB is missing its publication manifest.')
        package = ET.fromstring(archive.read(opf_path))
        items = {e.get('id'): e.get('href') for e in package.iter() if e.tag.split('}')[-1] == 'item'}
        order = [e.get('idref') for e in package.iter() if e.tag.split('}')[-1] == 'itemref']
        if not order:
            raise IngestionError('EPUB has no reading order.')
        sections, total = [], 0
        for i, item_id in enumerate(order, 1):
            href = items.get(item_id)
            if not href:
                raise IngestionError('EPUB reading order references a missing chapter.')
            path = posixpath.normpath(posixpath.join(posixpath.dirname(opf_path), unquote(href.split('#')[0])))
            if path not in paths:
                raise IngestionError('EPUB contains an unreadable chapter.')
            soup = BeautifulSoup(archive.read(path), 'html.parser')
            for element in soup(['script', 'style', 'noscript']):
                element.decompose()
            heading = soup.find(['h1', 'h2', 'h3'])
            text = (soup.body or soup).get_text(' ', strip=True)
            total += len(text)
            if total > MAX_TEXT:
                raise IngestionError('Book exceeds the 5 million character limit.')
            sections.append({'label': heading.get_text(' ', strip=True) if heading else f'Section {i}',
                             'section': i, 'path': path, 'text': text})
            progress(stage='extracting', progress=15 + round(i / len(order) * 65), detail=f'Reading section {i} of {len(order)}')
        return sections


def ingest(data, name, progress=lambda **update: None):
    extension = name.rsplit('.', 1)[-1].lower()
    if extension not in {'txt', 'pdf', 'epub'}:
        raise IngestionError('Choose a PDF, EPUB or TXT file.')
    if not data or len(data) > MAX_BYTES:
        raise IngestionError('File must be non-empty and no larger than 25 MB.')
    progress(stage='extracting', progress=10, detail='Validating and reading your book')
    warnings = []
    ocr_pages, ocr_attempts, low_confidence_pages = 0, 0, []
    if extension == 'txt':
        try:
            text = data.decode('utf-8-sig')
        except UnicodeDecodeError as exc:
            raise IngestionError('TXT files must use UTF-8 encoding.') from exc
        if re.search(r'[\x00-\x08\x0e-\x1f]', text):
            raise IngestionError('This file contains binary data, not plain text.')
        if len(text) > MAX_TEXT:
            raise IngestionError('Book exceeds the 5 million character limit.')
        sections = [{'label': 'Text document', 'section': 1, 'text': text}]
    elif extension == 'epub':
        sections = _epub_sections(data, progress)
    else:
        if b'%PDF-' not in data[:1024]:
            raise IngestionError('The file does not contain a valid PDF signature.')
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            raise IngestionError('Password-protected PDFs are not supported. Upload an unlocked copy.')
        if len(reader.pages) > 2000:
            raise IngestionError('PDF exceeds the 2,000 page limit.')
        sections, total, empty_pages = [], 0, 0
        ocr = None
        try:
            for number, page in enumerate(reader.pages, 1):
                try:
                    text = (page.extract_text() or '').strip()
                except Exception:
                    text = ''
                section = {'label': f'Page {number}', 'page': number, 'extraction': 'text'}
                # Sparse text with a raster image usually indicates a scan or a captioned scan.
                # Textless nonblank pages also cover inline images and outlined/vector text.
                content = page.get('/Contents')
                needs_ocr = (len(''.join(c for c in text if c.isalnum())) < 80 and has_raster_image(page)) or (not text and content is not None)
                if needs_ocr:
                    ocr_attempts += 1
                    if ocr_attempts > MAX_OCR_PAGES:
                        raise IngestionError(f'Book exceeds the {MAX_OCR_PAGES} OCR page limit. Split it into smaller PDFs.')
                    progress(stage='ocr', progress=15 + round((number - 1) / len(reader.pages) * 65), detail=f'OCR: recognizing page {number} of {len(reader.pages)} locally')
                    try:
                        if ocr is None:
                            ocr = PdfOCR(data)
                        recovered, confidence = ocr.read_page(number - 1)
                    except Exception as exc:
                        raise IngestionError(f'Local OCR failed on page {number}. Check the file and run uv sync --locked before retrying.') from exc
                    if len(''.join(c for c in recovered if c.isalnum())) > len(''.join(c for c in text if c.isalnum())):
                        text = recovered
                        section.update(extraction='ocr', ocrConfidence=confidence)
                        ocr_pages += 1
                        if confidence is not None and confidence < LOW_CONFIDENCE:
                            low_confidence_pages.append(number)
                total += len(text)
                if total > MAX_TEXT:
                    raise IngestionError('Book exceeds the 5 million character limit.')
                empty_pages += not bool(text)
                sections.append({**section, 'text': text})
                progress(stage='extracting', progress=15 + round(number / len(reader.pages) * 65), detail=f'Read page {number} of {len(reader.pages)}')
        finally:
            if ocr is not None:
                ocr.close()
        if empty_pages:
            warnings.append(f'{empty_pages} page(s) had no readable text after extraction/OCR and may be blank or unreadable.')
        if ocr_pages:
            warnings.append(f'OCR recovered text from {ocr_pages} page(s). Recognition can contain errors; verify important passages against the book.')
        if low_confidence_pages:
            warnings.append('Low OCR confidence on page(s): ' + ', '.join(map(str, low_confidence_pages)) + '. Review these pages carefully.')
    progress(stage='chunking', progress=90, detail='Creating source-linked passages')
    chunks = chunk_sections(sections, name)
    if not chunks:
        raise IngestionError('No readable text found after text extraction/OCR. The document may be blank or the scan too unclear.')
    return {'name': name, 'format': extension, 'bytes': len(data), 'sectionCount': len(sections),
            'characters': sum(len(s['text']) for s in sections), 'chunks': chunks, 'warnings': warnings,
            'ocrPageCount': ocr_pages, 'ocrAttemptedPages': ocr_attempts, 'ocrLowConfidencePages': low_confidence_pages}


def extraction_worker(connection, data, name):
    """Run in a separate process so deletion and timeouts can stop extraction."""
    try:
        book = ingest(data, name, lambda **update: connection.send({'type': 'progress', **update}))
        from backend.retrieval import build_index
        try:
            book = build_index(book, lambda **update: connection.send({'type': 'progress', **update}))
        except Exception as exc:
            raise IngestionError('Could not prepare local search. Run uv run prepare_models.py, then retry the upload.') from exc
        connection.send({'type': 'ready', 'book': book})
    except IngestionError as exc:
        connection.send({'type': 'failed', 'error': str(exc)})
    except Exception:
        connection.send({'type': 'failed', 'error': 'Unable to read this document. It may be malformed, damaged, or unsupported.'})
    finally:
        connection.close()
