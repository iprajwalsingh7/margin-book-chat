from io import BytesIO
import re
import time
from urllib.parse import quote
from zipfile import ZipFile, ZIP_DEFLATED

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

from backend.app import create_app
from backend.ingestion import ingest, IngestionError


def pdf_fixture(blank=False, encrypted=False, mixed=False):
    writer = PdfWriter()
    if blank:
        writer.add_blank_page(width=612, height=792)
    else:
        for text in ['First page discusses photosynthesis.', 'Second page discusses chlorophyll.']:
            page = writer.add_blank_page(width=612, height=792)
            font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
            page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
            stream = DecodedStreamObject()
            stream.set_data(f'BT /F1 12 Tf 72 720 Td ({text}) Tj ET'.encode())
            page[NameObject('/Contents')] = writer._add_object(stream)
        if mixed:
            writer.add_blank_page(width=612, height=792)
    if encrypted:
        writer.encrypt('test-password')
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def epub_fixture(extra=None):
    entries = {
        'mimetype': 'application/epub+zip',
        'META-INF/container.xml': '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OPS/book.opf"/></rootfiles></container>',
        'OPS/book.opf': '<package xmlns="http://www.idpf.org/2007/opf"><manifest><item id="second" href="two.xhtml"/><item id="first" href="one.xhtml"/></manifest><spine><itemref idref="first"/><itemref idref="second"/></spine></package>',
        'OPS/one.xhtml': '<html><body><h1>First chapter</h1><p>Alpha &amp; beta.</p><script>SECRET_SCRIPT</script></body></html>',
        'OPS/two.xhtml': '<html><body><h1>Second chapter</h1><p>Gamma delta.</p></body></html>',
        **(extra or {}),
    }
    output = BytesIO()
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        for path, text in entries.items():
            archive.writestr(path, text)
    return output.getvalue()


def test_txt_chunk_coverage_and_metadata():
    text = 'A passage about trees and carbon. ' * 250
    book = ingest(text.encode(), 'ecology.txt')
    normalized = re.sub(r'\s+', ' ', text).strip()
    assert len(book['chunks']) > 2
    last_end = 0
    for chunk in book['chunks']:
        assert chunk['text'] == normalized[chunk['start']:chunk['end']]
        assert chunk['start'] <= last_end
        assert len(chunk['text']) <= 1600
        assert chunk['source'] == 'ecology.txt'
        last_end = chunk['end']
    assert last_end == len(normalized)


def test_pdf_pages_warnings_and_encryption():
    book = ingest(pdf_fixture(mixed=True), 'biology.pdf')
    assert book['sectionCount'] == 3
    assert book['chunks'][0]['page'] == 1
    assert 'photosynthesis' in book['chunks'][0]['text']
    assert book['chunks'][1]['page'] == 2
    assert 'chlorophyll' in book['chunks'][1]['text']
    assert '1 page(s)' in book['warnings'][0]
    with pytest.raises(IngestionError, match='OCR'):
        ingest(pdf_fixture(blank=True), 'scan.pdf')
    with pytest.raises(IngestionError, match='Password-protected'):
        ingest(pdf_fixture(encrypted=True), 'locked.pdf')


def test_epub_reading_order_and_safe_text():
    book = ingest(epub_fixture(), 'test.epub')
    assert book['sectionCount'] == 2
    assert book['chunks'][0]['label'] == 'First chapter'
    assert book['chunks'][0]['path'] == 'OPS/one.xhtml'
    assert 'Alpha & beta' in book['chunks'][0]['text']
    assert 'SECRET_SCRIPT' not in book['chunks'][0]['text']
    assert book['chunks'][1]['section'] == 2
    with pytest.raises(IngestionError, match='Encrypted'):
        ingest(epub_fixture({'META-INF/encryption.xml': '<encryption/>'}), 'locked.epub')


@pytest.mark.parametrize(('data', 'name', 'message'), [
    (b'not a PDF', 'fake.pdf', 'signature'),
    (b'\x00\x01\x02', 'binary.txt', 'binary'),
    (b'\xff\xff', 'invalid.txt', 'UTF-8'),
    (b'   ', 'empty.txt', 'No readable text'),
    (b'broken archive', 'broken.epub', 'ZIP archive'),
    (b'x' * (25 * 1024 * 1024 + 1), 'large.txt', '25 MB'),
])
def test_invalid_inputs(data, name, message):
    with pytest.raises(IngestionError, match=message):
        ingest(data, name)


def wait_for_book(client, book_id):
    for _ in range(150):
        result = client.get(f'/api/books/{book_id}').json()
        if result['stage'] in {'ready', 'failed'}:
            return result
        time.sleep(0.05)
    pytest.fail('Book did not reach a terminal state')


def upload(client, name, data):
    return client.post('/api/books', content=data, headers={'Content-Type': 'application/octet-stream', 'X-File-Name': quote(name)})


@pytest.mark.parametrize(('name', 'data', 'expected'), [
    ('source.txt', b'Evidence from an uploaded source. ' * 100, 'Evidence'),
    ('source.pdf', pdf_fixture(), 'photosynthesis'),
    ('source.epub', epub_fixture(), 'Alpha & beta'),
])
def test_http_upload_to_process_preview_isolation_and_delete(name, data, expected):
    with TestClient(create_app(), base_url='http://127.0.0.1:3000') as client:
        assert client.get('/').status_code == 200
        response = upload(client, name, data)
        assert response.status_code == 202
        book_id = response.json()['id']
        book = wait_for_book(client, book_id)
        assert book['stage'] == 'ready', book.get('error')
        assert book['progress'] == 100
        assert expected in book['preview'][0]['text']
        cookie = client.cookies.get('margin_session')
        client.cookies.clear()
        assert client.get(f'/api/books/{book_id}').status_code == 404
        client.cookies.clear()
        client.cookies.set('margin_session', cookie)
        assert client.delete(f'/api/books/{book_id}').status_code == 200
        assert client.get(f'/api/books/{book_id}').status_code == 404


def test_http_failure_validation_and_origin_boundary():
    with TestClient(create_app(), base_url='http://127.0.0.1:3000') as client:
        assert upload(client, '../escape.txt', b'hello').status_code == 400
        assert upload(client, 'empty.txt', b'').status_code == 400
        assert client.post('/api/books', content=b'hello', headers={'origin': 'https://other.example'}).status_code == 403
        assert client.get('/requirements.txt').status_code == 404
        response = upload(client, 'broken.pdf', b'not a PDF')
        book = wait_for_book(client, response.json()['id'])
        assert book['stage'] == 'failed'
        assert 'signature' in book['error']

def scanned_pdf_fixture(blank=False):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.new('RGB', (1600, 1000), 'white')
    if not blank:
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=42)
        draw.text((100, 180), 'FOREST SYSTEMS', fill='black', font=font)
        draw.text((100, 280), 'Trees capture carbon through photosynthesis.', fill='black', font=font)
        draw.text((100, 380), 'Forests protect water and support wildlife.', fill='black', font=font)
    output = BytesIO()
    image.save(output, 'PDF', resolution=150)
    image.close()
    return output.getvalue()


def test_real_offline_ocr_on_image_only_pdf(monkeypatch):
    import requests
    def no_network(*args, **kwargs):
        pytest.fail('OCR must not make HTTP requests')
    monkeypatch.setattr(requests.sessions.Session, 'request', no_network)
    updates = []
    book = ingest(scanned_pdf_fixture(), 'scanned.pdf', lambda **update: updates.append(update))
    assert book['ocrPageCount'] == 1
    assert book['ocrAttemptedPages'] == 1
    assert book['chunks'][0]['page'] == 1
    assert book['chunks'][0]['extraction'] == 'ocr'
    assert 'photosynthesis' in book['chunks'][0]['text'].lower()
    assert 0 <= book['chunks'][0]['ocrConfidence'] <= 1
    assert any(update['stage'] == 'ocr' for update in updates)
    assert any('Recognition can contain errors' in warning for warning in book['warnings'])


def test_mixed_pdf_keeps_native_text_and_ocr_page_numbers():
    writer = PdfWriter()
    writer.append(BytesIO(pdf_fixture()))
    writer.append(BytesIO(scanned_pdf_fixture()))
    output = BytesIO()
    writer.write(output)
    book = ingest(output.getvalue(), 'mixed.pdf')
    assert book['sectionCount'] == 3
    assert book['ocrPageCount'] == 1
    assert [c['page'] for c in book['chunks']] == [1, 2, 3]
    assert [c['extraction'] for c in book['chunks']] == ['text', 'text', 'ocr']
    assert 'chlorophyll' in book['chunks'][1]['text']
    assert 'photosynthesis' in book['chunks'][2]['text'].lower()


def test_blank_scanned_pdf_is_not_hallucinated():
    with pytest.raises(IngestionError, match='No readable text'):
        ingest(scanned_pdf_fixture(blank=True), 'blank-scan.pdf')


def test_low_ocr_confidence_is_flagged(monkeypatch):
    class LowConfidenceOCR:
        def __init__(self, data):
            pass
        def read_page(self, index):
            return 'Uncertain recognized words from a damaged scan.', 0.65
        def close(self):
            pass
    monkeypatch.setattr('backend.ingestion.PdfOCR', LowConfidenceOCR)
    book = ingest(scanned_pdf_fixture(), 'faint.pdf')
    assert book['ocrLowConfidencePages'] == [1]
    assert any('Low OCR confidence' in warning for warning in book['warnings'])


def test_ocr_upload_to_api_includes_provenance():
    with TestClient(create_app(), base_url='http://127.0.0.1:3000') as client:
        response = upload(client, 'scan.pdf', scanned_pdf_fixture())
        assert response.status_code == 202
        book_id = response.json()['id']
        book = wait_for_book(client, book_id)
        assert book['stage'] == 'ready', book.get('error')
        assert book['ocrPageCount'] == 1
        assert book['preview'][0]['extraction'] == 'ocr'
        assert 'photosynthesis' in book['preview'][0]['text'].lower()
        assert client.delete(f'/api/books/{book_id}').status_code == 200


def test_delete_running_ocr_job():
    with TestClient(create_app(), base_url='http://127.0.0.1:3000') as client:
        response = upload(client, 'cancel-scan.pdf', scanned_pdf_fixture())
        book_id = response.json()['id']
        assert client.delete(f'/api/books/{book_id}').status_code == 200
        assert client.get(f'/api/books/{book_id}').status_code == 404
