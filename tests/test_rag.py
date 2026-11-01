"""RAG evaluation and Gemini adapter tests; never call a live Gemini API."""
import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend import answers
from backend.app import create_app
from backend.ingestion import chunk_sections
from backend.retrieval import build_index, retrieve
from tests.test_ingestion import upload, wait_for_book


@pytest.fixture(scope='module')
def indexed_book():
    sections = [
        {'label': 'Forest carbon', 'page': 1, 'text': 'Trees absorb carbon dioxide through photosynthesis. Carbon is stored in trunks, roots, and forest soil.'},
        {'label': 'Water cycle', 'page': 2, 'text': 'Forests regulate water through transpiration. Roots and forest soils absorb rainfall, reduce flooding, and protect streams.'},
        {'label': 'Space exploration', 'page': 3, 'text': 'Rockets travel through space. Astronauts perform experiments aboard an orbiting space station.'},
        {'label': 'Cooking', 'page': 4, 'text': 'Bread dough contains flour, water, yeast and salt. Fermentation helps the dough rise before baking.'},
    ]
    return build_index({'chunks': chunk_sections(sections, 'research.pdf')})


@pytest.mark.parametrize(('question', 'expected_page'), [
    ('How do trees remove carbon from the atmosphere?', 1),
    ('How do forests help prevent floods and protect rivers?', 2),
    ('What makes bread dough rise?', 4),
])
def test_semantic_retrieval_evaluation(indexed_book, question, expected_page):
    sources = retrieve(indexed_book, question)
    assert sources
    assert sources[0]['page'] == expected_page
    assert all(source['source'] == 'research.pdf' for source in sources)


def test_answer_citations_are_server_verified():
    chunks = [{'id': 'chunk-1', 'source': 'book.pdf', 'page': 7, 'label': 'Page 7', 'text': 'Trees absorb carbon dioxide through photosynthesis.'}]
    payload = answers.AnswerPayload(supported=True, statements=[{'text': 'Trees absorb carbon dioxide.', 'citations': [{'chunk_id': 'chunk-1', 'quote': 'Trees absorb carbon dioxide'}]}])
    result = answers.verify_answer(payload, chunks)
    assert result['answer'] == 'Trees absorb carbon dioxide. [1]'
    assert result['sources'][0]['page'] == 7
    assert result['sources'][0]['quotes'] == ['Trees absorb carbon dioxide']
    payload.statements[0].citations[0].chunk_id = 'invented-id'
    with pytest.raises(answers.RagError, match='unverifiable'):
        answers.verify_answer(payload, chunks)
    payload.statements[0].citations[0].chunk_id = 'chunk-1'
    payload.statements[0].citations[0].quote = 'An invented quote absent from the book.'
    with pytest.raises(answers.RagError, match='unverifiable'):
        answers.verify_answer(payload, chunks)


def test_unsupported_answer_is_explicit():
    result = answers.verify_answer(answers.AnswerPayload(supported=False, statements=[]), [])
    assert result['supported'] is False
    assert result['sources'] == []
    assert 'supporting evidence' in result['answer']


def test_gemini_sdk_adapter_without_network(monkeypatch):
    from google import genai
    monkeypatch.setattr(answers, 'settings', lambda: ('test-key-not-real', 'test-model'))
    captured = {}
    class FakeModels:
        async def generate_content(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(text=json.dumps({'supported': True, 'statements': [{'text': 'Trees absorb carbon.', 'citations': [{'chunk_id': 'chunk-1', 'quote': 'Trees absorb carbon dioxide'}]}]}))
    class FakeClient:
        def __init__(self, **kwargs):
            self.aio = self
            self.models = FakeModels()
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
    monkeypatch.setattr(genai, 'Client', FakeClient)
    result = asyncio.run(answers.generate_answer('How do trees capture carbon?', [{'id': 'chunk-1', 'source': 'test.txt', 'label': 'Text', 'text': 'Trees absorb carbon dioxide through photosynthesis.'}]))
    assert result['supported']
    assert captured['model'] == 'test-model'
    assert 'test-key-not-real' not in captured['contents']
    assert len(json.loads(captured['contents'])['passages']) == 1


def test_gemini_quota_error_is_actionable(monkeypatch):
    from google import genai
    from google.genai import errors
    monkeypatch.setattr(answers, 'settings', lambda: ('fake-key', 'test-model'))
    class FakeModels:
        async def generate_content(self, **kwargs):
            raise errors.ClientError(429, {'error': {'message': 'secret provider diagnostic'}})
    class FakeClient:
        def __init__(self, **kwargs):
            self.aio, self.models = self, FakeModels()
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
    monkeypatch.setattr(genai, 'Client', FakeClient)
    with pytest.raises(answers.RagError, match='quota or rate limit') as error:
        asyncio.run(answers.generate_answer('Question?', [{'id': 'chunk-1', 'text': 'Some supporting text.'}]))
    assert error.value.status == 429
    assert 'secret' not in str(error.value)


def test_search_without_key_and_book_isolation(monkeypatch):
    monkeypatch.setattr(answers, 'settings', lambda: ('', 'test-model'))
    with TestClient(create_app(), base_url='http://127.0.0.1:3000') as client:
        book_id = upload(client, 'forest.txt', b'Trees absorb carbon dioxide through photosynthesis.').json()['id']
        assert wait_for_book(client, book_id)['stage'] == 'ready'
        result = client.post(f'/api/books/{book_id}/search', json={'question': 'How do trees capture carbon?'})
        assert result.status_code == 200
        assert result.json()['localOnly'] is True
        assert 'photosynthesis' in result.json()['sources'][0]['text']
        assert client.post(f'/api/books/{book_id}/ask', json={'question': 'How do trees capture carbon?'}).status_code == 503
        assert client.get('/api/config').json()['geminiConfigured'] is False
        client.cookies.clear()
        assert client.post(f'/api/books/{book_id}/search', json={'question': 'How do trees capture carbon?'}).status_code == 404


def test_ask_flow_with_test_provider_and_question_validation(monkeypatch):
    monkeypatch.setattr(answers, 'settings', lambda: ('fake-test-key', 'test-model'))
    async def fake_generate(question, chunks):
        assert chunks
        return answers.verify_answer(answers.AnswerPayload(supported=True, statements=[{'text': 'Trees absorb carbon dioxide.', 'citations': [{'chunk_id': chunks[0]['id'], 'quote': 'Trees absorb carbon dioxide'}]}]), chunks)
    monkeypatch.setattr(answers, 'generate_answer', fake_generate)
    with TestClient(create_app(), base_url='http://127.0.0.1:3000') as client:
        book_id = upload(client, 'forest.txt', b'Trees absorb carbon dioxide through photosynthesis.').json()['id']
        assert wait_for_book(client, book_id)['stage'] == 'ready'
        assert client.post(f'/api/books/{book_id}/ask', json={'question': '  '}).status_code == 422
        assert client.post(f'/api/books/{book_id}/ask', json={'question': 'x' * 2001}).status_code == 422
        result = client.post(f'/api/books/{book_id}/ask', json={'question': 'How do trees capture carbon?'})
        assert result.status_code == 200
        assert result.json()['supported'] is True
        assert result.json()['sources'][0]['source'] == 'forest.txt'
        assert 'fake-test-key' not in result.text
        assert client.post(f'/api/books/{book_id}/ask', json={'question': 'Another question?'}).status_code == 429

@pytest.mark.parametrize('question', [
    'What is this book about?', 'what is the book all about',
    'Summarize the whole book', 'Give me an overview of this document',
    'What are the main themes in this book?', 'What does this book cover?',
])
def test_overview_intent(question):
    from backend.retrieval import is_overview_question
    assert is_overview_question(question)


@pytest.mark.parametrize('question', ['What is system design?', 'Summarize chapter 3', 'What does the author say about caching?'])
def test_focused_questions_do_not_use_overview(question):
    from backend.retrieval import is_overview_question
    assert not is_overview_question(question)


def test_overview_retrieval_covers_front_matter_and_later_pages():
    chunks = chunk_sections([
        {'page': i, 'label': f'Page {i}', 'text': ('Introduction: This book teaches software architecture and system design.' if i == 3 else f'Chapter material on page {i}. ' * 10)}
        for i in range(1, 101)
    ], 'book.pdf')
    # No vectors needed: broad questions must not depend on matching their vague wording.
    sources = retrieve({'chunks': chunks}, 'What is this book about?')
    pages = {source['page'] for source in sources}
    assert 3 in pages
    assert 100 in pages
    assert any(40 <= page <= 70 for page in pages)
    assert len(sources) <= 18
    assert len({c['id'] for c in sources}) == len(sources)
    assert all('overviewRole' in c for c in sources)


def test_overview_samples_long_text_documents():
    from backend.retrieval import retrieve_overview
    chunks = chunk_sections([{'section': 1, 'label': 'Text', 'text': 'A book passage. ' * 5000}], 'book.txt')
    sources = retrieve_overview({'chunks': chunks})
    assert sources[-1]['start'] > 0
    assert any(c['id'] == chunks[-1]['id'] for c in sources)
    assert len(sources) <= 18


def test_overview_api_routes_to_distributed_context(monkeypatch):
    monkeypatch.setattr(answers, 'settings', lambda: ('fake-test-key', 'test-model'))
    async def fake_generate(question, chunks):
        assert any('overviewRole' in c for c in chunks)
        return {**answers.verify_answer(answers.AnswerPayload(supported=True, statements=[{'text': 'The book teaches software architecture.', 'citations': [{'chunk_id': chunks[0]['id'], 'quote': 'This book teaches software architecture'}]}]), chunks), 'mode': 'overview'}
    monkeypatch.setattr(answers, 'generate_answer', fake_generate)
    with TestClient(create_app(), base_url='http://127.0.0.1:3000') as client:
        book_id = upload(client, 'guide.txt', b'This book teaches software architecture and system design interviews.').json()['id']
        assert wait_for_book(client, book_id)['stage'] == 'ready'
        result = client.post(f'/api/books/{book_id}/ask', json={'question': 'What is this book about?'})
        assert result.status_code == 200
        assert result.json()['supported']
        assert result.json()['mode'] == 'overview'
        assert result.json()['sources'][0]['source'] == 'guide.txt'
