"""Streaming, cancellation, history and follow-up regression tests."""
import asyncio
import importlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend import answers, conversation
from backend.app import create_app
from tests.test_ingestion import upload, wait_for_book


TEXT = 'Trees absorb carbon dioxide through photosynthesis and store carbon in their trunks and roots.'


def result_for(chunks):
    return answers.verify_answer(answers.AnswerPayload(supported=True, statements=[{
        'text': 'Trees absorb carbon dioxide.',
        'citations': [{'chunk_id': chunks[0]['id'], 'quote': 'Trees absorb carbon dioxide'}],
    }]), chunks)


def events(response):
    assert response.status_code == 200, response.text
    return [json.loads(line) for line in response.text.splitlines() if line]


@pytest.fixture
def chat(monkeypatch):
    monkeypatch.setattr(answers, 'settings', lambda: ('test-key', 'test-model'))
    offset = [0]
    module = importlib.import_module('backend.app')
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: time.monotonic() + offset[0]))
    with TestClient(create_app(), base_url='http://127.0.0.1:3000') as client:
        book_id = upload(client, 'forest.txt', TEXT.encode()).json()['id']
        assert wait_for_book(client, book_id)['stage'] == 'ready'
        yield client, book_id, offset


def test_stream_history_followup_and_clear(chat, monkeypatch):
    client, book_id, offset = chat
    calls = []
    async def fake_stream(question, chunks, history=None):
        calls.append((question, history, chunks))
        yield {'event': 'draft', 'text': 'Trees absorb'}
        yield {'event': 'final', 'result': result_for(chunks)}
    monkeypatch.setattr(answers, 'stream_answer', fake_stream)
    first = events(client.post(f'/api/books/{book_id}/chat/stream', json={'question': 'How do trees capture carbon?'}))
    assert [e['event'] for e in first] == ['started', 'status', 'draft', 'final']
    assert calls[0][1] == []
    offset[0] += 6
    second = events(client.post(f'/api/books/{book_id}/chat/stream', json={'question': 'Explain that in simple terms'}))
    assert second[-1]['result']['supported']
    assert len(calls[1][1]) == 1
    assert calls[1][1][0]['question'] == 'How do trees capture carbon?'
    assert any(c['id'] == first[-1]['result']['sources'][0]['id'] for c in calls[1][2])
    history = client.get(f'/api/books/{book_id}/chat/history').json()['turns']
    assert len(history) == 2
    assert all(t['status'] == 'complete' for t in history)
    assert all('draft' not in t for t in history)
    assert client.post(f'/api/books/{book_id}/chat/clear').status_code == 200
    assert client.get(f'/api/books/{book_id}/chat/history').json()['turns'] == []


def test_failed_stream_retry_does_not_duplicate_turn(chat, monkeypatch):
    client, book_id, offset = chat
    calls = []
    async def fake_stream(question, chunks, history=None):
        calls.append((question, history))
        if len(calls) == 1:
            yield {'event':'draft', 'text':'An incomplete draft'}
            raise answers.RagError('Quota reached', 429)
        yield {'event':'final', 'result':result_for(chunks)}
    monkeypatch.setattr(answers, 'stream_answer', fake_stream)
    first = events(client.post(f'/api/books/{book_id}/chat/stream', json={'question':'How do trees capture carbon?'}))
    assert first[-1]['event'] == 'error'
    turn_id = first[0]['turn']['id']
    history = client.get(f'/api/books/{book_id}/chat/history').json()['turns']
    assert history[0]['status'] == 'failed'
    assert 'result' not in history[0]
    offset[0] += 6
    retried = events(client.post(f'/api/books/{book_id}/chat/stream', json={'question':'ignored replacement question', 'retryTurnId':turn_id}))
    assert retried[-1]['event'] == 'final'
    assert calls[1][0] == calls[0][0]
    assert calls[1][1] == []
    assert len(client.get(f'/api/books/{book_id}/chat/history').json()['turns']) == 1


def test_stop_cancels_provider_and_releases_lock(chat, monkeypatch):
    client, book_id, offset = chat
    started, closed = threading.Event(), threading.Event()
    async def waiting_stream(question, chunks, history=None):
        try:
            yield {'event':'draft', 'text':'Provisional text'}
            started.set()
            await asyncio.Event().wait()
        finally:
            closed.set()
    monkeypatch.setattr(answers, 'stream_answer', waiting_stream)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(client.post, f'/api/books/{book_id}/chat/stream', json={'question':'How do trees capture carbon?'})
        assert started.wait(5)
        assert client.post(f'/api/books/{book_id}/chat/clear').status_code == 409
        assert client.post(f'/api/books/{book_id}/chat/stream', json={'question':'Another question'}).status_code == 429
        assert client.post(f'/api/books/{book_id}/chat/cancel').json()['stopping']
        messages = events(future.result(timeout=5))
    assert messages[-1]['event'] == 'stopped'
    assert closed.wait(2)
    turns = client.get(f'/api/books/{book_id}/chat/history').json()['turns']
    assert turns[0]['status'] == 'stopped'
    assert 'result' not in turns[0]
    assert client.post(f'/api/books/{book_id}/chat/clear').status_code == 200


def test_history_is_isolated_by_session_and_book(chat):
    client, book_id, offset = chat
    other_id = upload(client, 'other.txt', b'Bread is made with flour, yeast and water.').json()['id']
    assert wait_for_book(client, other_id)['stage'] == 'ready'
    assert client.get(f'/api/books/{other_id}/chat/history').json()['turns'] == []
    client.cookies.clear()
    assert client.get(f'/api/books/{book_id}/chat/history').status_code == 404
    assert client.post(f'/api/books/{book_id}/chat/cancel').status_code == 404
    assert client.post(f'/api/books/{book_id}/chat/clear').status_code == 404


def test_context_only_contains_recent_successful_answers():
    turns = [{'question': str(i), 'status':'complete', 'result':{'supported':True, 'answer':'Verified answer'}} for i in range(5)]
    turns += [{'question':'bad', 'status':'failed'}, {'question':'stopped', 'status':'stopped'}]
    assert [t['question'] for t in conversation.completed_context(turns)] == ['2','3','4']


def test_partial_json_draft_hides_citations():
    buffer = '{"supported":true,"statements":[{"text":"Trees absorb [99] carbon'
    assert answers.draft_text(buffer) == 'Trees absorb  carbon'
    assert answers.draft_text('{"supported":false,"statements":[') == ''


def test_stream_sdk_validates_final_and_closes_client(monkeypatch):
    from google import genai
    monkeypatch.setattr(answers, 'settings', lambda: ('fake-key', 'test-model'))
    closed = []
    data = json.dumps({'supported':True, 'statements':[{'text':'Trees absorb carbon dioxide.', 'citations':[{'chunk_id':'chunk-1','quote':'Trees absorb carbon dioxide'}]}]})
    class Models:
        async def generate_content_stream(self, **kwargs):
            async def pieces():
                for i in range(0, len(data), 12):
                    yield SimpleNamespace(text=data[i:i+12])
            return pieces()
    class Client:
        def __init__(self, **kwargs):
            self.aio, self.models = self, Models()
        async def __aenter__(self): return self
        async def __aexit__(self, *args): closed.append(True)
    monkeypatch.setattr(genai, 'Client', Client)
    chunks = [{'id':'chunk-1','source':'sample.txt','label':'Text','text':TEXT}]
    async def collect():
        return [message async for message in answers.stream_answer('How do trees capture carbon?', chunks)]
    messages = asyncio.run(collect())
    assert any(m['event'] == 'draft' for m in messages)
    assert messages[-1]['event'] == 'final'
    assert messages[-1]['result']['sources'][0]['quotes'] == ['Trees absorb carbon dioxide']
    assert closed
    data = data.replace('Trees absorb carbon dioxide"', 'This invented quote is not in the source"')
    with pytest.raises(answers.RagError, match='unverifiable'):
        asyncio.run(collect())
