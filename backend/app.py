"""Local FastAPI server for the Margin book workspace."""
import asyncio
import json
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
import multiprocessing as mp
from pathlib import Path
import re
import secrets
import time
from urllib.parse import unquote

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from backend.ingestion import MAX_BYTES, extraction_worker
from backend import answers, retrieval, conversation

PUBLIC = Path(__file__).resolve().parents[1] / 'public'
TTL = 3600


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=2000)

    @field_validator('question')
    @classmethod
    def nonblank(cls, value):
        value = value.strip()
        if len(value) < 3:
            raise ValueError('Enter a question of at least three characters.')
        return value


class ChatQuestion(Question):
    retryTurnId: str | None = Field(default=None, max_length=64)


@dataclass
class Job:
    id: str
    owner: str
    name: str
    stage: str = 'queued'
    progress: int = 5
    detail: str = 'Preparing extraction'
    error: str | None = None
    book: dict | None = None
    process: object = None
    task: asyncio.Task | None = None
    touched: float = field(default_factory=time.monotonic)
    answering: bool = False
    last_answer: float = 0
    turns: list = field(default_factory=list)
    stream_cancel: asyncio.Event | None = None


def create_app():
    jobs, sessions = {}, {}
    receiving = 0
    context = mp.get_context('spawn')

    async def dispose(job_id):
        job = jobs.pop(job_id, None)
        if not job:
            return
        if job.stream_cancel:
            job.stream_cancel.set()
        if job.task:
            job.task.cancel()
            with suppress(asyncio.CancelledError):
                await job.task
        # A task cancelled before its first run never enters monitor's finally block.
        if job.process is not None:
            if job.process.is_alive():
                job.process.terminate()
            await asyncio.to_thread(job.process.join, 2)
            job.process.close()
            job.process = None

    async def cleanup():
        while True:
            await asyncio.sleep(60)
            now = time.monotonic()
            for job_id, job in list(jobs.items()):
                if now - job.touched > TTL:
                    await dispose(job_id)
            for token, touched in list(sessions.items()):
                if now - touched > TTL:
                    sessions.pop(token, None)

    @asynccontextmanager
    async def lifespan(app):
        cleaner = asyncio.create_task(cleanup())
        yield
        cleaner.cancel()
        with suppress(asyncio.CancelledError):
            await cleaner
        for job_id in list(jobs):
            await dispose(job_id)

    app = FastAPI(title='Margin — book intelligence', version='0.4.0', lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware('http')
    async def session_and_headers(request, call_next):
        if request.url.hostname not in {'127.0.0.1', 'localhost'}:
            return JSONResponse({'error': 'Unrecognized host.'}, 403)
        origin = request.headers.get('origin')
        expected_origin = f'{request.url.scheme}://{request.headers.get("host", "")}'
        if request.method not in {'GET', 'HEAD'} and (
                (origin and origin != expected_origin) or request.headers.get('sec-fetch-site') == 'cross-site'):
            return JSONResponse({'error': 'Cross-origin requests are not allowed.'}, 403)
        token = request.cookies.get('margin_session')
        if token not in sessions:
            token = secrets.token_urlsafe(32)
        sessions[token] = time.monotonic()
        request.state.session = token
        response = await call_next(request)
        response.set_cookie('margin_session', token, httponly=True, samesite='strict', max_age=TTL)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Content-Security-Policy'] = "default-src 'self'; style-src 'self'; script-src 'self'; object-src 'none'; frame-ancestors 'none'"
        return response

    async def monitor(job, connection):
        started = time.monotonic()
        deadline = started + 60
        try:
            while time.monotonic() < deadline:
                if connection.poll():
                    message = await asyncio.to_thread(connection.recv)
                    if message['type'] == 'progress':
                        job.stage, job.progress, job.detail = message['stage'], message['progress'], message['detail']
                        if job.stage in {'ocr', 'indexing'}:
                            deadline = started + 900
                    elif message['type'] == 'ready':
                        job.book, job.stage, job.progress = message['book'], 'ready', 100
                        job.detail = 'Book indexed locally and ready for questions'
                        return
                    else:
                        job.stage, job.error = 'failed', message['error']
                        job.detail = job.error
                        return
                elif not job.process.is_alive():
                    # A final message can arrive just after the previous poll.
                    if connection.poll():
                        continue
                    raise RuntimeError('Processing stopped unexpectedly.')
                await asyncio.sleep(0.05)
            raise TimeoutError('Processing took too long. Try a smaller book.')
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            job.stage, job.error = 'failed', str(exc) or 'Document processing failed.'
            job.detail = job.error
        finally:
            if job.process.is_alive():
                job.process.terminate()
            await asyncio.to_thread(job.process.join, 2)
            connection.close()
            job.process.close()
            job.process = None

    @app.post('/api/books', status_code=202)
    async def upload(request: Request):
        nonlocal receiving
        if receiving + sum(j.process is not None for j in jobs.values()) >= 2 or len(jobs) + receiving >= 12:
            return JSONResponse({'error': 'Processing capacity is full. Remove a book or try again shortly.'}, 429)
        if request.headers.get('content-type') != 'application/octet-stream':
            return JSONResponse({'error': 'Expected a binary file upload.'}, 415)
        try:
            name = unquote(request.headers.get('x-file-name', ''), errors='strict')
            content_length = int(request.headers.get('content-length', '0'))
        except (ValueError, UnicodeError):
            return JSONResponse({'error': 'Invalid upload headers.'}, 400)
        if not name or len(name) > 240 or re.search(r'[\x00-\x1f/\\]', name) or not re.search(r'\.(pdf|epub|txt)$', name, re.I):
            return JSONResponse({'error': 'Choose a PDF, EPUB or TXT file with a valid name.'}, 400)
        if content_length > MAX_BYTES:
            return JSONResponse({'error': 'File exceeds the 25 MB limit.'}, 413)
        receiving += 1
        try:
            data = bytearray()
            async with asyncio.timeout(30):
                async for part in request.stream():
                    if len(data) + len(part) > MAX_BYTES:
                        return JSONResponse({'error': 'File exceeds the 25 MB limit.'}, 413)
                    data.extend(part)
            if not data:
                return JSONResponse({'error': 'File is empty.'}, 400)
            job_id = secrets.token_urlsafe(24)
            connection, child = context.Pipe(duplex=False)
            process = context.Process(target=extraction_worker, args=(child, bytes(data), name), daemon=True)
            try:
                process.start()
            except Exception:
                connection.close()
                child.close()
                return JSONResponse({'error': 'Could not start extraction. Please retry.'}, 500)
            child.close()
            job = Job(job_id, request.state.session, name, process=process)
            jobs[job_id] = job
            job.task = asyncio.create_task(monitor(job, connection))
            return {'id': job_id}
        except TimeoutError:
            return JSONResponse({'error': 'Upload timed out. Try again.'}, 408)
        finally:
            receiving -= 1

    @app.get('/api/books/{job_id}')
    async def book_status(job_id: str, request: Request):
        job = jobs.get(job_id)
        if not job or job.owner != request.state.session:
            return JSONResponse({'error': 'Book not found or this session has expired. Upload it again.'}, 404)
        job.touched = time.monotonic()
        result = {key: getattr(job, key) for key in ('id', 'name', 'stage', 'progress', 'detail', 'error')}
        if job.book:
            result.update({key: job.book[key] for key in ('format', 'bytes', 'sectionCount', 'characters', 'warnings', 'ocrPageCount', 'ocrAttemptedPages', 'ocrLowConfidencePages')})
            result['chunkCount'] = len(job.book['chunks'])
            result['preview'] = [{key: chunk[key] for key in ('id', 'text', 'label', 'page', 'section', 'extraction', 'ocrConfidence') if key in chunk} for chunk in job.book['chunks'][:3]]
        return result

    @app.delete('/api/books/{job_id}')
    async def delete_book(job_id: str, request: Request):
        job = jobs.get(job_id)
        if not job or job.owner != request.state.session:
            return JSONResponse({'error': 'Book not found or this session has expired. Upload it again.'}, 404)
        await dispose(job_id)
        return {'deleted': True}

    @app.get('/api/config')
    async def configuration():
        key, model = answers.settings()
        return {'geminiConfigured': bool(key), 'model': model, 'embeddingModel': retrieval.MODEL_NAME}

    def owned_job(job_id, request):
        job = jobs.get(job_id)
        return job if job and job.owner == request.state.session else None

    @app.get('/api/books/{job_id}/chat/history')
    async def chat_history(job_id: str, request: Request):
        job = owned_job(job_id, request)
        if not job:
            return JSONResponse({'error': 'Book not found or session expired.'}, 404)
        job.touched = time.monotonic()
        return {'turns': job.turns}

    @app.post('/api/books/{job_id}/chat/cancel')
    async def cancel_chat(job_id: str, request: Request):
        job = owned_job(job_id, request)
        if not job:
            return JSONResponse({'error': 'Book not found or session expired.'}, 404)
        if job.stream_cancel:
            job.stream_cancel.set()
        return {'stopping': bool(job.stream_cancel)}

    @app.post('/api/books/{job_id}/chat/clear')
    async def clear_chat(job_id: str, request: Request):
        job = owned_job(job_id, request)
        if not job:
            return JSONResponse({'error': 'Book not found or session expired.'}, 404)
        if job.answering:
            return JSONResponse({'error': 'Stop the current response before clearing the conversation.'}, 409)
        job.turns.clear()
        return {'cleared': True}

    @app.post('/api/books/{job_id}/chat/stream')
    async def stream_chat(job_id: str, body: ChatQuestion, request: Request):
        job = owned_job(job_id, request)
        if not job:
            return JSONResponse({'error': 'Book not found or session expired.'}, 404)
        if job.stage != 'ready' or not job.book:
            return JSONResponse({'error': 'Wait until indexing finishes.'}, 409)
        if job.answering:
            return JSONResponse({'error': 'Another question is already running for this book.'}, 429)
        if time.monotonic() - job.last_answer < 5:
            return JSONResponse({'error': 'Wait a few seconds before asking again.'}, 429)
        if not answers.settings()[0]:
            return JSONResponse({'error': 'Add GEMINI_API_KEY to .env to enable answers.'}, 503)
        if body.retryTurnId:
            if not job.turns or job.turns[-1]['id'] != body.retryTurnId or job.turns[-1]['status'] not in {'failed', 'stopped'}:
                return JSONResponse({'error': 'Only the most recent failed or stopped response can be retried.'}, 409)
            turn = job.turns[-1]
            turn.update(status='running', error=None)
        else:
            turn = {'id': secrets.token_urlsafe(18), 'question': body.question, 'status': 'running'}
            job.turns.append(turn)
            del job.turns[:-40]
        job.answering = True
        job.touched = time.monotonic()
        cancel_event = asyncio.Event()
        job.stream_cancel = cancel_event

        async def events():
            iterator, pending, cancel_wait = None, None, None
            try:
                yield json.dumps({'event': 'started', 'turn': turn}) + '\n'
                chunks = await asyncio.to_thread(conversation.retrieve_for_turn, job.book, turn['question'], job.turns[:-1])
                if cancel_event.is_set():
                    turn['status'] = 'stopped'
                    yield json.dumps({'event': 'stopped', 'turnId': turn['id']}) + '\n'
                    return
                job.last_answer = time.monotonic()
                yield json.dumps({'event': 'status', 'text': 'Writing a draft; citations will be checked when complete.'}) + '\n'
                iterator = answers.stream_answer(turn['question'], chunks, conversation.completed_context(job.turns[:-1]))
                cancel_wait = asyncio.create_task(cancel_event.wait())
                while True:
                    pending = asyncio.create_task(anext(iterator))
                    done, _ = await asyncio.wait({pending, cancel_wait}, timeout=60, return_when=asyncio.FIRST_COMPLETED)
                    if cancel_wait in done:
                        turn['status'] = 'stopped'
                        yield json.dumps({'event': 'stopped', 'turnId': turn['id']}) + '\n'
                        return
                    if not done:
                        raise answers.RagError('The response stalled. Please retry.', 504)
                    try:
                        message = pending.result()
                    except StopAsyncIteration:
                        raise answers.RagError('The response ended before verification. Please retry.') from None
                    if message['event'] == 'final':
                        turn.update(status='complete', result=message['result'])
                        yield json.dumps({**message, 'turnId': turn['id']}, ensure_ascii=False) + '\n'
                        return
                    yield json.dumps(message, ensure_ascii=False) + '\n'
            except asyncio.CancelledError:
                turn['status'] = 'stopped'
                raise
            except Exception as exc:
                message = str(exc) if isinstance(exc, answers.RagError) else 'The response failed. Please retry.'
                turn.update(status='failed', error=message)
                yield json.dumps({'event': 'error', 'error': message, 'turnId': turn['id']}) + '\n'
            finally:
                for task in (pending, cancel_wait):
                    if task and not task.done():
                        task.cancel()
                    if task:
                        with suppress(asyncio.CancelledError, Exception):
                            await task
                if iterator:
                    with suppress(Exception):
                        await iterator.aclose()
                if turn['status'] == 'running':
                    turn['status'] = 'stopped'
                job.stream_cancel = None
                job.answering = False

        return StreamingResponse(events(), media_type='application/x-ndjson', headers={'X-Accel-Buffering': 'no'})

    @app.post('/api/books/{job_id}/{action}')
    async def question(job_id: str, action: str, body: Question, request: Request):
        if action not in {'search', 'ask'}:
            return JSONResponse({'error': 'Not found.'}, 404)
        job = jobs.get(job_id)
        if not job or job.owner != request.state.session:
            return JSONResponse({'error': 'Book not found or this session has expired. Upload it again.'}, 404)
        if job.stage != 'ready' or not job.book:
            return JSONResponse({'error': 'Wait until the book finishes indexing.'}, 409)
        if job.answering:
            return JSONResponse({'error': 'A question is already being processed for this book.'}, 429)
        if action == 'ask' and time.monotonic() - job.last_answer < 5:
            return JSONResponse({'error': 'Please wait a few seconds before asking another question.'}, 429)
        if action == 'ask' and not answers.settings()[0]:
            return JSONResponse({'error': 'Add GEMINI_API_KEY to .env to enable answers. You can search passages locally now.'}, 503)
        job.answering = True
        job.touched = time.monotonic()
        try:
            chunks = await asyncio.to_thread(retrieval.retrieve, job.book, body.question)
            if jobs.get(job_id) is not job:
                return JSONResponse({'error': 'This book was removed.'}, 404)
            if action == 'search':
                return {'sources': retrieval.sources_for(chunks), 'localOnly': True}
            job.last_answer = time.monotonic()
            result = await answers.generate_answer(body.question, chunks)
            if jobs.get(job_id) is not job:
                return JSONResponse({'error': 'This book was removed.'}, 404)
            return result
        except answers.RagError as exc:
            return JSONResponse({'error': str(exc)}, exc.status)
        except Exception:
            return JSONResponse({'error': 'Local retrieval failed. Try again or upload the book again.'}, 500)
        finally:
            job.answering = False

    @app.api_route('/', methods=['GET', 'HEAD'])
    async def index():
        return FileResponse(PUBLIC / 'index.html')

    @app.api_route('/{asset}', methods=['GET', 'HEAD'])
    async def asset(asset: str):
        if asset not in {'app.js', 'styles.css'}:
            return JSONResponse({'error': 'Not found.'}, 404)
        return FileResponse(PUBLIC / asset, media_type='text/javascript' if asset.endswith('.js') else 'text/css')

    return app


app = create_app()
