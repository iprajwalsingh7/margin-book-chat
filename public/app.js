const input = document.querySelector('#book-input');
const dropzone = document.querySelector('#dropzone');
const error = document.querySelector('#file-error');
const selected = document.querySelector('#selected-book');
const status = document.querySelector('#chat-status');
const badge = document.querySelector('#book-state');
const progress = document.querySelector('#processing-progress');
const detail = document.querySelector('#processing-detail');
const preview = document.querySelector('#extraction-preview');
const retry = document.querySelector('#retry-book');
let currentId = null;
let currentFile = null;
let revision = 0;
let pollTimer;
let bookReady = false;
let busy = false;
let providerConfigured = false;
let activeController = null;
let historyTimer;
let historyRestoredFor = null;
let activeTurnId = null;
const turnViews = new Map();
const stopButton = document.querySelector('#stop-answer');
const retryAnswerButton = document.querySelector('#retry-answer');
const clearChatButton = document.querySelector('#clear-chat');
const questionInput = document.querySelector('#question');
const askButton = document.querySelector('#ask-question');
const searchButton = document.querySelector('#search-passages');
const answerPanel = document.querySelector('#answer-panel');
const questionProgress = document.querySelector('#question-progress');

function updateComposer() {
  questionInput.disabled = !bookReady || busy;
  askButton.disabled = !bookReady || busy || !providerConfigured;
  searchButton.disabled = !bookReady || busy;
  stopButton.hidden = !busy;
  clearChatButton.disabled = !bookReady || busy;
  retryAnswerButton.disabled = !bookReady || busy;
}
function resetQuestions() {
  activeController?.abort();
  activeController = null;
  clearTimeout(historyTimer);
  historyRestoredFor = null;
  activeTurnId = null;
  turnViews.clear();
  retryAnswerButton.hidden = true;
  bookReady = false;
  busy = false;
  answerPanel.hidden = true;
  answerPanel.replaceChildren();
  questionProgress.textContent = '';
  questionInput.value = '';
  updateComposer();
}
async function checkProvider() {
  try {
    const config = await api('/api/config');
    providerConfigured = config.geminiConfigured;
    document.querySelector('#model-status').textContent = providerConfigured
      ? `Gemini key configured · ${config.model} · local passage search available`
      : 'Gemini key needed: add GEMINI_API_KEY to .env. Local passage search is available without a key.';
    updateComposer();
  } catch {
    document.querySelector('#model-status').textContent = 'Could not check Gemini configuration. Refresh to retry.';
  }
}

async function api(path, options) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Request failed. Try again.');
  return data;
}
async function removeRemote(id) {
  if (!id) return;
  const response = await fetch(`/api/books/${id}`, { method: 'DELETE' });
  if (!response.ok && response.status !== 404) throw new Error('Could not remove the previous book. Try again.');
}
function setStage(stage, message, percent) {
  badge.textContent = stage;
  detail.textContent = message;
  if (percent === null) progress.removeAttribute('value');
  else progress.value = percent;
  document.querySelector('#source-eyebrow').textContent = stage === 'Text ready' ? 'SOURCE MATERIAL PREPARED' : 'SOURCE PROCESSING';
  document.querySelector('#source-heading').textContent = stage === 'Text ready' ? 'Your evidence is ready.' : 'Preparing your source.';
}
function showFailure(message) {
  bookReady = false;
  updateComposer();
  setStage('Processing failed', message, 0);
  document.querySelector('#source-eyebrow').textContent = 'SOURCE NEEDS ATTENTION';
  document.querySelector('#source-heading').textContent = 'Unable to read this source.';
  error.textContent = message;
  retry.hidden = !currentFile;
  status.textContent = 'This book could not be prepared. Try another file or retry the upload.';
}
function remember(id) {
  try { if (id) sessionStorage.setItem('margin-book', id); else sessionStorage.removeItem('margin-book'); } catch { /* Storage may be disabled. */ }
}
function renderBook(book) {
  document.querySelector('#book-name').textContent = book.name;
  if (book.bytes) document.querySelector('#book-size').textContent = `${(book.bytes / 1024 / 1024).toFixed(2)} MB · ${book.format.toUpperCase()}`;
  setStage(book.stage === 'ready' ? 'Text ready' : book.stage === 'ocr' ? 'Reading scan with OCR' : book.stage, book.detail, book.progress);
  if (book.stage !== 'ready') return;
  bookReady = true;
  if (historyRestoredFor !== currentId) {
    historyRestoredFor = currentId;
    restoreHistory(currentId, revision);
  }
  updateComposer();
  checkProvider();
  status.textContent = `${book.chunkCount.toLocaleString()} ${book.chunkCount === 1 ? 'passage' : 'passages'} indexed locally. Ask a focused question or find supporting passages.`;
  const unit = book.format === 'pdf' ? 'page' : 'section';
  document.querySelector('#extraction-stats').textContent = `${book.sectionCount.toLocaleString()} ${unit}${book.sectionCount === 1 ? '' : 's'} · ${book.characters.toLocaleString()} characters · ${book.chunkCount.toLocaleString()} ${book.chunkCount === 1 ? 'passage' : 'passages'}${book.ocrPageCount ? ` · ${book.ocrPageCount} OCR page(s)` : ''}`;
  document.querySelector('#extraction-warnings').textContent = book.warnings.join(' ');
  const list = document.querySelector('#passage-list');
  list.replaceChildren();
  for (const passage of book.preview) {
    const article = document.createElement('article');
    const heading = document.createElement('h4');
    const text = document.createElement('p');
    heading.textContent = `${passage.label} / ${passage.id}`;
    if (passage.extraction === 'ocr') {
      heading.textContent += ` / OCR${passage.ocrConfidence != null ? ` · recognition confidence ${Math.round(passage.ocrConfidence * 100)}%` : ''}`;
    }
    text.textContent = passage.text;
    article.append(heading, text);
    list.append(article);
  }
  preview.hidden = false;
}
async function poll(id, token, failures = 0) {
  if (token !== revision) return;
  try {
    const book = await api(`/api/books/${id}`);
    if (token !== revision) return;
    renderBook(book);
    if (book.stage === 'failed') { showFailure(book.error); return; }
    if (book.stage !== 'ready') pollTimer = setTimeout(() => poll(id, token), 500);
  } catch (err) {
    if (token !== revision) return;
    if (failures < 2) pollTimer = setTimeout(() => poll(id, token, failures + 1), 1000);
    else showFailure(err.message);
  }
}
async function selectBook(file) {
  error.textContent = '';
  if (!file) return;
  if (!/\.(pdf|epub|txt)$/i.test(file.name)) { error.textContent = 'Choose a PDF, EPUB or TXT file.'; input.value = ''; return; }
  if (!file.size || file.size > 25 * 1024 * 1024) { error.textContent = 'Choose a non-empty file no larger than 25 MB.'; input.value = ''; return; }
  const token = ++revision;
  resetQuestions();
  clearTimeout(pollTimer);
  const previousId = currentId;
  currentId = null;
  remember(null);
  currentFile = file;
  selected.hidden = false;
  preview.hidden = true;
  retry.hidden = true;
  document.querySelector('#book-name').textContent = file.name;
  document.querySelector('#book-size').textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB · ${file.name.split('.').pop().toUpperCase()}`;
  setStage('Uploading', 'Sending to your local processing server…', null);
  status.textContent = 'Preparing your source material. You can inspect the extracted passages when processing finishes.';
  try {
    await removeRemote(previousId);
    if (token !== revision) return;
    const result = await api('/api/books', { method: 'POST', headers: { 'Content-Type': 'application/octet-stream', 'X-File-Name': encodeURIComponent(file.name) }, body: file });
    if (token !== revision) { await removeRemote(result.id); return; }
    currentId = result.id;
    remember(currentId);
    await poll(currentId, token);
  } catch (err) { if (token === revision) showFailure(err.message); }
}
input.addEventListener('change', () => selectBook(input.files[0]));
for (const event of ['dragover', 'dragleave', 'drop']) {
  dropzone.addEventListener(event, (e) => {
    e.preventDefault();
    dropzone.classList.toggle('dragging', event === 'dragover');
    if (event === 'drop') {
      if (e.dataTransfer.files.length !== 1) { error.textContent = 'Choose one book at a time.'; return; }
      selectBook(e.dataTransfer.files[0]);
    }
  });
}
retry.addEventListener('click', () => selectBook(currentFile));
document.querySelector('#remove-book').addEventListener('click', async () => {
  ++revision;
  resetQuestions();
  clearTimeout(pollTimer);
  const id = currentId;
  currentId = null;
  currentFile = null;
  remember(null);
  input.value = '';
  selected.hidden = true;
  preview.hidden = true;
  error.textContent = '';
  status.textContent = 'Select a book to prepare your research workspace.';
  progress.value = 0;
  detail.textContent = 'Ready for a source document';
  document.querySelector('#source-eyebrow').textContent = 'AWAITING SOURCE MATERIAL';
  document.querySelector('#source-heading').textContent = 'Start with a source.';
  input.focus();
  try { await removeRemote(id); } catch { error.textContent = 'Could not confirm server deletion. Temporary data expires after one hour.'; }
});

function displayResult(result, searchOnly, target = answerPanel) {
  target.replaceChildren();
  target.hidden = false;
  const title = document.createElement('h3');
  title.textContent = searchOnly ? 'Matching passages / local search' : result.mode === 'overview' ? 'Book overview / Gemini' : 'Answer / Gemini';
  const answer = document.createElement('p');
  answer.textContent = searchOnly
    ? (result.sources.length ? 'These are retrieved passages, not a generated answer. Expand a source to inspect it.' : 'No sufficiently relevant passages found. Try a more specific question.')
    : result.answer;
  target.append(title, answer);
  for (const [index, source] of result.sources.entries()) {
    const item = document.createElement('details');
    const summary = document.createElement('summary');
    summary.textContent = `[${source.number || index + 1}] ${source.source} · ${source.label}${source.extraction === 'ocr' ? ' · OCR' : ''}`;
    const quote = document.createElement('blockquote');
    quote.textContent = source.quotes ? source.quotes.join('\n\n') : source.text;
    const metadata = document.createElement('p');
    metadata.textContent = `${source.id} · ${source.extraction === 'ocr' ? 'Recognized from a scan; verify against the page.' : 'Extracted from your book.'}`;
    item.append(summary, quote, metadata);
    target.append(item);
  }
}
function turnView(turn) {
  let view = turnViews.get(turn.id);
  if (!view) {
    const article = document.createElement('article');
    article.className = 'chat-turn';
    const question = document.createElement('h3');
    question.className = 'user-question';
    question.textContent = turn.question;
    const badge = document.createElement('p');
    badge.className = 'turn-status';
    const body = document.createElement('div');
    body.className = 'turn-answer';
    article.append(question, badge, body);
    answerPanel.append(article);
    answerPanel.hidden = false;
    view = {article, body, badge};
    turnViews.set(turn.id, view);
  }
  return view;
}
function renderStoredTurn(turn) {
  const view = turnView(turn);
  if (turn.status === 'complete') {
    view.badge.textContent = turn.result.supported ? 'Complete · source quotations checked' : 'Complete · insufficient supporting evidence';
    displayResult(turn.result, false, view.body);
  } else {
    view.badge.textContent = turn.status === 'running' ? 'Response in progress…' : turn.status === 'stopped' ? 'Stopped · incomplete draft discarded' : 'Response failed';
    view.body.textContent = turn.error || '';
  }
}
async function restoreHistory(id, token) {
  if (activeController || token !== revision) return;
  busy = true;
  updateComposer();
  try {
    const {turns} = await api(`/api/books/${id}/chat/history`);
    if (token !== revision || id !== currentId || activeController) return;
    for (const turn of turns) renderStoredTurn(turn);
    const last = turns.at(-1);
    busy = last?.status === 'running';
    activeTurnId = last?.id || null;
    retryAnswerButton.hidden = !last || !['failed', 'stopped'].includes(last.status);
    updateComposer();
    if (busy) historyTimer = setTimeout(() => restoreHistory(id, token), 600);
  } catch (err) {
    if (token === revision && !activeController) {
      busy = false;
      updateComposer();
      questionProgress.textContent = err.message;
    }
  }
}
async function submitQuestion(searchOnly = false, retryTurnId = null) {
  const question = retryTurnId ? (questionInput.value.trim() || 'Retry previous question') : questionInput.value.trim();
  if (!bookReady || busy) return;
  if (question.length < 3) { questionProgress.textContent = 'Enter at least three characters.'; return; }
  const token = revision;
  const id = currentId;
  const controller = new AbortController();
  activeController = controller;
  let terminal = false;
  let view = null;
  busy = true;
  retryAnswerButton.hidden = true;
  updateComposer();
  questionProgress.textContent = searchOnly ? 'Searching locally…' : 'Retrieving evidence…';
  try {
    if (searchOnly) {
      const result = await api(`/api/books/${id}/search`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({question}), signal:controller.signal});
      if (token !== revision) return;
      view = turnView({id:crypto.randomUUID(), question});
      view.badge.textContent = 'Local search · not added to conversation context';
      displayResult(result, true, view.body);
      questionProgress.textContent = 'Search completed on your computer.';
      return;
    }
    const response = await fetch(`/api/books/${id}/chat/stream`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question, retryTurnId}), signal:controller.signal,
    });
    if (!response.ok) { const data = await response.json(); throw new Error(data.error || 'Could not start the answer.'); }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    function handle(message) {
      if (token !== revision) return;
      if (message.event === 'started') {
        activeTurnId = message.turn.id;
        view = turnView(message.turn);
        view.body.replaceChildren();
        view.badge.textContent = 'Draft streaming · not yet verified';
        questionInput.value = '';
      } else if (message.event === 'status') {
        questionProgress.textContent = message.text;
      } else if (message.event === 'draft' && view) {
        view.body.textContent = message.text;
        view.badge.textContent = 'Draft · citations not yet verified';
      } else if (message.event === 'final' && view) {
        terminal = true;
        view.badge.textContent = message.result.supported ? 'Complete · source quotations checked' : 'Complete · insufficient supporting evidence';
        displayResult(message.result, false, view.body);
        questionProgress.textContent = 'Answer complete. You can ask a follow-up.';
      } else if (message.event === 'error' || message.event === 'stopped') {
        terminal = true;
        if (view) {
          view.body.textContent = message.error || '';
          view.badge.textContent = message.event === 'stopped' ? 'Stopped · incomplete draft discarded' : 'Response failed · draft discarded';
        }
        retryAnswerButton.hidden = false;
        questionProgress.textContent = message.error || 'Response stopped. You can retry.';
      }
    }
    while (true) {
      const {value, done} = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), {stream:!done});
      let newline;
      while ((newline = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, newline);
        buffer = buffer.slice(newline + 1);
        if (line.trim()) handle(JSON.parse(line));
      }
      if (done) break;
    }
    if (!terminal) throw new Error('The connection ended before the answer was verified. Retry the response.');
  } catch (err) {
    if (token !== revision) return;
    if (view && !terminal) {
      view.body.textContent = '';
      view.badge.textContent = err.name === 'AbortError' ? 'Stopped · incomplete draft discarded' : 'Interrupted · draft discarded';
      retryAnswerButton.hidden = searchOnly;
    }
    questionProgress.textContent = err.name === 'AbortError' ? 'Response stopped.' : err.message;
  } finally {
    if (token === revision) {
      busy = false;
      activeController = null;
      updateComposer();
      if (!searchOnly && !terminal) historyTimer = setTimeout(() => restoreHistory(id, token), 400);
    }
  }
}
stopButton.addEventListener('click', async () => {
  const id = currentId;
  const controller = activeController;
  questionProgress.textContent = 'Stopping…';
  try { await api(`/api/books/${id}/chat/cancel`, {method:'POST'}); }
  catch { controller?.abort(); }
  if (controller) controller.abort();
  else restoreHistory(id, revision);
});
retryAnswerButton.addEventListener('click', () => submitQuestion(false, activeTurnId));
clearChatButton.addEventListener('click', async () => {
  try {
    await api(`/api/books/${currentId}/chat/clear`, {method:'POST'});
    answerPanel.replaceChildren();
    answerPanel.hidden = true;
    turnViews.clear();
    activeTurnId = null;
    retryAnswerButton.hidden = true;
    questionProgress.textContent = 'Conversation cleared.';
  } catch (err) { questionProgress.textContent = err.message; }
});
askButton.addEventListener('click', () => submitQuestion(false));
searchButton.addEventListener('click', () => submitQuestion(true));
questionInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    submitQuestion(!providerConfigured);
  }
});
checkProvider();
let savedId;
try { savedId = sessionStorage.getItem('margin-book'); } catch { /* Optional reload recovery. */ }
if (savedId) {
  currentId = savedId;
  selected.hidden = false;
  document.querySelector('#book-name').textContent = 'Restoring your source…';
  poll(savedId, revision);
}
