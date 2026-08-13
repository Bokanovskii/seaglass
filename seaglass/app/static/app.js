const token = (() => {
  const hash = window.location.hash.replace(/^#/, '');
  if (hash) {
    sessionStorage.setItem('seaglass-token', hash);
    history.replaceState(null, '', window.location.pathname + window.location.search);
  }
  return sessionStorage.getItem('seaglass-token') || '';
})();

const state = {
  assist: 'auto',
  people: [],
  selectedChat: null,
  requestId: null,
  assistToken: null,
  results: [],
};

const els = {
  loading: document.getElementById('loading-screen'),
  buildScreen: document.getElementById('build-screen'),
  buildIdle: document.getElementById('build-idle'),
  buildStart: document.getElementById('build-start'),
  buildProgress: document.getElementById('build-progress'),
  buildStageText: document.getElementById('build-stage-text'),
  buildProgressBar: document.getElementById('build-progress-bar'),
  buildDetail: document.getElementById('build-detail'),
  buildError: document.getElementById('build-error'),
  app: document.getElementById('app'),
  progressBar: document.getElementById('progress-bar'),
  warmupSteps: document.getElementById('warmup-steps'),
  loadingError: document.getElementById('loading-error'),
  query: document.getElementById('query'),
  searchButton: document.getElementById('search-button'),
  results: document.getElementById('results'),
  resultsMeta: document.getElementById('results-meta'),
  statusBar: document.getElementById('status-bar'),
  assistBanner: document.getElementById('assist-banner'),
  syncBanner: document.getElementById('sync-banner'),
  peopleInput: document.getElementById('people-input'),
  peopleSuggestions: document.getElementById('people-suggestions'),
  peopleChips: document.getElementById('people-chips'),
  chatInput: document.getElementById('chat-input'),
  chatSuggestions: document.getElementById('chat-suggestions'),
  drawer: document.getElementById('drawer'),
  drawerContent: document.getElementById('drawer-content'),
  dateFrom: document.getElementById('date-from'),
  dateTo: document.getElementById('date-to'),
  hasMedia: document.getElementById('has-media'),
};

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set('Authorization', `Bearer ${token}`);
  if (options.body && !headers.has('Content-Type')) headers.set('Content-Type', 'application/json');
  const response = await fetch(path, {...options, headers});
  if (response.status === 204) return null;
  const payload = await response.json();
  if (!response.ok) {
    const err = new Error(payload.detail || `Request failed: ${response.status}`);
    err.status = response.status;
    throw err;
  }
  return payload;
}

async function pollHealth() {
  const health = await api('/api/health');
  if (health.state === 'NEEDS_INDEX' && !(health.build && health.build.running)) {
    els.loading.classList.add('hidden');
    els.app.classList.add('hidden');
    els.buildScreen.classList.remove('hidden');
    els.buildIdle.classList.remove('hidden');
    els.buildProgress.classList.add('hidden');
    return;
  }
  if (health.build && health.build.running) {
    showBuildProgress(health.build);
    setTimeout(pollHealth, 500);
    return;
  }
  els.progressBar.style.width = `${Math.round((health.progress || 0) * 100)}%`;
  els.warmupSteps.innerHTML = (health.steps || []).map(step => {
    const stepState = step.state || 'pending';
    const elapsed = step.elapsed_s?.toFixed?.(2) ?? step.elapsed_s ?? 0;
    return `<li class="step step-${escapeHtml(stepState)}"><span class="step-icon">${symbolFor(stepState)}</span><span class="step-name">${escapeHtml(step.name || '')}</span><span class="step-time">${escapeHtml(elapsed)}s</span></li>`;
  }).join('');
  if (health.error) {
    els.loadingError.textContent = health.error;
    els.loadingError.classList.remove('hidden');
  }
  if (health.state === 'READY' || health.state === 'DEGRADED') {
    els.loading.classList.add('hidden');
    els.buildScreen.classList.add('hidden');
    els.app.classList.remove('hidden');
    await refreshStatus();
    if (!window.__statusPollStarted) {
      window.__statusPollStarted = true;
      setInterval(refreshStatus, 60000);
    }
    return;
  }
  setTimeout(pollHealth, 250);
}

function showBuildProgress(build) {
  els.loading.classList.add('hidden');
  els.app.classList.add('hidden');
  els.buildScreen.classList.remove('hidden');
  els.buildIdle.classList.add('hidden');
  els.buildProgress.classList.remove('hidden');
  const stageLabels = {
    snapshotting: 'Snapshotting chat.db (safe, read-only copy)…',
    building: 'Embedding and indexing messages… this can take a while for large histories.',
    idle: 'Starting…',
  };
  els.buildStageText.textContent = stageLabels[build.stage] || build.stage;
  els.buildDetail.textContent = build.chunks_written ? `${build.chunks_written} chunks written so far · ${build.elapsed_s ?? 0}s elapsed` : `${build.elapsed_s ?? 0}s elapsed`;
}

async function startBuild() {
  els.buildError.classList.add('hidden');
  try {
    await api('/api/index/build', {method: 'POST'});
  } catch (err) {
    els.buildError.textContent = err.message;
    els.buildError.classList.remove('hidden');
    return;
  }
  els.buildIdle.classList.add('hidden');
  els.buildProgress.classList.remove('hidden');
  setTimeout(pollHealth, 250);
}

if (els.buildStart) els.buildStart.onclick = startBuild;

function symbolFor(state) {
  return state === 'done' ? '✓' : state === 'failed' ? '⚠' : state === 'running' ? '⟳' : '•';
}

// Owned by pollSyncProgress(): while a sync poll loop is live it owns the
// sync banner, and the 60s refreshStatus() tick must not stomp on it.
let syncInProgress = false;
// Incremented on every syncNow(); a poll loop whose token is stale exits so
// double-clicks / multiple entry points can never stack concurrent loops.
let syncPollToken = 0;

async function refreshStatus() {
  const status = await api('/api/status');
  const stale = status.n_messages_since_index
    ? `<span class="status-sep">·</span><span class="status-stale">⚠ ${status.n_messages_since_index.toLocaleString()} msgs since last index build</span>`
    : '';
  els.statusBar.innerHTML = `<span class="status-dot${status.hydration_available ? '' : ' is-off'}"></span><span>Ready</span><span class="status-sep">·</span><span>${Number(status.n_chunks || 0).toLocaleString()} chunks</span><span class="status-sep">·</span><span>${Number(status.n_chats || 0).toLocaleString()} chats</span>${stale}`;
  if (syncInProgress) return;  // the sync poll loop owns the banner right now
  if (status.n_messages_since_index > 0) {
    els.syncBanner.classList.remove('hidden');
    const plural = status.n_messages_since_index === 1 ? '' : 's';
    els.syncBanner.innerHTML = `<span class="banner-icon">${iconSvg('i-refresh')}</span><span class="banner-text"><strong>${status.n_messages_since_index.toLocaleString()} new message${plural}</strong> since the index was last built.</span><span class="banner-actions"><button id="sync-now" class="btn btn-sm btn-primary">Sync now</button></span>`;
    document.getElementById('sync-now').onclick = syncNow;
  } else {
    els.syncBanner.classList.add('hidden');
  }
}

function syncBannerMessage(text, {spin = false} = {}) {
  els.syncBanner.classList.remove('hidden');
  els.syncBanner.innerHTML = `<span class="banner-icon${spin ? ' spin' : ''}">${iconSvg('i-refresh')}</span><span class="banner-text">${escapeHtml(text)}</span>`;
}

async function syncNow() {
  if (syncInProgress) return;  // guard against double-clicks stacking poll loops
  syncInProgress = true;
  const token = ++syncPollToken;
  const button = document.getElementById('sync-now');
  if (button) { button.disabled = true; button.textContent = 'Syncing…'; }
  syncBannerMessage('Syncing…', {spin: true});
  try {
    await api('/api/index/build', {method: 'POST'});
  } catch (err) {
    if (err.status !== 409) {
      // Hard failure to even start: release ownership and surface it, then
      // let the next refreshStatus() tick restore a clickable banner.
      syncInProgress = false;
      syncBannerMessage(`Sync failed: ${err.message}`);
      return;
    }
    // 409: a build/sync was already in progress -- fine, just watch it.
  }
  pollSyncProgress(token);
}

// `failures` counts consecutive poll errors; transient network blips during a
// long build shouldn't tear down the whole progress UI.
const SYNC_POLL_MAX_FAILURES = 3;

async function pollSyncProgress(token = syncPollToken, failures = 0) {
  if (token !== syncPollToken) return;  // superseded by a newer sync run
  let build;
  try {
    build = await api('/api/index/build');
  } catch (err) {
    if (token !== syncPollToken) return;
    if (failures + 1 < SYNC_POLL_MAX_FAILURES) {
      setTimeout(() => pollSyncProgress(token, failures + 1), 750 * (failures + 1));
      return;
    }
    syncInProgress = false;
    syncBannerMessage(`Lost contact while syncing: ${err.message}`);
    return;
  }
  if (token !== syncPollToken) return;
  if (build.running) {
    const elapsed = build.elapsed_s ?? 0;
    const detail = build.chunks_written
      ? `${build.chunks_written.toLocaleString()} chunks written so far · ${elapsed}s elapsed`
      : `${elapsed}s elapsed`;
    syncBannerMessage(`Syncing… ${detail}`, {spin: true});
    setTimeout(() => pollSyncProgress(token, 0), 750);
    return;
  }
  if (build.stage === 'failed') {
    syncInProgress = false;
    syncBannerMessage(`Sync failed: ${build.error || 'unknown error'}`);
    return;
  }
  // Not running and not failed: either the run finished ('done'), or the
  // background thread hasn't flipped running=true yet (we polled in the gap
  // between POST returning and the worker starting -- in which case `stage`
  // is still 'idle'). Grace-poll a few times before declaring completion.
  if (build.stage === 'idle' && failures < 4) {
    setTimeout(() => pollSyncProgress(token, failures + 1), 750);
    return;
  }
  // Done: release banner ownership and let refreshStatus() render the truth
  // (hidden banner when nothing is stale, a fresh "Sync now" button if not).
  syncInProgress = false;
  try {
    await refreshStatus();
  } catch (err) {
    syncBannerMessage(`Sync finished, but status refresh failed: ${err.message}`);
  }
}

function buildFilters() {
  const groupValue = document.querySelector('input[name="group"]:checked').value;
  return {
    people_handles: state.people.flatMap(person => person.handles),
    is_group: groupValue === '' ? null : groupValue === 'true',
    chat_ids: state.selectedChat ? [state.selectedChat.chat_id] : null,
    date_from: els.dateFrom.value ? new Date(`${els.dateFrom.value}T00:00:00`).getTime() / 1000 : null,
    date_to: els.dateTo.value ? new Date(`${els.dateTo.value}T23:59:59`).getTime() / 1000 : null,
    has_media: els.hasMedia.checked ? true : null,
  };
}

async function search() {
  state.requestId = crypto.randomUUID();
  setBusy(true);
  let payload;
  try {
    payload = await api('/api/search', {
      method: 'POST',
      body: JSON.stringify({ query: els.query.value, filters: buildFilters(), options: {}, assist: state.assist, request_id: state.requestId })
    });
  } finally {
    setBusy(false);
  }
  if (payload.request_id !== state.requestId) return;
  renderResults(payload);
  if (payload.assist_token) {
    state.assistToken = payload.assist_token;
    pollAssist(payload.assist_token);
  }
}

function setBusy(busy) {
  els.searchButton.classList.toggle('is-busy', busy);
  els.searchButton.disabled = busy;
  els.results.classList.toggle('is-busy', busy);
}

function iconSvg(name, className = 'icon') {
  return `<svg class="${className}" aria-hidden="true"><use href="#${name}"></use></svg>`;
}

function renderResults(payload) {
  state.results = payload.sessions || [];
  const term = payload.effective_filters.semantic || els.query.value;
  els.resultsMeta.innerHTML = [
    term ? `<span class="meta-query">${escapeHtml(term)}</span>` : '',
    `<span class="meta-pill">${payload.n_sessions} session${payload.n_sessions === 1 ? '' : 's'}</span>`,
    `<span class="meta-pill">${payload.n_results} msg${payload.n_results === 1 ? '' : 's'}</span>`,
    `<span class="meta-pill">${payload.elapsed_s}s</span>`,
  ].filter(Boolean).join('');
  if (!payload.sessions?.length) {
    els.results.innerHTML = `<div class="empty-state"><div class="empty-icon">${iconSvg('i-search')}</div><div class="empty-title">No matching messages</div><p class="empty-hint">Nothing matched this search and the filters you have applied. Try loosening a filter, widening the date range, or rephrasing the query.</p></div>`;
    return;
  }
  els.results.innerHTML = payload.sessions.map((session, index) => renderSession(session, index)).join('');
  document.querySelectorAll('[data-chat-id]').forEach(link => link.addEventListener('click', openConversation));
}

function renderSession(session, index) {
  const hits = (session.messages || []).map(renderMessage).join('');
  const contextMessages = session.context_messages || [];
  const context = contextMessages.map(renderMessage).join('');
  const score = Number(session.score || 0);
  const scoreWidth = Math.max(6, Math.min(100, Math.round(score * 100)));
  const title = escapeHtml(session.title || `Chat ${session.chat_id}`);
  const groupBadge = session.is_group ? `<span class="meta-pill">${iconSvg('i-people', 'icon icon-sm')} ${session.participant_count || ''}</span>` : '';
  const contextBlock = context
    ? `<div class="context"><details><summary>Surrounding context <span class="meta-pill">${contextMessages.length}</span></summary><div class="context-body">${context}</div></details></div>`
    : '';
  return `<article class="result-card" data-index="${index}">`
    + `<header class="card-head"><span class="card-title">${title}</span>${groupBadge}<span class="card-day">${escapeHtml(session.day || '')}</span>`
    + `<span class="score-pill" title="relevance score"><span class="score-bar"><i style="width:${scoreWidth}%"></i></span>${score.toFixed(2)}</span></header>`
    + `<div class="card-hits">${hits}</div>`
    + contextBlock
    + `<div class="card-foot-row"><button class="btn btn-sm" data-chat-id="${session.chat_id}" data-around-ts="${session.messages?.[0]?.ts || ''}">${iconSvg('i-arrow-out', 'icon icon-sm')} Open full conversation</button></div>`
    + `</article>`;
}

function renderMessage(message) {
  const isMe = message.is_from_me || !message.sender;
  const sender = message.sender || 'Me';
  const clip = message.has_attachment ? `<span class="msg-clip" title="has attachment">${iconSvg('i-clip', 'icon icon-sm')}</span>` : '';
  return `<div class="msg${isMe ? ' msg-me' : ''}">`
    + `<span class="msg-avatar" style="--avatar-h:${hueFor(sender)}" aria-hidden="true">${escapeHtml(initialsFor(sender))}</span>`
    + `<div class="msg-body"><div class="msg-head"><span class="msg-sender">${escapeHtml(sender)}</span><span class="msg-time">${formatTime(message.ts)}</span>${clip}</div>`
    + `<div class="msg-text">${highlight(message.text || '')}</div></div></div>`;
}

function initialsFor(name) {
  const parts = String(name).trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return '?';
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

function hueFor(name) {
  let hash = 0;
  for (const char of String(name)) hash = (hash * 31 + char.charCodeAt(0)) % 360;
  return hash;
}

function formatTime(ts) {
  return ts ? new Date(ts * 1000).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'}) : '';
}

function highlight(text) {
  let output = escapeHtml(text);
  for (const term of (els.query.value.match(/[A-Za-z0-9]{3,}/g) || [])) {
    const re = new RegExp(`(${term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'ig');
    output = output.replace(re, '<mark>$1</mark>');
  }
  return output;
}

async function pollAssist(tokenValue) {
  const result = await api(`/api/assist/${tokenValue}`);
  if (result === null) {
    setTimeout(() => pollAssist(tokenValue), 1000);
    return;
  }
  if (result.status === 'ready') {
    els.assistBanner.classList.remove('hidden');
    els.assistBanner.innerHTML = `<span class="banner-icon">${iconSvg('i-sparkle')}</span><span class="banner-text">Copilot read this as <span class="assist-parse">${escapeHtml(describeAssist(result))}</span></span><span class="banner-actions"><button id="apply-assist" class="btn btn-sm btn-primary">Apply</button><button id="dismiss-assist" class="btn btn-sm btn-ghost">Dismiss</button></span>`;
    document.getElementById('apply-assist').onclick = () => applyAssist(tokenValue);
    document.getElementById('dismiss-assist').onclick = () => els.assistBanner.classList.add('hidden');
  }
}

function describeAssist(result) {
  const parse = result.parse || {};
  const parts = [];
  if ((parse.people || []).length) parts.push(`people: ${parse.people.join(', ')}`);
  if (parse.date_from && parse.date_to) parts.push(`${parse.date_from} → ${parse.date_to}`);
  if (typeof parse.is_group === 'boolean') parts.push(parse.is_group ? 'group chats' : '1:1 chats');
  if (parse.semantic) parts.push(parse.semantic);
  return parts.join(' · ');
}

async function applyAssist(tokenValue) {
  const payload = await api('/api/search/apply-assist', {
    method: 'POST',
    body: JSON.stringify({ assist_token: tokenValue, query: els.query.value, filters: buildFilters(), options: {} })
  });
  renderResults(payload);
  els.assistBanner.classList.add('hidden');
}

async function openConversation(event) {
  const button = event.currentTarget;
  const params = new URLSearchParams({ chat_id: button.dataset.chatId, limit: '50' });
  if (button.dataset.aroundTs) params.set('around_ts', button.dataset.aroundTs);
  const payload = await api(`/api/conversation?${params.toString()}`);
  els.drawer.classList.remove('hidden');
  const count = payload.messages.length;
  const subtitle = payload.is_group
    ? `Group · ${(payload.participants || []).length || ''} people · ${count} messages`
    : `${count} messages`;
  els.drawerContent.innerHTML = `<header class="drawer-head"><div class="drawer-titles"><div class="drawer-title">${escapeHtml(payload.title || `Chat ${payload.chat_id}`)}</div><div class="drawer-sub">${escapeHtml(subtitle)}</div></div><button id="close-drawer" class="icon-btn" aria-label="Close conversation" title="Close (Esc)">${iconSvg('i-close', 'icon')}</button></header><div class="drawer-messages">${payload.messages.map(renderMessage).join('')}</div>`;
  document.getElementById('close-drawer').onclick = () => els.drawer.classList.add('hidden');
  // The messages container is recreated per open, so it starts at scrollTop 0,
  // but be explicit: WebKit can restore scroll offsets on replaced subtrees.
  const messages = els.drawerContent.querySelector('.drawer-messages');
  if (messages) messages.scrollTop = 0;
}

function renderPeopleChips() {
  els.peopleChips.innerHTML = state.people.map((person, index) => `<button class="chip" data-index="${index}" title="Remove ${escapeHtml(person.display_name)}">${escapeHtml(person.display_name)}<span class="chip-x" aria-hidden="true">×</span></button>`).join('');
  els.peopleChips.querySelectorAll('[data-index]').forEach(button => button.addEventListener('click', () => {
    state.people.splice(Number(button.dataset.index), 1);
    renderPeopleChips();
  }));
}

async function suggestPeople() {
  const query = els.peopleInput.value.trim();
  if (!query) { els.peopleSuggestions.innerHTML = ''; return; }
  const suggestions = await api(`/api/contacts/suggest?q=${encodeURIComponent(query)}&limit=10`);
  els.peopleSuggestions.innerHTML = suggestions.map((item, index) => `<button class="suggestion" data-index="${index}"><span class="s-title">${escapeHtml(item.display_name)}</span><span class="s-meta">${item.n_handles} handle${item.n_handles === 1 ? '' : 's'}</span></button>`).join('');
  els.peopleSuggestions.querySelectorAll('[data-index]').forEach(button => button.onclick = () => {
    const item = suggestions[Number(button.dataset.index)];
    state.people.push(item);
    renderPeopleChips();
    els.peopleInput.value = '';
    els.peopleSuggestions.innerHTML = '';
  });
}

async function suggestChats() {
  const query = els.chatInput.value.trim();
  const suggestions = await api(`/api/chats/suggest?q=${encodeURIComponent(query)}&limit=20`);
  els.chatSuggestions.innerHTML = suggestions.map((item, index) => `<button class="suggestion" data-index="${index}"><span class="s-title">${escapeHtml(item.title)}</span><span class="s-meta">${item.is_group ? 'group' : '1:1'}</span></button>`).join('');
  els.chatSuggestions.querySelectorAll('[data-index]').forEach(button => button.onclick = () => {
    state.selectedChat = suggestions[Number(button.dataset.index)];
    els.chatInput.value = state.selectedChat.title;
    els.chatSuggestions.innerHTML = '';
  });
}

function clearFilters() {
  state.people = [];
  state.selectedChat = null;
  renderPeopleChips();
  els.peopleInput.value = '';
  els.chatInput.value = '';
  els.dateFrom.value = '';
  els.dateTo.value = '';
  els.hasMedia.checked = false;
  document.querySelector('input[name="group"][value=""]').checked = true;
}

function applyPreset(days) {
  if (!days) {
    els.dateFrom.value = '';
    els.dateTo.value = '';
    return;
  }
  const end = new Date();
  const start = new Date(Date.now() - Number(days) * 86400 * 1000);
  els.dateFrom.value = start.toISOString().slice(0, 10);
  els.dateTo.value = end.toISOString().slice(0, 10);
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
}

document.querySelectorAll('.assist').forEach(button => button.onclick = () => {
  state.assist = button.dataset.assist;
  document.querySelectorAll('.assist').forEach(el => el.classList.toggle('active', el === button));
});
els.searchButton.onclick = search;
els.query.addEventListener('keydown', event => {
  if (event.key === 'Enter') search();
  if (event.key === 'Escape') els.drawer.classList.add('hidden');
});
document.addEventListener('keydown', event => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
    event.preventDefault();
    els.query.focus();
  }
  if (event.key === 'Escape') els.drawer.classList.add('hidden');
});
els.peopleInput.addEventListener('input', debounce(suggestPeople, 150));
els.chatInput.addEventListener('input', debounce(suggestChats, 150));
document.getElementById('clear-filters').onclick = clearFilters;
document.querySelectorAll('.presets button').forEach(button => button.onclick = () => applyPreset(button.dataset.days));
document.querySelectorAll('[data-drawer-close]').forEach(el => el.addEventListener('click', () => els.drawer.classList.add('hidden')));
document.addEventListener('click', event => {
  const target = event.target;
  if (!(target instanceof Element) || !target.closest('.combo')) {
    els.peopleSuggestions.innerHTML = '';
    els.chatSuggestions.innerHTML = '';
  }
});

function debounce(fn, delay) {
  let handle;
  return (...args) => {
    clearTimeout(handle);
    handle = setTimeout(() => fn(...args), delay);
  };
}

// Date inputs should start empty (no implicit range filter) -- clear any
// value WebKit's form-state restoration may have applied on load so a fresh
// search never silently inherits a previous session's date range.
els.dateFrom.value = '';
els.dateTo.value = '';

pollHealth();
