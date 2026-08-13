const token = (() => {
  const hash = window.location.hash.replace(/^#/, '');
  if (hash) {
    sessionStorage.setItem('seaglass-token', hash);
    history.replaceState(null, '', window.location.pathname + window.location.search);
  }
  return sessionStorage.getItem('seaglass-token') || '';
})();

const state = {
  assist: 'off',
  people: [],
  selectedChat: null,
  requestId: null,
  assistToken: null,
  results: [],
};

const els = {
  loading: document.getElementById('loading-screen'),
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
  if (!response.ok) throw new Error(payload.detail || `Request failed: ${response.status}`);
  return payload;
}

async function pollHealth() {
  const health = await api('/api/health');
  els.progressBar.style.width = `${Math.round((health.progress || 0) * 100)}%`;
  els.warmupSteps.innerHTML = (health.steps || []).map(step => `<li>${symbolFor(step.state)} ${step.name} <span>${step.elapsed_s?.toFixed?.(2) || step.elapsed_s || 0}s</span></li>`).join('');
  if (health.error) {
    els.loadingError.textContent = health.error;
    els.loadingError.classList.remove('hidden');
  }
  if (health.state === 'READY' || health.state === 'DEGRADED') {
    els.loading.classList.add('hidden');
    els.app.classList.remove('hidden');
    await refreshStatus();
    return;
  }
  setTimeout(pollHealth, 250);
}

function symbolFor(state) {
  return state === 'done' ? '✓' : state === 'failed' ? '⚠' : state === 'running' ? '⟳' : '•';
}

async function refreshStatus() {
  const status = await api('/api/status');
  const stale = status.n_messages_since_index ? ` · ⚠ ${status.n_messages_since_index} msgs since last index build` : '';
  els.statusBar.textContent = `${status.hydration_available ? '●' : '◌'} Ready · ${status.n_chunks} chunks · ${status.n_chats} chats${stale}`;
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
  const payload = await api('/api/search', {
    method: 'POST',
    body: JSON.stringify({ query: els.query.value, filters: buildFilters(), options: {}, assist: state.assist, request_id: state.requestId })
  });
  if (payload.request_id !== state.requestId) return;
  renderResults(payload);
  if (payload.assist_token) {
    state.assistToken = payload.assist_token;
    pollAssist(payload.assist_token);
  }
}

function renderResults(payload) {
  state.results = payload.sessions || [];
  els.resultsMeta.textContent = `${payload.effective_filters.semantic || els.query.value} · ${payload.n_sessions} sessions · ${payload.n_results} msgs · ${payload.elapsed_s}s`;
  if (!payload.sessions?.length) {
    els.results.innerHTML = `<div class="result-card">No messages match your filters. Try removing a filter.</div>`;
    return;
  }
  els.results.innerHTML = payload.sessions.map((session, index) => renderSession(session, index)).join('');
  document.querySelectorAll('[data-chat-id]').forEach(link => link.addEventListener('click', openConversation));
}

function renderSession(session, index) {
  const hits = (session.messages || []).map(renderMessage).join('');
  const context = (session.context_messages || []).map(renderMessage).join('');
  return `<article class="result-card" data-index="${index}"><h3>${escapeHtml(session.title || `Chat ${session.chat_id}`)} · ${session.day} · score ${Number(session.score || 0).toFixed(2)}</h3><div>${hits}</div><div class="context"><details><summary>± surrounding context</summary>${context}</details></div><button data-chat-id="${session.chat_id}" data-around-ts="${session.messages?.[0]?.ts || ''}">Open full conversation ↗</button></article>`;
}

function renderMessage(message) {
  return `<div class="message"><strong>${escapeHtml(message.sender || 'Me')}</strong> ${formatTime(message.ts)} ${message.has_attachment ? '📎' : ''}<br>${highlight(message.text || '')}</div>`;
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
    els.assistBanner.innerHTML = `Copilot read this as: ${escapeHtml(describeAssist(result))} <button id="apply-assist">Apply</button> <button id="dismiss-assist">Dismiss</button>`;
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
  els.drawerContent.innerHTML = `<button id="close-drawer">Close</button><h2>${escapeHtml(payload.title || `Chat ${payload.chat_id}`)}</h2>${payload.messages.map(renderMessage).join('')}`;
  document.getElementById('close-drawer').onclick = () => els.drawer.classList.add('hidden');
}

function renderPeopleChips() {
  els.peopleChips.innerHTML = state.people.map((person, index) => `<button class="chip" data-index="${index}">${escapeHtml(person.display_name)} ×</button>`).join('');
  els.peopleChips.querySelectorAll('[data-index]').forEach(button => button.addEventListener('click', () => {
    state.people.splice(Number(button.dataset.index), 1);
    renderPeopleChips();
  }));
}

async function suggestPeople() {
  const query = els.peopleInput.value.trim();
  if (!query) { els.peopleSuggestions.innerHTML = ''; return; }
  const suggestions = await api(`/api/contacts/suggest?q=${encodeURIComponent(query)}&limit=10`);
  els.peopleSuggestions.innerHTML = suggestions.map((item, index) => `<button class="suggestion" data-index="${index}">${escapeHtml(item.display_name)} · ${item.n_handles} handle${item.n_handles === 1 ? '' : 's'}</button>`).join('');
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
  els.chatSuggestions.innerHTML = suggestions.map((item, index) => `<button class="suggestion" data-index="${index}">${escapeHtml(item.title)} · ${item.is_group ? 'group' : '1:1'}</button>`).join('');
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
  if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') search();
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

function debounce(fn, delay) {
  let handle;
  return (...args) => {
    clearTimeout(handle);
    handle = setTimeout(() => fn(...args), delay);
  };
}

pollHealth();
