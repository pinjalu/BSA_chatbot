// BSA chat widget — single-file vanilla JS.
//
// Dealer integration (one tag in the page):
//   <script src="https://YOUR-SERVER/widget.js"
//           data-dealer="bsa-mumbai"></script>
//
// Optional data attributes on the script tag:
//   data-dealer        dealer id (saved with each session for analytics)
//   data-vehicle       lock the chat to one model (e.g. "BSA Bantam")
//   data-title         header text (default "BSA Assistant")
//   data-subtitle      subtitle under header (default "Online · replies in seconds")
//   data-color         primary color (default #b71c1c)
//   data-poll-ms       admin-reply poll interval (default 3000)

(function () {
    const script = document.currentScript;
    const cfg = {
        // The widget is served from the same Node host as the API, so
        // derive the API base from the script's own URL.
        apiBase: new URL(script.src).origin,
        dealer: script.dataset.dealer || '',
        vehicle: script.dataset.vehicle || '',
        title: script.dataset.title || 'BSA Assistant',
        subtitle: script.dataset.subtitle || 'Online · replies in seconds',
        color: script.dataset.color || '#b71c1c',
        colorDark: script.dataset.colorDark || '#7f1313',
        pollMs: parseInt(script.dataset.pollMs || '3000', 10),
    };

    const STORAGE_KEY = 'bsa_chat_session';
    let sessionId = localStorage.getItem(STORAGE_KEY) || null;
    let lastSeenMessageId = 0;
    let pollTimer = null;
    let suggestedHandoff = false;
    let handoffOpen = false;
    let isStreaming = false;
    let historyLoaded = false;
    let lastUserQuestion = '';

    // ─────────────────────── styles ───────────────────────
    const css = `
.bsa-fab {
    position: fixed; right: 22px; bottom: 22px; width: 60px; height: 60px;
    border-radius: 50%; border: none; cursor: pointer; z-index: 2147483646;
    background: ${cfg.color}; color: #fff;
    box-shadow: 0 6px 20px rgba(0,0,0,.18), 0 2px 6px rgba(0,0,0,.10);
    display: flex; align-items: center; justify-content: center;
    transition: transform .18s ease, box-shadow .18s ease;
}
.bsa-fab:hover { transform: translateY(-2px); box-shadow: 0 10px 28px rgba(0,0,0,.22); }
.bsa-fab:active { transform: translateY(0); }
.bsa-fab svg { width: 28px; height: 28px; }
.bsa-fab .bsa-fab-close { display: none; }
.bsa-fab.open .bsa-fab-open { display: none; }
.bsa-fab.open .bsa-fab-close { display: block; }

.bsa-panel {
    position: fixed; right: 22px; bottom: 96px; width: 380px; height: 580px;
    max-height: calc(100vh - 120px); background: #fff;
    border-radius: 16px; box-shadow: 0 24px 60px rgba(0,0,0,.18), 0 4px 12px rgba(0,0,0,.08);
    display: flex; flex-direction: column; overflow: hidden;
    z-index: 2147483647; opacity: 0; transform: translateY(20px) scale(.96);
    pointer-events: none; transition: opacity .18s ease, transform .18s ease;
    font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
        "Helvetica Neue", Arial, sans-serif; color: #1a1a1a;
}
.bsa-panel.open { opacity: 1; transform: translateY(0) scale(1); pointer-events: auto; }

@media (max-width: 480px) {
    .bsa-panel { right: 0; bottom: 0; width: 100vw; height: 100vh;
        max-height: 100vh; border-radius: 0; }
    .bsa-fab { right: 16px; bottom: 16px; }
}

/* Header */
.bsa-header {
    background: linear-gradient(135deg, ${cfg.color} 0%, ${cfg.colorDark} 100%);
    color: #fff; padding: 16px 18px; display: flex; align-items: center;
    justify-content: space-between; gap: 12px;
}
.bsa-header-info { display: flex; align-items: center; gap: 12px; min-width: 0; }
.bsa-avatar {
    width: 38px; height: 38px; border-radius: 50%;
    background: rgba(255,255,255,.2); display: flex; align-items: center;
    justify-content: center; flex-shrink: 0; font-weight: 700; font-size: 14px;
}
.bsa-titles { min-width: 0; }
.bsa-titles h3 { margin: 0; font-size: 15px; font-weight: 600;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bsa-titles .sub { font-size: 12px; opacity: .85; margin-top: 2px;
    display: flex; align-items: center; gap: 5px; }
.bsa-dot { width: 7px; height: 7px; background: #4ade80; border-radius: 50%;
    box-shadow: 0 0 0 3px rgba(74,222,128,.25); }
.bsa-header button {
    background: transparent; color: #fff; border: 0; font-size: 22px;
    cursor: pointer; line-height: 1; padding: 4px 6px; opacity: .85;
    border-radius: 6px; transition: opacity .15s, background .15s;
}
.bsa-header button:hover { opacity: 1; background: rgba(255,255,255,.12); }

/* Messages area */
.bsa-msgs {
    flex: 1; overflow-y: auto; padding: 16px 14px; background: #f5f6f8;
    scroll-behavior: smooth;
}
.bsa-msgs::-webkit-scrollbar { width: 8px; }
.bsa-msgs::-webkit-scrollbar-thumb { background: #cfd2d8; border-radius: 4px; }
.bsa-msgs::-webkit-scrollbar-track { background: transparent; }

.bsa-msg { margin-bottom: 12px; display: flex; flex-direction: column;
    animation: bsaIn .22s ease-out; }
.bsa-msg.user { align-items: flex-end; }
.bsa-msg.assistant, .bsa-msg.admin { align-items: flex-start; }

@keyframes bsaIn {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}

.bsa-bubble {
    max-width: 82%; padding: 10px 14px; border-radius: 16px;
    white-space: pre-wrap; word-wrap: break-word; font-size: 14px;
    box-shadow: 0 1px 2px rgba(0,0,0,.04);
}
.bsa-msg.user .bsa-bubble {
    background: ${cfg.color}; color: #fff; border-bottom-right-radius: 4px;
}
.bsa-msg.assistant .bsa-bubble {
    background: #fff; border: 1px solid #e8e9ec; color: #1a1a1a;
    border-bottom-left-radius: 4px;
}
.bsa-msg.admin .bsa-bubble {
    background: #fff8e7; border: 1px solid #f3d77a; color: #1a1a1a;
    border-bottom-left-radius: 4px;
}
.bsa-bubble p { margin: 0 0 10px; }
.bsa-bubble p:last-child { margin-bottom: 0; }
.bsa-bubble strong { font-weight: 600; color: #0c0c0d; }
.bsa-bubble em { font-style: italic; }
.bsa-bubble code {
    background: #f0f1f4; border-radius: 4px; padding: 1px 5px;
    font: 12.5px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    color: #1a1a1a;
}
.bsa-bubble ul, .bsa-bubble ol {
    margin: 4px 0 10px; padding-left: 22px;
}
.bsa-bubble li { margin-bottom: 6px; line-height: 1.45; }
.bsa-bubble li:last-child { margin-bottom: 0; }
.bsa-bubble ul li { list-style: disc; }
.bsa-bubble ol li { list-style: decimal; }
.bsa-bubble ol li::marker { font-weight: 600; color: #5a5d63; }
.bsa-bubble .bsa-sources-line {
    margin-top: 10px; padding-top: 8px; border-top: 1px dashed #e0e2e6;
    font-size: 12px; color: #6b6e73; font-style: italic;
}
.bsa-msg.user .bsa-bubble strong { color: #fff; }

.bsa-figure {
    display: block; margin: 8px 0; border-radius: 8px; overflow: hidden;
    background: #f5f6f8; border: 1px solid #e8e9ec; line-height: 0;
    transition: transform .15s, box-shadow .15s;
}
.bsa-figure:hover { transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,.10); }
.bsa-figure img { width: 100%; height: auto; max-height: 280px;
    object-fit: contain; display: block; background: #fff; }
.bsa-msg.system { align-items: center; }
.bsa-msg.system .bsa-bubble {
    background: transparent; color: #8a8d93; font-size: 12px;
    font-style: italic; padding: 4px 10px; box-shadow: none;
    text-align: center; max-width: 92%;
}

.bsa-tag {
    display: inline-block; font-size: 10px; font-weight: 600;
    color: #c46300; background: #fff3d6; padding: 2px 8px;
    border-radius: 10px; margin-bottom: 4px; letter-spacing: .3px;
    text-transform: uppercase;
}
.bsa-tag-learned {
    color: #2e6b3a; background: #e3f5e8;
    text-transform: none; letter-spacing: 0;
    font-size: 11px; padding: 3px 10px;
}

/* Citations */
.bsa-cites { margin-top: 6px; max-width: 82%; }
.bsa-cites-toggle {
    background: none; border: 0; padding: 4px 0; color: #5a5d63;
    font-size: 11px; cursor: pointer; display: flex; align-items: center;
    gap: 4px; font-family: inherit;
}
.bsa-cites-toggle:hover { color: ${cfg.color}; }
.bsa-cites-toggle .arrow { transition: transform .18s; display: inline-block; }
.bsa-cites.open .arrow { transform: rotate(90deg); }
.bsa-cites-list { display: none; margin-top: 4px; flex-wrap: wrap; gap: 4px; }
.bsa-cites.open .bsa-cites-list { display: flex; }
.bsa-cite-pill {
    background: #f0f1f4; border: 1px solid #e0e1e5; border-radius: 10px;
    padding: 3px 8px; font-size: 11px; color: #4a4d53;
}
.bsa-timings {
    font-size: 10.5px; color: #a0a3a8; margin-top: 4px;
    padding-top: 2px; cursor: help; max-width: 82%;
}

/* Typing indicator */
.bsa-typing { display: flex; align-items: center; gap: 4px; padding: 8px 4px; }
.bsa-typing span {
    width: 7px; height: 7px; background: #b8bbc0; border-radius: 50%;
    animation: bsaPulse 1.2s ease-in-out infinite;
}
.bsa-typing span:nth-child(2) { animation-delay: .15s; }
.bsa-typing span:nth-child(3) { animation-delay: .3s; }
@keyframes bsaPulse {
    0%, 60%, 100% { opacity: .35; transform: translateY(0); }
    30%           { opacity: 1; transform: translateY(-3px); }
}

/* Status text shown while the pre-LLM pipeline runs ("Searching the
   manuals...", "Reading 12 relevant pages..."). Sits next to the
   typing dots so the user sees both motion AND a label of what the
   bot is actually doing. The whole block is replaced once the LLM
   starts streaming tokens. */
.bsa-status { display: flex; align-items: center; gap: 8px; padding: 4px 0; }
.bsa-status-text {
    color: #7a7d83;
    font-size: 13px;
    font-style: italic;
    line-height: 1.3;
}

/* Blinking caret on the actively-streaming assistant bubble — same
   visual cue ChatGPT uses so the user knows new tokens are still
   coming in. */
.bsa-bubble.bsa-streaming::after {
    content: '▊';
    display: inline-block;
    color: ${cfg.color};
    margin-left: 2px;
    vertical-align: baseline;
    animation: bsaCaret 1s steps(1) infinite;
}
@keyframes bsaCaret {
    0%, 49%   { opacity: 1; }
    50%, 100% { opacity: 0; }
}

/* Suggested actions / handoff cards */
.bsa-card {
    align-self: stretch; background: #fff7e6; border: 1px solid #f5d089;
    border-radius: 12px; padding: 12px 14px; margin: 8px 0; font-size: 13px;
    color: #5a3a00; animation: bsaIn .22s ease-out;
}
.bsa-card .bsa-card-title { font-weight: 600; margin-bottom: 4px; color: #4a2e00; }
.bsa-card button {
    background: ${cfg.color}; color: #fff; border: 0; padding: 10px 16px;
    border-radius: 8px; font-size: 14px; font-weight: 600; cursor: pointer;
    margin-top: 10px; font-family: inherit; transition: background .15s, transform .1s;
    width: 100%;
}
.bsa-card button:hover { background: ${cfg.colorDark}; }
.bsa-card button:active { transform: scale(.98); }
.bsa-card-prominent {
    background: linear-gradient(135deg, #fff5e0 0%, #ffe8c2 100%);
    border-color: #f0b95d;
    box-shadow: 0 2px 12px rgba(240,185,93,.25);
}

.bsa-form-row { display: flex; flex-direction: column; gap: 6px; margin-top: 8px; }
.bsa-form-row input {
    width: 100%; padding: 9px 10px; border: 1px solid #d8dade; border-radius: 8px;
    font: inherit; box-sizing: border-box; background: #fff; color: #1a1a1a;
}
.bsa-form-row input:focus { outline: none; border-color: ${cfg.color}; }

/* Input area */
.bsa-input-wrap {
    border-top: 1px solid #e8e9ec; padding: 10px 12px 12px; background: #fff;
}
.bsa-input-row {
    display: flex; align-items: flex-end; gap: 8px; background: #f5f6f8;
    border: 1px solid #e0e2e6; border-radius: 22px; padding: 6px 6px 6px 14px;
    transition: border-color .15s, box-shadow .15s;
}
.bsa-input-row:focus-within {
    border-color: ${cfg.color};
    box-shadow: 0 0 0 3px ${cfg.color}15;
}
.bsa-input-row textarea {
    flex: 1; resize: none; border: 0; background: transparent; outline: none;
    font: inherit; padding: 8px 0; max-height: 120px; min-height: 22px;
    color: #1a1a1a; font-size: 14px;
}
.bsa-input-row textarea::placeholder { color: #9a9da3; }
.bsa-send {
    background: ${cfg.color}; color: #fff; border: 0; border-radius: 50%;
    width: 36px; height: 36px; cursor: pointer; display: flex;
    align-items: center; justify-content: center; flex-shrink: 0;
    transition: background .15s, transform .15s;
}
.bsa-send:hover { background: ${cfg.colorDark}; }
.bsa-send:active { transform: scale(.94); }
.bsa-send:disabled { background: #cfd2d8; cursor: not-allowed; }
.bsa-send svg { width: 16px; height: 16px; }

/* Attach (image upload) */
.bsa-attach {
    background: transparent; border: 0; cursor: pointer; padding: 6px;
    display: flex; align-items: center; justify-content: center;
    color: #6b6e74; flex-shrink: 0; border-radius: 50%;
    transition: background .15s, color .15s;
}
.bsa-attach:hover { background: #e9eaee; color: ${cfg.color}; }
.bsa-attach svg { width: 20px; height: 20px; }
.bsa-attach input[type="file"] { display: none; }

/* Preview thumbnail of an attached image, shown above the input */
.bsa-preview {
    display: none; align-items: center; gap: 8px; margin-bottom: 6px;
    padding: 6px 8px; background: #f5f6f8; border: 1px solid #e0e2e6;
    border-radius: 10px;
}
.bsa-preview.show { display: flex; }
.bsa-preview img {
    width: 44px; height: 44px; object-fit: cover; border-radius: 6px;
    border: 1px solid #d8dade;
}
.bsa-preview .name {
    font-size: 12px; color: #4a4d53; flex: 1; overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap;
}
.bsa-preview .clear {
    background: transparent; border: 0; cursor: pointer; color: #6b6e74;
    font-size: 18px; line-height: 1; padding: 0 4px;
}
.bsa-preview .clear:hover { color: ${cfg.color}; }

/* User-image bubble (their attached image, shown in chat history) */
.bsa-msg.user .bsa-bubble img.bsa-userimg {
    display: block; max-width: 220px; max-height: 180px;
    border-radius: 8px; margin-bottom: 6px;
}

`;
    const styleEl = document.createElement('style');
    styleEl.textContent = css;
    document.head.appendChild(styleEl);

    // ─────────────────────── DOM ───────────────────────
    const fab = el('button', { class: 'bsa-fab', 'aria-label': 'Open chat' });
    fab.innerHTML = `
        <svg class="bsa-fab-open" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 2C6.48 2 2 6.04 2 11c0 2.5 1.13 4.76 2.96 6.36L4 22l4.84-1.27C10.06 21.55 11.01 21.7 12 21.7c5.52 0 10-4.04 10-10S17.52 2 12 2z"/>
        </svg>
        <svg class="bsa-fab-close" viewBox="0 0 24 24" fill="currentColor">
            <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
        </svg>
    `;

    const panel = el('div', { class: 'bsa-panel', role: 'dialog',
        'aria-label': cfg.title });
    const initials = cfg.title.split(/\s+/).slice(0, 2)
        .map(w => w[0] || '').join('').toUpperCase() || 'B';
    panel.innerHTML = `
        <div class="bsa-header">
            <div class="bsa-header-info">
                <div class="bsa-avatar">${escapeHtml(initials)}</div>
                <div class="bsa-titles">
                    <h3>${escapeHtml(cfg.title)}</h3>
                    <div class="sub"><span class="bsa-dot"></span>${escapeHtml(cfg.subtitle)}</div>
                </div>
            </div>
            <button class="bsa-close" aria-label="Close">×</button>
        </div>
        <div class="bsa-msgs" id="bsa-msgs"></div>
        <div class="bsa-input-wrap">
            <div class="bsa-preview" id="bsa-preview">
                <img id="bsa-preview-img" alt="">
                <div class="name" id="bsa-preview-name"></div>
                <button class="clear" id="bsa-preview-clear"
                    aria-label="Remove attached image">×</button>
            </div>
            <div class="bsa-input-row">
                <label class="bsa-attach" aria-label="Attach image" title="Attach image">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
                        stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
                    </svg>
                    <input type="file" id="bsa-file" accept="image/*">
                </label>
                <textarea id="bsa-input" rows="1"
                    placeholder="Type your question..."></textarea>
                <button class="bsa-send" id="bsa-send" aria-label="Send">
                    <svg viewBox="0 0 24 24" fill="currentColor">
                        <path d="M2 21l21-9L2 3v7l15 2-15 2v7z"/>
                    </svg>
                </button>
            </div>
        </div>
    `;

    document.body.appendChild(fab);
    document.body.appendChild(panel);

    const msgsEl = panel.querySelector('#bsa-msgs');
    const inputEl = panel.querySelector('#bsa-input');
    const sendBtn = panel.querySelector('#bsa-send');
    const fileEl = panel.querySelector('#bsa-file');
    const previewEl = panel.querySelector('#bsa-preview');
    const previewImg = panel.querySelector('#bsa-preview-img');
    const previewName = panel.querySelector('#bsa-preview-name');
    const previewClear = panel.querySelector('#bsa-preview-clear');

    // Pending image for the next send. Held in-memory only — never
    // persisted, never re-sent on subsequent turns.
    let pendingImage = null;   // { name, dataUrl, base64 }

    fab.addEventListener('click', () => togglePanel());
    panel.querySelector('.bsa-close').addEventListener('click', () => togglePanel(false));

    inputEl.addEventListener('input', autoresize);
    inputEl.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            send();
        }
    });
    sendBtn.addEventListener('click', send);

    fileEl.addEventListener('change', () => {
        const f = fileEl.files && fileEl.files[0];
        if (!f) return;
        // Cap at ~6 MB so we don't blow up the request payload. OpenAI's
        // vision input is also size-limited.
        if (f.size > 6 * 1024 * 1024) {
            alert('Image is too large (max 6 MB). Please pick a smaller one.');
            fileEl.value = '';
            return;
        }
        const reader = new FileReader();
        reader.onload = () => {
            const dataUrl = reader.result;
            const comma = dataUrl.indexOf(',');
            const base64 = comma >= 0 ? dataUrl.slice(comma + 1) : '';
            pendingImage = { name: f.name, dataUrl, base64 };
            previewImg.src = dataUrl;
            previewName.textContent = f.name;
            previewEl.classList.add('show');
        };
        reader.readAsDataURL(f);
    });

    previewClear.addEventListener('click', clearPendingImage);

    function clearPendingImage() {
        pendingImage = null;
        fileEl.value = '';
        previewImg.removeAttribute('src');
        previewName.textContent = '';
        previewEl.classList.remove('show');
    }

    function togglePanel(force) {
        const willOpen = force === undefined ? !panel.classList.contains('open') : force;
        panel.classList.toggle('open', willOpen);
        fab.classList.toggle('open', willOpen);
        fab.setAttribute('aria-label', willOpen ? 'Close chat' : 'Open chat');
        if (willOpen) {
            inputEl.focus();
            if (!historyLoaded) {
                historyLoaded = true;
                greet();
            }
        }
    }

    function autoresize() {
        inputEl.style.height = 'auto';
        inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
    }

    // ─────────────────────── conversation ───────────────────────
    function greet() {
        appendBubble('assistant',
            `Hi 👋  I'm the BSA assistant — happy to help. What would you like to know?`);
    }

    function resetSession() {
        sessionId = null;
        lastSeenMessageId = 0;
        handoffOpen = false;
        suggestedHandoff = false;
        historyLoaded = false;
        localStorage.removeItem(STORAGE_KEY);
        if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
        // Clear chat UI and show fresh greeting
        msgsEl.innerHTML = '';
        historyLoaded = true;
        greet();
    }

    function startPolling() {
        if (pollTimer) return;
        pollTimer = setInterval(pollOnce, cfg.pollMs);
    }

    async function pollOnce() {
        if (!sessionId) return;
        try {
            const r = await fetch(
                `${cfg.apiBase}/api/messages/${sessionId}?after=${lastSeenMessageId}`,
            );
            if (!r.ok) return;
            const data = await r.json();
            for (const m of data.messages) {
                // The streaming pipeline already rendered user/assistant; only
                // surface admin replies and system notes through polling.
                if (m.role === 'admin' || m.role === 'system') {
                    appendBubble(m.role, m.content);
                }
                lastSeenMessageId = Math.max(lastSeenMessageId, m.id);
            }
            handoffOpen = data.handoffOpen;
        } catch (_) { /* network blip — retry next tick */ }
    }

    async function send() {
        const text = inputEl.value.trim();
        if (!text || isStreaming) return;
        lastUserQuestion = text;
        // Snapshot the pending image so the user can attach a new one
        // for the next turn while this one is still streaming.
        const sentImage = pendingImage;
        inputEl.value = '';
        autoresize();
        const userWrap = appendBubble('user', text);
        if (sentImage) {
            const userBubble = userWrap.querySelector('.bsa-bubble');
            const img = document.createElement('img');
            img.className = 'bsa-userimg';
            img.src = sentImage.dataUrl;
            img.alt = 'attached image';
            userBubble.insertBefore(img, userBubble.firstChild);
        }
        clearPendingImage();

        isStreaming = true;
        sendBtn.disabled = true;

        // Show typing dots until first token arrives.
        const assistantWrap = appendBubble('assistant', '');
        const bubble = assistantWrap.querySelector('.bsa-bubble');
        bubble.innerHTML = `<div class="bsa-typing">
            <span></span><span></span><span></span></div>`;
        let firstToken = true;

        try {
            const resp = await fetch(`${cfg.apiBase}/api/chat`, {
                method: 'POST',
                headers: { 'content-type': 'application/json' },
                body: JSON.stringify({
                    sessionId,
                    dealerId: cfg.dealer,
                    message: text,
                    vehicle: cfg.vehicle || null,
                    imageB64: sentImage ? sentImage.base64 : null,
                }),
            });
            if (!resp.ok || !resp.body) {
                bubble.textContent = 'Sorry — connection failed.';
                isStreaming = false; sendBtn.disabled = false; return;
            }

            const reader = resp.body.getReader();
            const decoder = new TextDecoder();
            let buf = '';
            let acc = '';            // full text from the model
            let displayed = '';      // text currently shown on screen
            let typeTimer = 0;       // setTimeout handle for the typewriter
            let streamFinished = false;
            let citations = null;
            let images = {};
            let learnedMeta = null;
            const INSTANT_STREAM_RENDER = false;

            // Typewriter rendering — same effect ChatGPT uses. The model
            // often emits tokens in a tight burst (especially gpt-5-mini
            // with reasoning_effort=minimal), which would make the answer
            // pop in all at once and hide the streaming progress. We
            // instead drain `acc` to the visible bubble at a steady pace
            // so the user sees text appear smoothly.
            const TYPE_CHARS_PER_TICK = 4;   // slower, clearer progressive rendering
            const TYPE_TICK_MS = 16;         // ~60 fps with visible typing effect
            const startTypewriter = () => {
                if (typeTimer) return;
                const tick = () => {
                    typeTimer = 0;
                    if (displayed.length < acc.length) {
                        displayed = acc.slice(
                            0, displayed.length + TYPE_CHARS_PER_TICK,
                        );
                        bubble.innerHTML = renderMarkdown(displayed, images);
                        msgsEl.scrollTop = msgsEl.scrollHeight;
                        typeTimer = setTimeout(tick, TYPE_TICK_MS);
                    } else if (!streamFinished) {
                        // Caught up to the model — wait briefly and
                        // re-check, the next tokens may have arrived.
                        typeTimer = setTimeout(tick, TYPE_TICK_MS);
                    }
                };
                typeTimer = setTimeout(tick, 0);
            };
            const stopTypewriter = () => {
                if (typeTimer) { clearTimeout(typeTimer); typeTimer = 0; }
            };

            // Wait for the visible text to catch up to the full answer,
            // then attach citations / timings / handoff prompts. Keeps
            // the bubble from suddenly snapping to the complete reply.
            const finishStream = (opts) => {
                const finalize = () => {
                    stopTypewriter();
                    bubble.classList.remove('bsa-streaming');
                    if (acc) bubble.innerHTML = renderMarkdown(acc, images);
                    if (opts.citations && opts.citations.length) {
                        renderCitations(assistantWrap, opts.citations);
                    }
                    if (opts.timings) {
                        renderTimings(assistantWrap, opts.timings);
                    }
                    if (opts.suggestHandoff && !suggestedHandoff
                            && !opts.handoffOpen) {
                        suggestedHandoff = true;
                        showHandoffPrompt();
                    }
                };
                if (INSTANT_STREAM_RENDER) {
                    finalize();
                    return;
                }
                const wait = () => {
                    if (displayed.length >= acc.length) {
                        finalize();
                    } else {
                        setTimeout(wait, TYPE_TICK_MS);
                    }
                };
                wait();
            };

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                buf += decoder.decode(value, { stream: true });
                let m;
                while ((m = /\r?\n\r?\n/.exec(buf)) !== null) {
                    const idx = m.index;
                    const sepLen = m[0].length;
                    const raw = buf.slice(0, idx);
                    buf = buf.slice(idx + sepLen);
                    const ev = parseSse(raw);
                    if (!ev) continue;

                    if (ev.event === 'session') {
                        sessionId = ev.data.sessionId;
                        localStorage.setItem(STORAGE_KEY, sessionId);
                        startPolling();
                    } else if (ev.event === 'status') {
                        // Server pushes a short progress label between
                        // pipeline steps ("Searching the manuals...",
                        // "Reading 12 relevant pages...", etc). Show it
                        // in place of the typing dots so the user sees
                        // real progress instead of staring at silence.
                        // Once tokens start arriving, the first-token
                        // handler clears this and the answer takes over.
                        if (firstToken) {
                            const txt = (ev.data && ev.data.text) || '';
                            bubble.innerHTML =
                                '<div class="bsa-status">'
                                + '<div class="bsa-typing">'
                                + '<span></span><span></span><span></span>'
                                + '</div>'
                                + '<div class="bsa-status-text"></div>'
                                + '</div>';
                            const txtEl = bubble.querySelector('.bsa-status-text');
                            if (txtEl) txtEl.textContent = txt;
                        }
                    } else if (ev.event === 'token') {
                        const piece = ev.data.text || '';
                        acc += piece;
                        // Switch to the answer bubble immediately on first
                        // token so users see streaming start right away.
                        if (firstToken) {
                            bubble.innerHTML = '';
                            bubble.classList.add('bsa-streaming');
                            firstToken = false;
                            displayed = acc;
                            // Paint immediately on first token (even if it's
                            // mostly whitespace) to avoid "token in logs but
                            // nothing on UI yet" perception.
                            bubble.innerHTML = renderMarkdown(displayed || '…', images);
                            msgsEl.scrollTop = msgsEl.scrollHeight;
                            if (!INSTANT_STREAM_RENDER) startTypewriter();
                            continue;
                        }
                        if (INSTANT_STREAM_RENDER) {
                            displayed = acc;
                            bubble.innerHTML = renderMarkdown(displayed, images);
                            msgsEl.scrollTop = msgsEl.scrollHeight;
                        }
                    } else if (ev.event === 'meta') {
                        citations = ev.data.citations || [];
                        images = ev.data.images || {};
                        if (ev.data.learned) {
                            learnedMeta = ev.data.learned;
                            // Tag the bubble immediately so the user
                            // sees "Saved reply from our team" before
                            // the text starts streaming.
                            ensureLearnedTag(assistantWrap);
                        }
                    } else if (ev.event === 'error') {
                        streamFinished = true;
                        stopTypewriter();
                        bubble.classList.remove('bsa-streaming');
                        bubble.textContent = `Error: ${ev.data.message}`;
                    } else if (ev.event === 'done') {
                        streamFinished = true;
                        // Let the typewriter finish — schedule the
                        // final render after the visible text has
                        // caught up to acc, so the bubble doesn't
                        // suddenly jump to the full answer.
                        finishStream({
                            citations,
                            timings: ev.data.timings,
                            suggestHandoff: ev.data.suggestHandoff,
                            handoffOpen: ev.data.handoffOpen,
                        });
                    }
                }
            }
        } catch (e) {
            bubble.textContent = `Error: ${e.message}`;
        } finally {
            isStreaming = false;
            sendBtn.disabled = false;
            inputEl.focus();
        }
    }

    // ─────────────────────── handoff ───────────────────────
    function showHandoffPrompt() {
        const card = el('div', { class: 'bsa-card bsa-card-prominent' });
        const qPreview = lastUserQuestion
            ? `<div style="font-size:12px;color:#7a5000;margin:6px 0 10px;
                padding:8px 10px;background:rgba(0,0,0,.05);border-radius:6px;
                font-style:italic;">"${escapeHtml(lastUserQuestion.slice(0, 100))}${lastUserQuestion.length > 100 ? '…' : ''}"</div>`
            : '';
        card.innerHTML = `
            <div class="bsa-card-title">Our team can help with this</div>
            ${qPreview}
            <div style="font-size:13px;">Our dealership team will reply right here in this chat.</div>
            <button>💬  Talk to a human</button>
        `;
        card.querySelector('button').addEventListener('click', () => {
            card.remove();
            showHandoffForm('low_confidence');
        });
        msgsEl.appendChild(card);
        msgsEl.scrollTop = msgsEl.scrollHeight;
    }

    function showHandoffForm(reason) {
        if (handoffOpen) {
            appendBubble('system',
                'Your request has already been sent. We will reply here shortly.');
            return;
        }
        const card = el('div', { class: 'bsa-card' });
        card.innerHTML = `
            <div class="bsa-card-title">Connect with our team</div>
            <div>Leave your details — we'll reply right here.</div>
            <div class="bsa-form-row">
                <input type="text" placeholder="Your name" data-name>
                <input type="email" placeholder="Email" data-email required>
                <button>Send request</button>
            </div>
        `;
        const nameEl = card.querySelector('[data-name]');
        const emailEl = card.querySelector('[data-email]');
        const btn = card.querySelector('button');
        const submit = async () => {
            const name = nameEl.value.trim();
            const email = emailEl.value.trim();
            if (!email) { emailEl.focus(); return; }
            btn.disabled = true; btn.textContent = 'Sending…';
            try {
                const r = await fetch(`${cfg.apiBase}/api/handoff`, {
                    method: 'POST',
                    headers: { 'content-type': 'application/json' },
                    body: JSON.stringify({ sessionId, name, email, reason }),
                });
                if (!r.ok) throw new Error('failed');
                card.remove();
                handoffOpen = true;
                appendBubble('system',
                    `Thanks${name ? ', ' + name : ''} — a human will reply ` +
                    `here shortly. You can keep this window open or come back later.`);
                startPolling();
            } catch (_) {
                btn.disabled = false; btn.textContent = 'Send request';
                appendBubble('system', 'Could not send request. Please try again.');
            }
        };
        btn.addEventListener('click', submit);
        emailEl.addEventListener('keydown', e => {
            if (e.key === 'Enter') submit();
        });
        msgsEl.appendChild(card);
        nameEl.focus();
        msgsEl.scrollTop = msgsEl.scrollHeight;
    }

    // ─────────────────────── render helpers ───────────────────────
    function ensureLearnedTag(wrap) {
        if (!wrap || wrap.querySelector('.bsa-tag-learned')) return;
        const tag = el('div', { class: 'bsa-tag bsa-tag-learned' });
        tag.textContent = '⚡ Saved reply from our team';
        wrap.insertBefore(tag, wrap.firstChild);
    }

    function appendBubble(role, content) {
        const wrap = el('div', { class: `bsa-msg ${role}` });
        if (role === 'admin') {
            const tag = el('div', { class: 'bsa-tag' });
            tag.textContent = 'Support team';
            wrap.appendChild(tag);
        }
        const bubble = el('div', { class: 'bsa-bubble' });
        // User and system messages stay as plain text (no formatting needed,
        // and prevents reflecting any user-typed markdown). Assistant and
        // admin replies render light markdown for emphasis + lists.
        if (role === 'assistant' || role === 'admin') {
            bubble.innerHTML = renderMarkdown(content);
        } else {
            bubble.textContent = content;
        }
        wrap.appendChild(bubble);
        msgsEl.appendChild(wrap);
        msgsEl.scrollTop = msgsEl.scrollHeight;
        return wrap;
    }

    // Tiny markdown renderer: escapes HTML first, then applies a small,
    // safe subset (bold, italic, inline code, bullet/numbered lists,
    // paragraph breaks). [[SHOW_IMAGE: IMG_X]] tags are pulled out
    // before markdown processing and reinjected as <img> figures
    // afterwards, using the per-turn images map.
    function renderMarkdown(raw, imageMap) {
        if (!raw) return '';

        // 1. Replace [[SHOW_IMAGE: IMG_X]] with placeholders so the HTML
        //    escaper and markdown rules don't mangle them.
        const imgs = imageMap || {};
        const placeholders = [];
        // Hard dedup safeguard: if the LLM emits the same image
        // twice in one answer (or two IDs resolving to the same
        // URL), only the FIRST occurrence renders. Stops the
        // page filling with copies of the same diagram if the
        // model gets enthusiastic.
        const seenUrls = new Set();
        let withPlaceholders = String(raw).replace(
            /\[\[SHOW_IMAGE:\s*([A-Za-z0-9_-]+)\s*\]\]/g,
            (_match, id) => {
                const url = imgs[id];
                if (url && seenUrls.has(url)) return ''; // dup - drop
                if (url) seenUrls.add(url);
                if (!url) return ''; // unknown id → drop the tag
                const idx = placeholders.length;
                placeholders.push(url);
                return ` IMG${idx} `;
            },
        );

        let s = escapeHtml(withPlaceholders);

        // Inline code first (so its content isn't touched by other rules).
        s = s.replace(/`([^`\n]+)`/g, '<code>$1</code>');
        // Bold + italic. Order matters: ** before *.
        s = s.replace(/\*\*([^*\n][^*]*?)\*\*/g, '<strong>$1</strong>');
        s = s.replace(/(^|[^\*])\*([^*\n]+?)\*(?!\*)/g, '$1<em>$2</em>');

        // Build paragraphs and lists line-by-line.
        const lines = s.split(/\r?\n/);
        const out = [];
        let listType = null;   // 'ul' | 'ol' | null
        let para = [];

        const flushPara = () => {
            if (para.length) { out.push('<p>' + para.join('<br>') + '</p>'); para = []; }
        };
        const closeList = () => {
            if (listType) { out.push(`</${listType}>`); listType = null; }
        };

        for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) { flushPara(); closeList(); continue; }

            const ulMatch = /^[-*•]\s+(.*)$/.exec(trimmed);
            const olMatch = /^(\d+)[.)]\s+(.*)$/.exec(trimmed);
            if (ulMatch) {
                flushPara();
                if (listType !== 'ul') { closeList(); out.push('<ul>'); listType = 'ul'; }
                out.push('<li>' + ulMatch[1] + '</li>');
            } else if (olMatch) {
                flushPara();
                if (listType !== 'ol') { closeList(); out.push('<ol>'); listType = 'ol'; }
                out.push('<li>' + olMatch[2] + '</li>');
            } else {
                closeList();
                para.push(trimmed);
            }
        }
        flushPara();
        closeList();

        let html = out.join('').replace(
            /<p>(Sources:[^<]*)<\/p>/i,
            '<p class="bsa-sources-line">$1</p>',
        );

        // 2. Reinject images as block-level figures so each image
        //    occupies its own row. Order matters:
        //    a) lone-image paragraph → unwrap and replace with figure;
        //    b) any remaining inline IMG marker → close the surrounding
        //       paragraph, drop the figure, reopen the paragraph; tidy
        //       empty <p></p> after.
        //    Without (b), the LLM packing several markers on one line
        //    (e.g. "[[SHOW_IMAGE: IMG_1]] [[SHOW_IMAGE: IMG_2]]") leaves
        //    inline-only images that look like tiny thumbnails next to
        //    prose.
        html = html.replace(/<p>\s*IMG(\d+)\s*<\/p>/g, (_m, n) =>
            imageBlock(placeholders[Number(n)]),
        );
        html = html.replace(/\s?IMG(\d+)\s?/g, (_m, n) =>
            '</p>' + imageBlock(placeholders[Number(n)]) + '<p>',
        );
        html = html.replace(/<p>\s*<\/p>/g, '');
        return html;
    }

    function imageBlock(url) {
        if (!url) return '';
        const safe = String(url).replace(/"/g, '&quot;');
        return `<a class="bsa-figure" href="${safe}" target="_blank" `
            + `rel="noopener" title="Open full size">`
            + `<img src="${safe}" alt="figure" loading="lazy">`
            + `</a>`;
    }

    function renderTimings(bubbleWrap, t) {
        // A tiny "answered in 4.2s" line, with the breakdown on hover.
        // Useful for tuning latency without cluttering the UI.
        if (!t || !t.total) return;
        const total = (t.total / 1000).toFixed(1);
        const lines = [
            `total ${t.total} ms`,
            `  embed     ${t.embed} ms`,
            `  pinecone  ${t.pinecone} ms`,
        ];
        if (typeof t.rerank === 'number' && t.rerank > 0) {
            lines.push(`  rerank    ${t.rerank} ms`);
        }
        lines.push(
            `  TTFT      ${t.llm_ttft} ms`,
            `  LLM       ${t.llm_total} ms`,
        );
        const el = document.createElement('div');
        el.className = 'bsa-timings';
        el.textContent = `answered in ${total}s`;
        el.title = lines.join('\n');
        bubbleWrap.appendChild(el);
    }

    function renderCitations(bubbleWrap, citations) {
        const seen = new Set();
        const pills = [];
        for (const c of citations) {
            if (!c || c.score < 0.30) continue;
            const key = `${c.source_pdf}|${c.page}`;
            if (seen.has(key)) continue;
            seen.add(key);
            pills.push({
                label: shortDoc(c.source_pdf) + ` · p.${formatPage(c.page)}`,
                tooltip: `${c.section || '—'} (${c.source_pdf}, p. ${formatPage(c.page)})`,
            });
        }
        if (!pills.length) return;
        const cites = el('div', { class: 'bsa-cites' });
        cites.innerHTML = `
            <button class="bsa-cites-toggle" type="button">
                <span class="arrow">▸</span>
                <span>${pills.length} source${pills.length > 1 ? 's' : ''}</span>
            </button>
            <div class="bsa-cites-list"></div>
        `;
        const list = cites.querySelector('.bsa-cites-list');
        for (const p of pills) {
            const pill = el('span', { class: 'bsa-cite-pill', title: p.tooltip });
            pill.textContent = p.label;
            list.appendChild(pill);
        }
        cites.querySelector('.bsa-cites-toggle').addEventListener('click', () => {
            cites.classList.toggle('open');
        });
        bubbleWrap.appendChild(cites);
    }

    function shortDoc(pdf) {
        if (!pdf) return '—';
        // Strip extension and obvious noise so the pill label stays compact.
        return pdf.replace(/\.pdf$/i, '')
            .replace(/_REV-\d+/i, '')
            .replace(/_\d{2}\.\d{2}\.\d{4}.*$/, '')
            .replace(/_+/g, ' ')
            .replace(/\s+/g, ' ')
            .trim()
            .slice(0, 36);
    }

    function formatPage(p) {
        if (p == null) return '?';
        const n = Number(p);
        return Number.isFinite(n) ? Math.trunc(n) : p;
    }

    function parseSse(raw) {
        let event = 'message';
        let dataLines = [];
        for (const line of raw.split('\n')) {
            if (line.startsWith('event:')) event = line.slice(6).trim();
            else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim());
        }
        if (!dataLines.length) return null;
        try { return { event, data: JSON.parse(dataLines.join('\n')) }; }
        catch { return { event, data: dataLines.join('\n') }; }
    }

    function el(tag, attrs) {
        const e = document.createElement(tag);
        if (attrs) for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
        return e;
    }

    function escapeHtml(s) {
        return String(s).replace(/[&<>"']/g, (c) => (
            { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
        ));
    }
})();
