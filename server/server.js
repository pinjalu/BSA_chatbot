// BSA chat front layer — Express server.
//
// Routes
//   GET  /widget.js                  — vanilla JS chat widget (dealers embed this)
//   POST /api/chat                   — SSE stream of assistant answer
//   POST /api/handoff                — flag conversation for human reply
//   GET  /api/messages/:sessionId    — poll for new messages since lastId
//   GET  /admin                      — admin login + queue page (HTML)
//   GET  /api/admin/handoffs         — list open handoffs (auth required)
//   GET  /api/admin/session/:id      — full transcript for one session (auth)
//   POST /api/admin/reply            — human-admin posts a reply (auth)
//   POST /api/admin/resolve          — close a handoff without replying (auth)

const path = require('path');
const fs = require('fs');
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const { request, FormData, fetch } = require('undici');
require('dotenv').config({ path: path.join(__dirname, '.env') });

const db = require('./db');

// Project root — one level up from server/ — used to safely resolve
// image paths that the Python RAG core surfaces in [[SHOW_IMAGE]] tags.
const PROJECT_ROOT = path.resolve(__dirname, '..');

const PORT = parseInt(process.env.PORT || '3000', 10);
const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://127.0.0.1:8000';
const PIPELINE_API_URL = process.env.PIPELINE_API_URL || 'http://127.0.0.1:8001';
const ADMIN_PASSWORD = process.env.ADMIN_PASSWORD || '';
const CONFIDENCE_THRESHOLD = parseFloat(process.env.CONFIDENCE_THRESHOLD || '0.45');
const HISTORY_TURNS = parseInt(process.env.HISTORY_TURNS || '5', 10);
const CORS_ORIGINS = (process.env.CORS_ORIGINS || '*')
    .split(',').map(s => s.trim()).filter(Boolean);

// Heuristic: when the assistant's reply matches one of these phrases,
// it's saying "I don't have the answer" regardless of what the rerank
// score was. We force a handoff suggestion in that case so the widget
// surfaces the "Talk to a human" prompt automatically.
const NO_ANSWER_RX = new RegExp(
    [
        "couldn'?t find",
        "could not find",
        "don'?t have (?:that|this|the answer|enough info)",
        "do not have (?:that|this|the answer)",
        "passages? (?:don'?t|do not) cover",
        "not (?:in|covered in) (?:the )?(?:manuals?|docs?|documentation|context|excerpts?)",
        "no (?:relevant )?information",
        "i'?m not finding",
        "i can'?t find",
        "outside (?:the )?scope",
    ].join('|'),
    'i',
);

if (!ADMIN_PASSWORD) {
    console.warn('[warn] ADMIN_PASSWORD is empty — admin panel is OPEN');
}

const app = express();

// CORS — the widget is loaded from a dealer site origin different from
// this server, so /api/* must allow it. /admin is same-origin, no CORS needed.
app.use('/api', cors({
    origin: CORS_ORIGINS.includes('*') ? true : CORS_ORIGINS,
    methods: ['GET', 'POST', 'OPTIONS'],
    credentials: false,
}));
app.use('/widget.js', cors({ origin: true }));

app.use(express.json({ limit: '64kb' }));
app.use(express.static(path.join(__dirname, 'public'), {
    setHeaders: (res, filePath) => {
        if (filePath.endsWith(path.sep + 'widget.js')) {
            res.setHeader('Cache-Control', 'no-store, max-age=0');
            res.setHeader('Pragma', 'no-cache');
            res.setHeader('Expires', '0');
        }
    },
}));

// ─────────────────────── images ───────────────────────
//
// Python's /chat endpoint returns image_paths like
// "Data/Data/Bantam/images/page_1_img_2.png" inside its meta event.
// The widget asks for these via /api/images/<that-path>. We resolve
// the file under PROJECT_ROOT and refuse anything that escapes (no
// ".." or absolute paths) before streaming the bytes back.
const ALLOWED_IMG_EXT = new Set(['.png', '.jpg', '.jpeg', '.gif', '.webp']);
app.get('/api/images/*', cors({ origin: true }), (req, res) => {
    const rel = decodeURIComponent(req.params[0] || '');
    const ext = path.extname(rel).toLowerCase();
    if (!rel || !ALLOWED_IMG_EXT.has(ext)) {
        return res.status(400).send('bad path');
    }
    const target = path.resolve(PROJECT_ROOT, rel);
    if (!target.startsWith(PROJECT_ROOT + path.sep)
            && target !== PROJECT_ROOT) {
        return res.status(403).send('out of bounds');
    }
    fs.stat(target, (err, st) => {
        if (err || !st.isFile()) return res.status(404).send('not found');
        res.type(ext);
        res.setHeader('Cache-Control', 'public, max-age=86400');
        fs.createReadStream(target).pipe(res);
    });
});

// ─────────────────────── widget + admin pages ───────────────────────
app.get('/widget.js', (_req, res) => {
    res.type('application/javascript');
    // Always serve the latest widget code; stale cached JS causes
    // old streaming behaviour to persist on dealer sites.
    res.setHeader('Cache-Control', 'no-store, max-age=0');
    res.setHeader('Pragma', 'no-cache');
    res.setHeader('Expires', '0');
    res.sendFile(path.join(__dirname, 'public', 'widget.js'));
});

app.get('/admin', (_req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'admin.html'));
});

app.get('/', (_req, res) => {
    res.type('text/plain').send(
        'BSA chat server running. Admin: /admin   Widget: /widget.js',
    );
});

// ─────────────────────── chat (streaming) ───────────────────────
app.post('/api/chat', async (req, res) => {
    const { sessionId: incomingSessionId, dealerId, message,
            vehicle, docType, imageB64 } = req.body || {};
    if (!message || typeof message !== 'string') {
        return res.status(400).json({ error: 'message is required' });
    }

    const sessionId = await db.getOrCreateSession(incomingSessionId, dealerId);

    // Persist user turn before calling the model. If the LLM call fails
    // halfway, we still have the question on record for retry / handoff.
    await db.addMessage(sessionId, 'user', message);

    const history = (await db.getHistoryPairs(sessionId, HISTORY_TURNS))
        .filter(p => p.question !== message || p.answer);

    // Set up SSE response to the browser. We stream a small superset of
    // the Python events so the widget can ignore python-internal noise.
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no');
    res.flushHeaders();

    const send = (event, data) => {
        res.write(`event: ${event}\n`);
        res.write(`data: ${JSON.stringify(data)}\n\n`);
        if (typeof res.flush === 'function') res.flush();
    };

    send('session', { sessionId });

    let pyResp;
    try {
        pyResp = await request(`${PYTHON_API_URL}/chat`, {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify({
                message,
                history,
                vehicle: vehicle || null,
                doc_type: docType || null,
                image_b64: imageB64 || null,
            }),
        });
    } catch (e) {
        send('error', { message: `RAG core unreachable: ${e.message}` });
        return res.end();
    }

    if (pyResp.statusCode >= 400) {
        const text = await pyResp.body.text().catch(() => '');
        send('error', { message: `RAG core ${pyResp.statusCode}: ${text}` });
        return res.end();
    }

    // Parse SSE from Python and re-emit to the browser, accumulating the
    // answer + meta to persist when the stream finishes.
    let assistantText = '';
    let topScore = 0;
    let citations = null;
    let buf = '';

    pyResp.body.on('data', (chunk) => {
        buf += chunk.toString('utf8');
        let m;
        while ((m = /\r?\n\r?\n/.exec(buf)) !== null) {
            const idx = m.index;
            const sepLen = m[0].length;
            const raw = buf.slice(0, idx);
            buf = buf.slice(idx + sepLen);
            const ev = parseSseEvent(raw);
            if (!ev) continue;

            if (ev.event === 'token') {
                assistantText += ev.data.text || '';
                send('token', { text: ev.data.text });
            } else if (ev.event === 'meta') {
                topScore = ev.data.top_score || 0;
                citations = ev.data.matches || [];
                // Convert Python's IMG_N → path map into a {IMG_N: url}
                // the widget can load directly. After the S3 migration
                // these are full https URLs in Pinecone — pass those
                // through unchanged. Anything still on a local path
                // gets the legacy /api/images/<path> wrapper.
                const rawImages = ev.data.images || {};
                const images = {};
                for (const [id, p] of Object.entries(rawImages)) {
                    if (typeof p === 'string' && p) {
                        if (/^https?:\/\//i.test(p)) {
                            images[id] = p;
                        } else {
                            images[id] = '/api/images/'
                                + p.split(/[\\/]/).map(encodeURIComponent).join('/');
                        }
                    }
                }
                if (ev.data.learned) learnedMeta = ev.data.learned;
                send('meta', {
                    topScore, citations, images,
                    learned: ev.data.learned || null,
                });
            } else if (ev.event === 'error') {
                send('error', ev.data);
            } else if (ev.event === 'done') {
                // The 'done' event from Python carries per-stage timings
                // (embed / pinecone / TTFT / LLM total / total) and a
                // `learned: true` flag if this was a saved-answer hit.
                // We stash them here and the upstream 'end' handler
                // folds them into the widget-facing 'done' event.
                if (ev.data && ev.data.timings_ms) {
                    timings = ev.data.timings_ms;
                }
                if (ev.data && ev.data.learned) {
                    learnedMeta = learnedMeta || { match_score: topScore };
                }
            }
        }
    });

    let timings = null;
    let learnedMeta = null;

    pyResp.body.on('end', async () => {
        if (assistantText) {
            await db.addMessage(sessionId, 'assistant', assistantText, {
                citations,
                topScore,
            });
        }

        // Auto-handoff trigger fires on EITHER signal:
        //   1. Low rerank confidence (top match below threshold), OR
        //   2. The model's reply itself says it doesn't have the
        //      answer — in that case it doesn't matter what the
        //      retrieval score was, the user clearly didn't get
        //      what they asked for.
        const lowConfidence = topScore < CONFIDENCE_THRESHOLD;
        const noAnswerReply = NO_ANSWER_RX.test(assistantText || '');
        const alreadyOpen = await db.hasOpenHandoff(sessionId);
        send('done', {
            topScore,
            suggestHandoff: (lowConfidence || noAnswerReply) && !alreadyOpen,
            handoffOpen: alreadyOpen,
            timings,
            learned: learnedMeta,
        });
        if (timings) {
            console.log(
                `[chat] total=${timings.total}ms `
                + `embed=${timings.embed}ms `
                + `pinecone=${timings.pinecone}ms `
                + `ttft=${timings.llm_ttft}ms `
                + `llm=${timings.llm_total}ms`,
            );
        }
        res.end();
    });

    pyResp.body.on('error', (e) => {
        send('error', { message: `stream error: ${e.message}` });
        res.end();
    });

    req.on('close', () => {
        // Browser disconnected mid-stream. Close the upstream too.
        pyResp.body.destroy();
    });
});

// ─────────────────────── handoff ───────────────────────
app.post('/api/handoff', async (req, res) => {
    const { sessionId, name, email, reason } = req.body || {};
    if (!sessionId) {
        return res.status(400).json({ error: 'sessionId required' });
    }
    const session = await db.getOrCreateSession(sessionId);
    if (name || email) await db.setSessionContact(session, name, email);
    const handoff = await db.openHandoff(session, reason || 'user_requested');
    await db.addMessage(
        session, 'system',
        `Handoff requested${name ? ` by ${name}` : ''}` +
        `${email ? ` (${email})` : ''}.`,
    );
    res.json({ ok: true, handoff });
});

// ─────────────────────── polling for new messages ───────────────────────
app.get('/api/messages/:sessionId', async (req, res) => {
    const { sessionId } = req.params;
    const after = parseInt(req.query.after || '0', 10);
    const messages = await db.getMessagesAfter(sessionId, after);
    const handoffOpen = await db.hasOpenHandoff(sessionId);
    res.json({ messages, handoffOpen });
});

// ─────────────────────── admin auth ───────────────────────
function requireAdmin(req, res, next) {
    const auth = req.get('authorization') || '';
    const expected = `Bearer ${ADMIN_PASSWORD}`;
    if (!ADMIN_PASSWORD || auth !== expected) {
        return res.status(401).json({ error: 'unauthorized' });
    }
    next();
}

app.post('/api/admin/login', (req, res) => {
    const { password } = req.body || {};
    if (!ADMIN_PASSWORD || password !== ADMIN_PASSWORD) {
        return res.status(401).json({ error: 'wrong password' });
    }
    res.json({ token: ADMIN_PASSWORD });
});

app.get('/api/admin/handoffs', requireAdmin, async (_req, res) => {
    res.json({ handoffs: await db.listOpenHandoffs() });
});

app.get('/api/admin/session/:id', requireAdmin, async (req, res) => {
    const messages = await db.getMessages(req.params.id);
    res.json({ sessionId: req.params.id, messages });
});

app.post('/api/admin/reply', requireAdmin, async (req, res) => {
    const { sessionId, message, resolve } = req.body || {};
    if (!sessionId || !message) {
        return res.status(400).json({ error: 'sessionId and message required' });
    }
    await db.addMessage(sessionId, 'admin', message);

    let learned = null;
    if (resolve) {
        await db.resolveHandoff(sessionId);
        // Save this admin reply as a learned answer for future visitors.
        // The "question" is the visitor's most recent message in this
        // session — the one that triggered the human handoff. Errors
        // here don't block the reply itself; we just log and skip.
        const userQuestion = await db.getLastUserQuestion(sessionId);
        if (userQuestion) {
            try {
                const r = await request(`${PYTHON_API_URL}/learn`, {
                    method: 'POST',
                    headers: { 'content-type': 'application/json' },
                    body: JSON.stringify({
                        question: userQuestion,
                        answer: message,
                        session_id: sessionId,
                    }),
                });
                if (r.statusCode < 400) {
                    learned = await r.body.json();
                    console.log(
                        `[learn] saved admin reply for future similar Qs `
                        + `(session=${sessionId}, id=${learned.id})`,
                    );
                } else {
                    const t = await r.body.text().catch(() => '');
                    console.warn(`[learn] failed: ${r.statusCode} ${t}`);
                }
            } catch (e) {
                console.warn(`[learn] error reaching Python: ${e.message}`);
            }
        }
    }
    res.json({ ok: true, learned });
});

app.post('/api/admin/resolve', requireAdmin, async (req, res) => {
    const { sessionId } = req.body || {};
    if (!sessionId) return res.status(400).json({ error: 'sessionId required' });
    await db.resolveHandoff(sessionId);
    res.json({ ok: true });
});

// ─────────────────────── pipeline (PDF ingestion) ───────────────────────
//
// All pipeline routes proxy to the Python pipeline_api.py on
// PIPELINE_API_URL. Uploads stream the PDF body straight through;
// status / log SSE is relayed line-by-line.

const pdfUpload = multer({
    storage: multer.memoryStorage(),
    limits: { fileSize: 200 * 1024 * 1024 },  // 200 MiB cap — workshop manuals are big
    fileFilter: (_req, file, cb) => {
        if (file.mimetype === 'application/pdf'
                || file.originalname.toLowerCase().endsWith('.pdf')) {
            return cb(null, true);
        }
        cb(new Error('only PDF files are accepted'));
    },
});

app.post('/api/admin/pipeline/upload', requireAdmin,
    pdfUpload.single('file'),
    async (req, res) => {
        if (!req.file) {
            return res.status(400).json({ error: 'file field required' });
        }
        try {
            const fd = new FormData();
            fd.set('file', new Blob([req.file.buffer],
                { type: 'application/pdf' }), req.file.originalname);
            // Forward optional form fields
            if (req.body.skip)     fd.set('skip',     req.body.skip);
            if (req.body.force)    fd.set('force',    req.body.force);
            if (req.body.category) fd.set('category', req.body.category);

            const r = await fetch(`${PIPELINE_API_URL}/pipeline/upload`, {
                method: 'POST',
                body: fd,
            });
            const text = await r.text();
            res.status(r.status).type(r.headers.get('content-type')
                || 'application/json').send(text);
        } catch (e) {
            console.error('[pipeline] upload relay failed:', e);
            res.status(502).json({ error: 'pipeline service unreachable' });
        }
    });

app.get('/api/admin/pipeline/jobs', requireAdmin, async (_req, res) => {
    try {
        const r = await fetch(`${PIPELINE_API_URL}/pipeline/jobs`);
        const text = await r.text();
        res.status(r.status).type('application/json').send(text);
    } catch (e) {
        res.status(502).json({ error: 'pipeline service unreachable' });
    }
});

app.get('/api/admin/pipeline/jobs/:id', requireAdmin, async (req, res) => {
    try {
        const r = await fetch(
            `${PIPELINE_API_URL}/pipeline/jobs/${encodeURIComponent(req.params.id)}`,
        );
        const text = await r.text();
        res.status(r.status).type('application/json').send(text);
    } catch (e) {
        res.status(502).json({ error: 'pipeline service unreachable' });
    }
});

app.delete('/api/admin/pipeline/jobs/:id', requireAdmin, async (req, res) => {
    try {
        const r = await fetch(
            `${PIPELINE_API_URL}/pipeline/jobs/${encodeURIComponent(req.params.id)}`,
            { method: 'DELETE' },
        );
        const text = await r.text();
        res.status(r.status).type('application/json').send(text);
    } catch (e) {
        res.status(502).json({ error: 'pipeline service unreachable' });
    }
});

app.post('/api/admin/pipeline/jobs/:id/retry', requireAdmin, async (req, res) => {
    try {
        const fd = new FormData();
        if (req.body.skip)  fd.set('skip',  req.body.skip);
        if (req.body.force) fd.set('force', req.body.force);
        const r = await fetch(
            `${PIPELINE_API_URL}/pipeline/jobs/${encodeURIComponent(req.params.id)}/retry`,
            { method: 'POST', body: fd },
        );
        const text = await r.text();
        res.status(r.status).type('application/json').send(text);
    } catch (e) {
        res.status(502).json({ error: 'pipeline service unreachable' });
    }
});

app.post('/api/admin/pipeline/jobs/:id/cancel', requireAdmin, async (req, res) => {
    try {
        const r = await fetch(
            `${PIPELINE_API_URL}/pipeline/jobs/${encodeURIComponent(req.params.id)}/cancel`,
            { method: 'POST' },
        );
        const text = await r.text();
        res.status(r.status).type('application/json').send(text);
    } catch (e) {
        res.status(502).json({ error: 'pipeline service unreachable' });
    }
});

// SSE relay: client → Node → Python. We can't use the same auth
// because EventSource doesn't send custom headers. Instead, accept
// the admin token as a query param and check it here.
app.get('/api/admin/pipeline/jobs/:id/stream', async (req, res) => {
    if (!ADMIN_PASSWORD || req.query.token !== ADMIN_PASSWORD) {
        return res.status(401).end();
    }
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no');
    res.flushHeaders();

    let upstream;
    try {
        upstream = await request(
            `${PIPELINE_API_URL}/pipeline/jobs/${encodeURIComponent(req.params.id)}/stream`,
        );
    } catch (e) {
        res.write(`event: error\ndata: ${JSON.stringify({
            message: 'pipeline service unreachable: ' + e.message,
        })}\n\n`);
        return res.end();
    }
    if (upstream.statusCode >= 400) {
        const body = await upstream.body.text().catch(() => '');
        res.write(`event: error\ndata: ${JSON.stringify({
            status: upstream.statusCode, body,
        })}\n\n`);
        return res.end();
    }

    upstream.body.on('data', chunk => res.write(chunk));
    upstream.body.on('end',   () => res.end());
    upstream.body.on('error', e => {
        res.write(`event: error\ndata: ${JSON.stringify({
            message: e.message,
        })}\n\n`);
        res.end();
    });
    req.on('close', () => upstream.body.destroy());
});

app.get('/api/admin/analytics', requireAdmin, async (req, res) => {
    const days = parseInt(req.query.days || '7', 10);
    const data = await db.getAnalytics(days);
    res.json(data);
});

// ─────────────────────── content management ───────────────────────
app.get('/api/admin/content', requireAdmin, async (_req, res) => {
    res.json({ items: await db.listLearnedQA() });
});

app.post('/api/admin/content', requireAdmin, async (req, res) => {
    const { question, answer, dealer_id } = req.body || {};
    if (!question || !answer) return res.status(400).json({ error: 'question and answer required' });
    const r = await request(`${PYTHON_API_URL}/learn`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ question, answer, dealer_id: dealer_id || '' }),
    });
    if (r.statusCode >= 400) return res.status(502).json({ error: 'python error' });
    const data = await r.body.json();
    await db.addLearnedQA(data.id, question, answer, dealer_id);
    res.json({ ok: true, id: data.id });
});

app.put('/api/admin/content/:id', requireAdmin, async (req, res) => {
    const { question, answer } = req.body || {};
    if (!question || !answer) return res.status(400).json({ error: 'question and answer required' });
    const r = await request(`${PYTHON_API_URL}/learned/${encodeURIComponent(req.params.id)}`, {
        method: 'PUT',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ question, answer }),
    });
    if (r.statusCode >= 400) return res.status(502).json({ error: 'python error' });
    await db.updateLearnedQA(req.params.id, question, answer);
    res.json({ ok: true });
});

app.delete('/api/admin/content/:id', requireAdmin, async (req, res) => {
    await request(`${PYTHON_API_URL}/learned/${encodeURIComponent(req.params.id)}`, { method: 'DELETE' });
    await db.deleteLearnedQA(req.params.id);
    res.json({ ok: true });
});

// ─────────────────────── knowledge base ───────────────────────
// SSE stream — same answer as chat widget, just without session tracking
app.post('/api/kb/search', cors({ origin: true }), async (req, res) => {
    const { query, vehicle } = req.body || {};
    if (!query) return res.status(400).json({ error: 'query required' });

    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no');
    res.flushHeaders();

    const send = (event, data) => {
        res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
        if (typeof res.flush === 'function') res.flush();
    };

    let pyResp;
    try {
        pyResp = await request(`${PYTHON_API_URL}/chat`, {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify({
                message: query,
                history: [],
                vehicle: vehicle || null,
            }),
        });
    } catch (e) {
        send('error', { message: `RAG core unreachable: ${e.message}` });
        return res.end();
    }

    if (pyResp.statusCode >= 400) {
        send('error', { message: `RAG core error ${pyResp.statusCode}` });
        return res.end();
    }

    let buf = '';
    pyResp.body.on('data', (chunk) => {
        buf += chunk.toString('utf8');
        let m;
        while ((m = /\r?\n\r?\n/.exec(buf)) !== null) {
            const raw = buf.slice(0, m.index);
            buf = buf.slice(m.index + m[0].length);
            const ev = parseSseEvent(raw);
            if (!ev) continue;
            if (ev.event === 'token') send('token', ev.data);
            else if (ev.event === 'status') send('status', ev.data);
            else if (ev.event === 'meta') {
                const rawImages = ev.data.images || {};
                const images = {};
                for (const [id, p] of Object.entries(rawImages)) {
                    if (typeof p === 'string' && p) {
                        images[id] = /^https?:\/\//i.test(p) ? p
                            : '/api/images/' + p.split(/[\\/]/).map(encodeURIComponent).join('/');
                    }
                }
                send('meta', { citations: ev.data.matches || [], images });
            } else if (ev.event === 'done') {
                send('done', { timings: ev.data.timings_ms || null });
            } else if (ev.event === 'error') {
                send('error', ev.data);
            }
        }
    });
    pyResp.body.on('end', () => res.end());
    pyResp.body.on('error', (e) => { send('error', { message: e.message }); res.end(); });
    req.on('close', () => pyResp.body.destroy());
});

app.get('/admin/analytics', (_req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'analytics.html'));
});

app.get('/admin/content', (_req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'content.html'));
});

app.post('/api/kb/handoff', cors({ origin: true }), async (req, res) => {
    const { name, email, question } = req.body || {};
    if (!email) return res.status(400).json({ error: 'email required' });
    const sessionId = await db.createSession('kb');
    if (name || email) await db.setSessionContact(sessionId, name, email);
    if (question) await db.addMessage(sessionId, 'user', question);
    await db.openHandoff(sessionId, 'kb_search');
    await db.addMessage(sessionId, 'system',
        `KB handoff${name ? ` from ${name}` : ''}${email ? ` (${email})` : ''}. Question: "${question || '—'}"`);
    res.json({ ok: true, sessionId });
});

app.get('/kb', (_req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'kb.html'));
});

app.get('/admin/pipeline', (_req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'pipeline.html'));
});

// ─────────────────────── React admin app ───────────────────────
// Serve the Vite-built React app from /app/
// The build output lands in server/public/app/ (configured in vite.config.js).
// Catch-all for /app/* routes so React Router can handle client-side navigation.
app.get('/app', (_req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'app', 'index.html'));
});
app.get('/app/*', (_req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'app', 'index.html'));
});

// ─────────────────────── helpers ───────────────────────
function parseSseEvent(raw) {
    let event = 'message';
    let dataLines = [];
    for (const line of raw.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim());
    }
    if (!dataLines.length) return null;
    try {
        return { event, data: JSON.parse(dataLines.join('\n')) };
    } catch {
        return { event, data: dataLines.join('\n') };
    }
}

// ─────────────────────── start ───────────────────────
db.init()
    .then(() => {
        app.listen(PORT, () => {
            console.log(`BSA chat server on http://127.0.0.1:${PORT}`);
            console.log(`  Python RAG core   : ${PYTHON_API_URL}`);
            console.log(`  Confidence cutoff : ${CONFIDENCE_THRESHOLD}`);
            console.log(`  Admin panel       : http://127.0.0.1:${PORT}/admin`);
        });
    })
    .catch(err => {
        console.error('[fatal] Database init failed:', err.message);
        process.exit(1);
    });
