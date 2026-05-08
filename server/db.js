// PostgreSQL data layer for the BSA chat front layer.
//
// One database holds three tables:
//   sessions  — one row per visitor conversation
//   messages  — every user / assistant / human-admin message
//   handoffs  — open or resolved human-handoff requests
//
// All functions are async (pg Pool is Promise-based).
// Call init() once at startup to create tables if they don't exist.
//
// Env:
//   DATABASE_URL   — e.g. postgres://user:pass@localhost:5432/bsa_chat
//   DATABASE_SSL   — set to 'true' for cloud providers (Render, Supabase, etc.)

const { Pool } = require('pg');
const crypto = require('crypto');

const pool = new Pool({
    connectionString: process.env.DATABASE_URL,
    ssl: process.env.DATABASE_SSL === 'true' ? { rejectUnauthorized: false } : false,
});

const now = () => Date.now();
const newId = () => crypto.randomUUID();

// ─────────────────────── schema init ───────────────────────
async function init() {
    // Run each statement separately so IF NOT EXISTS checks work cleanly.
    await pool.query(`
        CREATE TABLE IF NOT EXISTS sessions (
            id           TEXT PRIMARY KEY,
            dealer_id    TEXT,
            user_name    TEXT,
            user_email   TEXT,
            created_at   BIGINT NOT NULL,
            updated_at   BIGINT NOT NULL
        )
    `);
    await pool.query(`
        CREATE TABLE IF NOT EXISTS messages (
            id           BIGSERIAL PRIMARY KEY,
            session_id   TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            role         TEXT NOT NULL CHECK (role IN ('user','assistant','admin','system')),
            content      TEXT NOT NULL,
            citations    TEXT,
            top_score    REAL,
            created_at   BIGINT NOT NULL
        )
    `);
    await pool.query(`
        CREATE INDEX IF NOT EXISTS idx_messages_session_created
            ON messages(session_id, created_at)
    `);
    await pool.query(`
        CREATE TABLE IF NOT EXISTS handoffs (
            id           BIGSERIAL PRIMARY KEY,
            session_id   TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            reason       TEXT,
            status       TEXT NOT NULL CHECK (status IN ('open','resolved')),
            created_at   BIGINT NOT NULL,
            resolved_at  BIGINT,
            UNIQUE(session_id, status)
        )
    `);
    await pool.query(`
        CREATE INDEX IF NOT EXISTS idx_handoffs_status_created
            ON handoffs(status, created_at)
    `);
    await pool.query(`
        CREATE TABLE IF NOT EXISTS learned_qa (
            id          TEXT PRIMARY KEY,
            question    TEXT NOT NULL,
            answer      TEXT NOT NULL,
            dealer_id   TEXT,
            created_at  BIGINT NOT NULL,
            updated_at  BIGINT NOT NULL
        )
    `);
}

// ─────────────────────── sessions ───────────────────────
async function createSession(dealerId) {
    const id = newId();
    const t = now();
    await pool.query(
        `INSERT INTO sessions (id, dealer_id, created_at, updated_at)
         VALUES ($1, $2, $3, $4)`,
        [id, dealerId || null, t, t],
    );
    return id;
}

async function getOrCreateSession(sessionId, dealerId) {
    if (sessionId) {
        const { rows } = await pool.query(
            `SELECT id FROM sessions WHERE id = $1`, [sessionId],
        );
        if (rows.length > 0) {
            await pool.query(
                `UPDATE sessions SET updated_at = $1 WHERE id = $2`,
                [now(), sessionId],
            );
            return sessionId;
        }
    }
    return createSession(dealerId);
}

async function setSessionContact(sessionId, name, email) {
    await pool.query(
        `UPDATE sessions SET user_name = $1, user_email = $2, updated_at = $3
         WHERE id = $4`,
        [name || null, email || null, now(), sessionId],
    );
}

// ─────────────────────── messages ───────────────────────
async function addMessage(sessionId, role, content, opts = {}) {
    const { citations = null, topScore = null } = opts;
    const { rows } = await pool.query(
        `INSERT INTO messages (session_id, role, content, citations, top_score, created_at)
         VALUES ($1, $2, $3, $4, $5, $6) RETURNING id`,
        [
            sessionId,
            role,
            content,
            citations ? JSON.stringify(citations) : null,
            topScore,
            now(),
        ],
    );
    await pool.query(
        `UPDATE sessions SET updated_at = $1 WHERE id = $2`,
        [now(), sessionId],
    );
    return rows[0].id;
}

async function getMessages(sessionId) {
    const { rows } = await pool.query(
        `SELECT id, role, content, citations, top_score, created_at
         FROM messages WHERE session_id = $1 ORDER BY id ASC`,
        [sessionId],
    );
    return rows.map(parseRow);
}

async function getMessagesAfter(sessionId, lastId) {
    const { rows } = await pool.query(
        `SELECT id, role, content, citations, top_score, created_at
         FROM messages WHERE session_id = $1 AND id > $2 ORDER BY id ASC`,
        [sessionId, lastId || 0],
    );
    return rows.map(parseRow);
}

// Returns prior turns oldest-first as [{question, answer}] for the
// Python /chat endpoint. limitPairs applies to Q+A pairs.
async function getHistoryPairs(sessionId, limitPairs) {
    if (!limitPairs || limitPairs <= 0) return [];
    const { rows } = await pool.query(
        `SELECT role, content FROM messages
         WHERE session_id = $1 AND role IN ('user', 'assistant')
         ORDER BY id DESC LIMIT $2`,
        [sessionId, limitPairs * 2],
    );
    const reversed = rows.reverse();
    const pairs = [];
    let pendingUser = null;
    for (const row of reversed) {
        if (row.role === 'user') {
            pendingUser = row.content;
        } else if (row.role === 'assistant' && pendingUser) {
            pairs.push({ question: pendingUser, answer: row.content });
            pendingUser = null;
        }
    }
    return pairs.slice(-limitPairs);
}

async function getLastUserQuestion(sessionId) {
    const { rows } = await pool.query(
        `SELECT content FROM messages
         WHERE session_id = $1 AND role = 'user'
         ORDER BY id DESC LIMIT 1`,
        [sessionId],
    );
    return rows.length > 0 ? rows[0].content : null;
}

// ─────────────────────── handoffs ───────────────────────
async function openHandoff(sessionId, reason) {
    await pool.query(
        `INSERT INTO handoffs (session_id, reason, status, created_at)
         VALUES ($1, $2, 'open', $3)
         ON CONFLICT(session_id, status) DO NOTHING`,
        [sessionId, reason || 'user_requested', now()],
    );
    const { rows } = await pool.query(
        `SELECT * FROM handoffs WHERE session_id = $1 AND status = 'open'`,
        [sessionId],
    );
    return rows[0] || null;
}

async function resolveHandoff(sessionId) {
    await pool.query(
        `UPDATE handoffs SET status = 'resolved', resolved_at = $1
         WHERE session_id = $2 AND status = 'open'`,
        [now(), sessionId],
    );
}

async function listOpenHandoffs() {
    const { rows } = await pool.query(
        `SELECT h.id, h.session_id, h.reason, h.created_at,
                s.dealer_id, s.user_name, s.user_email,
                (SELECT content FROM messages
                 WHERE session_id = h.session_id AND role = 'user'
                 ORDER BY id DESC LIMIT 1) AS last_question
         FROM handoffs h
         JOIN sessions s ON s.id = h.session_id
         WHERE h.status = 'open'
         ORDER BY h.created_at ASC`,
    );
    return rows;
}

async function hasOpenHandoff(sessionId) {
    const { rows } = await pool.query(
        `SELECT id FROM handoffs WHERE session_id = $1 AND status = 'open'`,
        [sessionId],
    );
    return rows.length > 0;
}

// ─────────────────────── helpers ───────────────────────
function parseRow(row) {
    return {
        id: row.id,
        role: row.role,
        content: row.content,
        citations: row.citations ? JSON.parse(row.citations) : null,
        topScore: row.top_score,
        createdAt: row.created_at,
    };
}

// ─────────────────────── learned Q&A ───────────────────────
async function addLearnedQA(id, question, answer, dealerId) {
    const t = now();
    await pool.query(
        `INSERT INTO learned_qa (id, question, answer, dealer_id, created_at, updated_at)
         VALUES ($1,$2,$3,$4,$5,$5)
         ON CONFLICT (id) DO UPDATE SET question=$2, answer=$3, updated_at=$5`,
        [id, question, answer, dealerId || null, t],
    );
}

async function listLearnedQA() {
    const { rows } = await pool.query(
        `SELECT id, question, answer, dealer_id, created_at, updated_at
         FROM learned_qa ORDER BY updated_at DESC`,
    );
    return rows;
}

async function updateLearnedQA(id, question, answer) {
    await pool.query(
        `UPDATE learned_qa SET question=$2, answer=$3, updated_at=$4 WHERE id=$1`,
        [id, question, answer, now()],
    );
}

async function deleteLearnedQA(id) {
    await pool.query(`DELETE FROM learned_qa WHERE id=$1`, [id]);
}

async function searchLearnedQA(query) {
    const { rows } = await pool.query(
        `SELECT id, question, answer, dealer_id, created_at
         FROM learned_qa
         WHERE question ILIKE $1 OR answer ILIKE $1
         ORDER BY updated_at DESC LIMIT 5`,
        [`%${query}%`],
    );
    return rows;
}

async function getAnalytics(days = 7) {
    const since = (Date.now() - days * 86400 * 1000);

    const NO_ANS = `(
        content ILIKE '%couldn''t find%' OR
        content ILIKE '%could not find%' OR
        content ILIKE '%don''t have%' OR
        content ILIKE '%do not have%' OR
        content ILIKE '%not covered in%' OR
        content ILIKE '%no relevant information%' OR
        content ILIKE '%i''m not finding%' OR
        content ILIKE '%i can''t find%' OR
        content ILIKE '%outside%scope%' OR
        content ILIKE '%talk to a human%'
    )`;

    const ov = await pool.query(`
        SELECT
            (SELECT COUNT(DISTINCT id) FROM sessions) AS total_sessions,
            COUNT(*) FILTER (WHERE role = 'user') AS total_questions,
            COUNT(*) FILTER (WHERE role = 'assistant' AND NOT ${NO_ANS}) AS answered,
            COUNT(*) FILTER (WHERE role = 'assistant' AND ${NO_ANS}) AS unanswered
        FROM messages
    `);

    const hf = await pool.query(`
        SELECT
            COUNT(*) FILTER (WHERE status = 'open')              AS open_count,
            COUNT(*) FILTER (WHERE status = 'resolved')          AS resolved_count,
            COUNT(*) FILTER (WHERE reason = 'low_confidence')    AS low_confidence,
            COUNT(*) FILTER (WHERE reason = 'user_requested')    AS user_requested
        FROM handoffs
    `);

    const daily = await pool.query(`
        SELECT
            TO_CHAR(TO_TIMESTAMP(created_at / 1000.0), 'YYYY-MM-DD') AS day,
            COUNT(*) FILTER (WHERE role = 'user')                              AS questions,
            COUNT(*) FILTER (WHERE role = 'assistant' AND NOT ${NO_ANS})       AS answered,
            COUNT(*) FILTER (WHERE role = 'assistant' AND ${NO_ANS})           AS unanswered
        FROM messages
        WHERE created_at >= $1
        GROUP BY day
        ORDER BY day ASC
    `, [since]);

    const unans = await pool.query(`
        WITH no_ans AS (
            SELECT id, session_id, created_at
            FROM messages
            WHERE role = 'assistant' AND ${NO_ANS}
            ORDER BY created_at DESC
            LIMIT 30
        )
        SELECT
            u.content    AS question,
            n.session_id,
            n.created_at
        FROM no_ans n
        JOIN LATERAL (
            SELECT content FROM messages
            WHERE session_id = n.session_id AND id < n.id AND role = 'user'
            ORDER BY id DESC LIMIT 1
        ) u ON true
        ORDER BY n.created_at DESC
    `);

    return {
        overview:  ov.rows[0],
        handoffs:  hf.rows[0],
        daily:     daily.rows,
        unanswered: unans.rows,
    };
}

module.exports = {
    pool,
    init,
    createSession,
    getOrCreateSession,
    setSessionContact,
    addMessage,
    getMessages,
    getMessagesAfter,
    getHistoryPairs,
    openHandoff,
    resolveHandoff,
    listOpenHandoffs,
    hasOpenHandoff,
    getLastUserQuestion,
    getAnalytics,
    addLearnedQA,
    listLearnedQA,
    updateLearnedQA,
    deleteLearnedQA,
    searchLearnedQA,
};
