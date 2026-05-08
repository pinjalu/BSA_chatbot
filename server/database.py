"""Async PostgreSQL data layer — replaces server/db.js.

Tables:
  sessions   — one row per visitor conversation
  messages   — every user / assistant / human-admin message
  handoffs   — open or resolved human-handoff requests
  learned_qa — saved Q&A pairs from admin replies

Env:
  DATABASE_URL  — e.g. postgres://user:pass@localhost:5432/bsa_chat
  DATABASE_SSL  — set to 'true' for cloud providers
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, Optional

import asyncpg

_pool: Optional[asyncpg.Pool] = None


def _now() -> int:
    return int(time.time() * 1000)


def _new_id() -> str:
    return str(uuid.uuid4())


def _parse_row(row: asyncpg.Record) -> dict:
    return {
        "id": row["id"],
        "role": row["role"],
        "content": row["content"],
        "citations": json.loads(row["citations"]) if row["citations"] else None,
        "topScore": row["top_score"],
        "createdAt": row["created_at"],
    }


# ── init ──────────────────────────────────────────────

async def init(database_url: str, ssl: bool = False) -> None:
    global _pool
    # asyncpg requires postgresql:// scheme
    dsn = database_url.replace("postgres://", "postgresql://", 1)
    ssl_ctx: Any = "require" if ssl else None
    _pool = await asyncpg.create_pool(dsn, ssl=ssl_ctx)
    await _create_tables()


async def _create_tables() -> None:
    async with _pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id           TEXT PRIMARY KEY,
                dealer_id    TEXT,
                user_name    TEXT,
                user_email   TEXT,
                created_at   BIGINT NOT NULL,
                updated_at   BIGINT NOT NULL
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id           BIGSERIAL PRIMARY KEY,
                session_id   TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                role         TEXT NOT NULL CHECK (role IN ('user','assistant','admin','system')),
                content      TEXT NOT NULL,
                citations    TEXT,
                top_score    REAL,
                created_at   BIGINT NOT NULL
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session_created
                ON messages(session_id, created_at)
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS handoffs (
                id           BIGSERIAL PRIMARY KEY,
                session_id   TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                reason       TEXT,
                status       TEXT NOT NULL CHECK (status IN ('open','resolved')),
                created_at   BIGINT NOT NULL,
                resolved_at  BIGINT,
                UNIQUE(session_id, status)
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_handoffs_status_created
                ON handoffs(status, created_at)
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS learned_qa (
                id          TEXT PRIMARY KEY,
                question    TEXT NOT NULL,
                answer      TEXT NOT NULL,
                dealer_id   TEXT,
                created_at  BIGINT NOT NULL,
                updated_at  BIGINT NOT NULL
            )
        """)


# ── sessions ──────────────────────────────────────────

async def create_session(dealer_id: Optional[str] = None) -> str:
    sid = _new_id()
    t = _now()
    await _pool.execute(
        "INSERT INTO sessions (id, dealer_id, created_at, updated_at) VALUES ($1, $2, $3, $4)",
        sid, dealer_id or None, t, t,
    )
    return sid


async def get_or_create_session(
    session_id: Optional[str], dealer_id: Optional[str] = None,
) -> str:
    if session_id:
        row = await _pool.fetchrow("SELECT id FROM sessions WHERE id = $1", session_id)
        if row:
            await _pool.execute(
                "UPDATE sessions SET updated_at = $1 WHERE id = $2",
                _now(), session_id,
            )
            return session_id
    return await create_session(dealer_id)


async def set_session_contact(
    session_id: str, name: Optional[str], email: Optional[str],
) -> None:
    await _pool.execute(
        "UPDATE sessions SET user_name = $1, user_email = $2, updated_at = $3 WHERE id = $4",
        name or None, email or None, _now(), session_id,
    )


# ── messages ──────────────────────────────────────────

async def add_message(
    session_id: str,
    role: str,
    content: str,
    citations: Any = None,
    top_score: Optional[float] = None,
) -> int:
    row = await _pool.fetchrow(
        """INSERT INTO messages (session_id, role, content, citations, top_score, created_at)
           VALUES ($1, $2, $3, $4, $5, $6) RETURNING id""",
        session_id,
        role,
        content,
        json.dumps(citations) if citations is not None else None,
        top_score,
        _now(),
    )
    await _pool.execute(
        "UPDATE sessions SET updated_at = $1 WHERE id = $2",
        _now(), session_id,
    )
    return row["id"]


async def get_messages(session_id: str) -> list[dict]:
    rows = await _pool.fetch(
        "SELECT id, role, content, citations, top_score, created_at "
        "FROM messages WHERE session_id = $1 ORDER BY id ASC",
        session_id,
    )
    return [_parse_row(r) for r in rows]


async def get_messages_after(session_id: str, last_id: int) -> list[dict]:
    rows = await _pool.fetch(
        "SELECT id, role, content, citations, top_score, created_at "
        "FROM messages WHERE session_id = $1 AND id > $2 ORDER BY id ASC",
        session_id, last_id or 0,
    )
    return [_parse_row(r) for r in rows]


async def get_history_pairs(session_id: str, limit_pairs: int) -> list[dict]:
    if not limit_pairs or limit_pairs <= 0:
        return []
    rows = await _pool.fetch(
        "SELECT role, content FROM messages "
        "WHERE session_id = $1 AND role IN ('user', 'assistant') "
        "ORDER BY id DESC LIMIT $2",
        session_id, limit_pairs * 2,
    )
    rows = list(reversed(rows))
    pairs: list[dict] = []
    pending_user: Optional[str] = None
    for row in rows:
        if row["role"] == "user":
            pending_user = row["content"]
        elif row["role"] == "assistant" and pending_user:
            pairs.append({"question": pending_user, "answer": row["content"]})
            pending_user = None
    return pairs[-limit_pairs:]


async def get_last_user_question(session_id: str) -> Optional[str]:
    row = await _pool.fetchrow(
        "SELECT content FROM messages "
        "WHERE session_id = $1 AND role = 'user' ORDER BY id DESC LIMIT 1",
        session_id,
    )
    return row["content"] if row else None


# ── handoffs ──────────────────────────────────────────

async def open_handoff(
    session_id: str, reason: str = "user_requested",
) -> Optional[dict]:
    await _pool.execute(
        """INSERT INTO handoffs (session_id, reason, status, created_at)
           VALUES ($1, $2, 'open', $3)
           ON CONFLICT(session_id, status) DO NOTHING""",
        session_id, reason, _now(),
    )
    row = await _pool.fetchrow(
        "SELECT * FROM handoffs WHERE session_id = $1 AND status = 'open'",
        session_id,
    )
    return dict(row) if row else None


async def resolve_handoff(session_id: str) -> None:
    await _pool.execute(
        "UPDATE handoffs SET status = 'resolved', resolved_at = $1 "
        "WHERE session_id = $2 AND status = 'open'",
        _now(), session_id,
    )


async def list_open_handoffs() -> list[dict]:
    rows = await _pool.fetch("""
        SELECT h.id, h.session_id, h.reason, h.created_at,
               s.dealer_id, s.user_name, s.user_email,
               (SELECT content FROM messages
                WHERE session_id = h.session_id AND role = 'user'
                ORDER BY id DESC LIMIT 1) AS last_question
        FROM handoffs h
        JOIN sessions s ON s.id = h.session_id
        WHERE h.status = 'open'
        ORDER BY h.created_at ASC
    """)
    return [dict(r) for r in rows]


async def has_open_handoff(session_id: str) -> bool:
    row = await _pool.fetchrow(
        "SELECT id FROM handoffs WHERE session_id = $1 AND status = 'open'",
        session_id,
    )
    return row is not None


# ── learned Q&A ───────────────────────────────────────

async def add_learned_qa(
    item_id: str,
    question: str,
    answer: str,
    dealer_id: Optional[str] = None,
) -> None:
    t = _now()
    await _pool.execute(
        """INSERT INTO learned_qa (id, question, answer, dealer_id, created_at, updated_at)
           VALUES ($1, $2, $3, $4, $5, $5)
           ON CONFLICT (id) DO UPDATE SET question=$2, answer=$3, updated_at=$5""",
        item_id, question, answer, dealer_id or None, t,
    )


async def list_learned_qa() -> list[dict]:
    rows = await _pool.fetch(
        "SELECT id, question, answer, dealer_id, created_at, updated_at "
        "FROM learned_qa ORDER BY updated_at DESC"
    )
    return [dict(r) for r in rows]


async def update_learned_qa(item_id: str, question: str, answer: str) -> None:
    await _pool.execute(
        "UPDATE learned_qa SET question=$2, answer=$3, updated_at=$4 WHERE id=$1",
        item_id, question, answer, _now(),
    )


async def delete_learned_qa(item_id: str) -> None:
    await _pool.execute("DELETE FROM learned_qa WHERE id=$1", item_id)


# ── analytics ─────────────────────────────────────────

async def get_analytics(days: int = 7) -> dict:
    since = _now() - days * 86_400_000

    # Static SQL fragment — not user input, so f-string interpolation is safe.
    no_ans = """(
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
    )"""

    ov = await _pool.fetchrow(f"""
        SELECT
            (SELECT COUNT(DISTINCT id) FROM sessions) AS total_sessions,
            COUNT(*) FILTER (WHERE role = 'user') AS total_questions,
            COUNT(*) FILTER (WHERE role = 'assistant' AND NOT {no_ans}) AS answered,
            COUNT(*) FILTER (WHERE role = 'assistant' AND {no_ans}) AS unanswered
        FROM messages
    """)

    hf = await _pool.fetchrow(f"""
        SELECT
            COUNT(*) FILTER (WHERE status = 'open')           AS open_count,
            COUNT(*) FILTER (WHERE status = 'resolved')       AS resolved_count,
            COUNT(*) FILTER (WHERE reason = 'low_confidence') AS low_confidence,
            COUNT(*) FILTER (WHERE reason = 'user_requested') AS user_requested
        FROM handoffs
    """)

    daily = await _pool.fetch(f"""
        SELECT
            TO_CHAR(TO_TIMESTAMP(created_at / 1000.0), 'YYYY-MM-DD') AS day,
            COUNT(*) FILTER (WHERE role = 'user')                        AS questions,
            COUNT(*) FILTER (WHERE role = 'assistant' AND NOT {no_ans})  AS answered,
            COUNT(*) FILTER (WHERE role = 'assistant' AND {no_ans})      AS unanswered
        FROM messages
        WHERE created_at >= $1
        GROUP BY day
        ORDER BY day ASC
    """, since)

    unans = await _pool.fetch(f"""
        WITH no_ans AS (
            SELECT id, session_id, created_at
            FROM messages
            WHERE role = 'assistant' AND {no_ans}
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
    """)

    return {
        "overview":   dict(ov),
        "handoffs":   dict(hf),
        "daily":      [dict(r) for r in daily],
        "unanswered": [dict(r) for r in unans],
    }
