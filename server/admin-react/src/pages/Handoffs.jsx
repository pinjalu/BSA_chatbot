import React, { useState, useEffect, useRef, useCallback } from 'react';
import Header from '../components/Header.jsx';
import { useAuth } from '../hooks/useAuth.jsx';
import './Handoffs.css';

function formatTime(ts) {
  if (!ts) return '';
  const d = new Date(ts);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function formatDate(ts) {
  if (!ts) return '';
  const d = new Date(ts);
  const now = new Date();
  if (d.toDateString() === now.toDateString()) return formatTime(ts);
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ' ' + formatTime(ts);
}

function ReasonBadge({ reason }) {
  const map = {
    user_requested: ['blue', 'User Request'],
    low_confidence: ['orange', 'Low Confidence'],
    kb_search: ['gray', 'KB Search'],
  };
  const [color, label] = map[reason] || ['gray', reason || 'Unknown'];
  return <span className={`badge badge-${color}`}>{label}</span>;
}

function MessageBubble({ msg }) {
  // db.js parseRow returns camelCase (createdAt, content), but support both
  const content = msg.content;
  const createdAt = msg.createdAt || msg.created_at;
  const roleClass = { user: 'bubble-user', admin: 'bubble-admin', assistant: 'bubble-bot', system: 'bubble-system' }[msg.role] || 'bubble-system';
  const roleLabel = { user: 'Visitor', admin: 'Admin', assistant: 'Bot', system: 'System' }[msg.role] || msg.role;

  if (msg.role === 'system') {
    return (
      <div className="bubble-system-wrap">
        <span className="bubble-system-text">{content}</span>
      </div>
    );
  }

  return (
    <div className={`bubble-wrap bubble-wrap--${msg.role}`}>
      <div className={`bubble ${roleClass}`}>
        <div className="bubble-meta">
          <span className="bubble-role">{roleLabel}</span>
          <span className="bubble-time">{formatDate(createdAt)}</span>
        </div>
        <div className="bubble-content">{content}</div>
      </div>
    </div>
  );
}

export default function Handoffs() {
  const { authHeader, token } = useAuth();
  const [handoffs, setHandoffs] = useState([]);
  const [selected, setSelected] = useState(null);
  const [messages, setMessages] = useState([]);
  const [reply, setReply] = useState('');
  const [resolveOnReply, setResolveOnReply] = useState(false);
  const [sending, setSending] = useState(false);
  const [loadingSession, setLoadingSession] = useState(false);
  const transcriptRef = useRef(null);
  const pollRef = useRef(null);
  const selectedRef = useRef(null);

  selectedRef.current = selected;

  const fetchHandoffs = useCallback(async () => {
    try {
      const res = await fetch('/api/admin/handoffs', { headers: authHeader });
      if (!res.ok) return;
      const data = await res.json();
      setHandoffs(data.handoffs || []);
    } catch {}
  }, [authHeader]);

  const fetchSession = useCallback(async (sessionId, silent = false) => {
    if (!silent) setLoadingSession(true);
    try {
      const res = await fetch(`/api/admin/session/${sessionId}`, { headers: authHeader });
      if (!res.ok) return;
      const data = await res.json();
      setMessages(data.messages || []);
    } catch {} finally {
      if (!silent) setLoadingSession(false);
    }
  }, [authHeader]);

  // Polling: every 5s fetch handoffs + current session
  useEffect(() => {
    fetchHandoffs();
    pollRef.current = setInterval(() => {
      fetchHandoffs();
      if (selectedRef.current) {
        fetchSession(selectedRef.current, true);
      }
    }, 5000);
    return () => clearInterval(pollRef.current);
  }, [fetchHandoffs, fetchSession]);

  // Scroll to bottom when messages change
  useEffect(() => {
    if (transcriptRef.current) {
      transcriptRef.current.scrollTop = transcriptRef.current.scrollHeight;
    }
  }, [messages]);

  async function selectHandoff(h) {
    setSelected(h.session_id);
    setMessages([]);
    setReply('');
    await fetchSession(h.session_id);
  }

  async function handleReply(e) {
    e.preventDefault();
    if (!reply.trim() || !selected) return;
    setSending(true);
    try {
      const res = await fetch('/api/admin/reply', {
        method: 'POST',
        headers: { ...authHeader, 'Content-Type': 'application/json' },
        body: JSON.stringify({ sessionId: selected, message: reply.trim(), resolve: resolveOnReply }),
      });
      if (!res.ok) throw new Error('Failed');
      setReply('');
      if (resolveOnReply) {
        setSelected(null);
        setMessages([]);
        await fetchHandoffs();
      } else {
        await fetchSession(selected);
      }
    } catch (err) {
      alert('Failed to send reply: ' + err.message);
    } finally {
      setSending(false);
    }
  }

  async function handleResolve() {
    if (!selected) return;
    if (!confirm('Resolve this handoff without replying?')) return;
    try {
      await fetch('/api/admin/resolve', {
        method: 'POST',
        headers: { ...authHeader, 'Content-Type': 'application/json' },
        body: JSON.stringify({ sessionId: selected }),
      });
      setSelected(null);
      setMessages([]);
      await fetchHandoffs();
    } catch (err) {
      alert('Failed: ' + err.message);
    }
  }

  const selectedHandoff = handoffs.find(h => h.session_id === selected);

  return (
    <div className="page-layout">
      <Header />
      <div className="page-content" style={{ paddingBottom: 0 }}>
        <div className="split-layout handoffs-layout">
          {/* Left: handoff list */}
          <div className="panel">
            <div className="panel-header">
              Open Handoffs
              <span className="badge badge-red">{handoffs.length}</span>
            </div>
            <div className="panel-body">
              {handoffs.length === 0 ? (
                <div className="empty-state">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <p>No open handoffs</p>
                </div>
              ) : (
                handoffs.map(h => (
                  <div
                    key={h.session_id}
                    className={`handoff-item ${selected === h.session_id ? 'handoff-item--active' : ''}`}
                    onClick={() => selectHandoff(h)}
                  >
                    <div className="handoff-item__top">
                      <span className="handoff-item__name">{h.user_name || h.name || 'Anonymous'}</span>
                      <span className="handoff-item__time">{formatDate(h.created_at)}</span>
                    </div>
                    {(h.user_email || h.email) && <div className="handoff-item__email">{h.user_email || h.email}</div>}
                    <div className="handoff-item__question">{h.last_question || '—'}</div>
                    <div className="handoff-item__reason">
                      <ReasonBadge reason={h.reason} />
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Right: transcript + reply */}
          <div className="panel transcript-panel">
            {!selected ? (
              <div className="empty-state" style={{ flex: 1 }}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
                <p>Select a handoff to view transcript</p>
              </div>
            ) : (
              <>
                <div className="panel-header">
                  <div>
                    <span>{selectedHandoff?.user_name || selectedHandoff?.name || 'Anonymous'}</span>
                    {(selectedHandoff?.user_email || selectedHandoff?.email) && (
                      <span className="transcript-email"> — {selectedHandoff.user_email || selectedHandoff.email}</span>
                    )}
                  </div>
                  <button className="btn btn-sm btn-secondary" onClick={handleResolve}>
                    Resolve
                  </button>
                </div>
                <div className="transcript-body" ref={transcriptRef}>
                  {loadingSession ? (
                    <div className="empty-state">
                      <p className="loading-dots">Loading</p>
                    </div>
                  ) : messages.length === 0 ? (
                    <div className="empty-state"><p>No messages</p></div>
                  ) : (
                    messages.map(m => <MessageBubble key={m.id} msg={m} />)
                  )}
                </div>
                <div className="reply-area">
                  <form onSubmit={handleReply}>
                    <textarea
                      className="reply-input"
                      placeholder="Type your reply…"
                      value={reply}
                      onChange={e => setReply(e.target.value)}
                      rows={3}
                      onKeyDown={e => {
                        if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) handleReply(e);
                      }}
                    />
                    <div className="reply-actions">
                      <label className="reply-resolve-check">
                        <input
                          type="checkbox"
                          checked={resolveOnReply}
                          onChange={e => setResolveOnReply(e.target.checked)}
                        />
                        Resolve after reply
                      </label>
                      <button
                        type="submit"
                        className="btn btn-primary"
                        disabled={sending || !reply.trim()}
                      >
                        {sending ? 'Sending…' : 'Send Reply'}
                      </button>
                    </div>
                  </form>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
