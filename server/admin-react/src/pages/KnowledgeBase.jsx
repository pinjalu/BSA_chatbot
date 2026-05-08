import React, { useState, useRef, useCallback } from 'react';
import './KnowledgeBase.css';

const VEHICLES = [
  { value: '', label: 'All Models' },
  { value: 'Gold Star', label: 'Gold Star' },
  { value: 'Bantam', label: 'Bantam' },
  { value: 'Scrambler', label: 'Scrambler' },
];

const PROGRESS_STEPS = [
  { delay: 0, text: 'Looking at your question…' },
  { delay: 1200, text: 'Searching the manuals…' },
  { delay: 2800, text: 'Finding relevant sections…' },
  { delay: 4500, text: 'Reviewing technical details…' },
  { delay: 6500, text: 'Compiling your answer…' },
];

function Citation({ cite, index }) {
  return (
    <span className="citation" title={`${cite.source || cite.document || ''} p.${cite.page || '?'}`}>
      [{index + 1}]
    </span>
  );
}

function ContactForm({ question }) {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [sent, setSent] = useState(false);
  const [sending, setSending] = useState(false);
  const [err, setErr] = useState('');

  async function handleSubmit(e) {
    e.preventDefault();
    setErr('');
    setSending(true);
    try {
      const res = await fetch('/api/kb/handoff', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, email, question }),
      });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d.error || res.statusText);
      }
      setSent(true);
    } catch (e) {
      setErr('Failed to send: ' + e.message);
    } finally {
      setSending(false);
    }
  }

  if (sent) {
    return (
      <div className="contact-form contact-form--sent">
        <span className="contact-sent-icon">✓</span>
        <div>
          <strong>Message sent!</strong>
          <p>Our team will get back to you at {email}.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="contact-form">
      <div className="contact-form__title">
        Can't find what you need? Our team can help.
      </div>
      <form onSubmit={handleSubmit}>
        <div className="contact-form__row">
          <div className="form-group">
            <label>Name</label>
            <input
              type="text"
              value={name}
              onChange={e => setName(e.target.value)}
              placeholder="Your name"
            />
          </div>
          <div className="form-group">
            <label>Email *</label>
            <input
              type="email"
              value={email}
              onChange={e => setEmail(e.target.value)}
              placeholder="your@email.com"
              required
            />
          </div>
        </div>
        {err && <div className="error-msg">{err}</div>}
        <button
          type="submit"
          className="btn btn-primary"
          disabled={sending || !email}
        >
          {sending ? 'Sending…' : 'Contact a Specialist'}
        </button>
      </form>
    </div>
  );
}

export default function KnowledgeBase() {
  const [query, setQuery] = useState('');
  const [vehicle, setVehicle] = useState('');
  const [searching, setSearching] = useState(false);
  const [progressStep, setProgressStep] = useState(0);
  const [answer, setAnswer] = useState('');
  const [citations, setCitations] = useState([]);
  const [responseTime, setResponseTime] = useState(null);
  const [error, setError] = useState('');
  const [showContact, setShowContact] = useState(false);
  const [hasResult, setHasResult] = useState(false);
  const abortRef = useRef(null);
  const timersRef = useRef([]);
  const startTimeRef = useRef(null);

  const NO_ANSWER_RX = /couldn'?t find|could not find|don'?t have|do not have|not in (?:the )?manuals?|no (?:relevant )?information|outside (?:the )?scope|i'?m not finding|i can'?t find/i;

  function clearTimers() {
    timersRef.current.forEach(t => clearTimeout(t));
    timersRef.current = [];
  }

  const doSearch = useCallback(async () => {
    if (!query.trim() || searching) return;

    // Reset state
    setSearching(true);
    setAnswer('');
    setCitations([]);
    setResponseTime(null);
    setError('');
    setShowContact(false);
    setHasResult(false);
    setProgressStep(0);
    clearTimers();
    startTimeRef.current = Date.now();

    if (abortRef.current) abortRef.current.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    // Start progress step timers
    PROGRESS_STEPS.forEach((step, i) => {
      const t = setTimeout(() => setProgressStep(i), step.delay);
      timersRef.current.push(t);
    });

    let accumulated = '';

    try {
      const res = await fetch('/api/kb/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query.trim(), vehicle: vehicle || null }),
        signal: controller.signal,
      });

      if (!res.ok) {
        throw new Error(`Server error ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });

        // Parse SSE events
        let boundary;
        while ((boundary = buf.indexOf('\n\n')) !== -1) {
          const block = buf.slice(0, boundary);
          buf = buf.slice(boundary + 2);

          let eventType = 'message';
          let dataStr = '';
          for (const line of block.split('\n')) {
            if (line.startsWith('event:')) eventType = line.slice(6).trim();
            else if (line.startsWith('data:')) dataStr += line.slice(5).trim();
          }
          if (!dataStr) continue;

          let eventData;
          try { eventData = JSON.parse(dataStr); } catch { continue; }

          if (eventType === 'token') {
            accumulated += eventData.text || '';
            setAnswer(accumulated);
          } else if (eventType === 'meta') {
            setCitations(eventData.citations || []);
          } else if (eventType === 'done') {
            const elapsed = Date.now() - startTimeRef.current;
            setResponseTime(eventData.timings?.total || elapsed);
          } else if (eventType === 'error') {
            throw new Error(eventData.message || 'Search error');
          }
        }
      }

      clearTimers();
      setSearching(false);
      setHasResult(true);

      // Check if the answer says it couldn't find anything
      if (NO_ANSWER_RX.test(accumulated)) {
        setShowContact(true);
      }
    } catch (e) {
      if (e.name === 'AbortError') return;
      clearTimers();
      setSearching(false);
      setError(e.message || 'Search failed');
    }
  }, [query, vehicle, searching]);

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      doSearch();
    }
  }

  return (
    <div className="kb-page">
      {/* Hero */}
      <div className="kb-hero">
        <div className="kb-hero__inner">
          <div className="kb-hero__brand">BSA</div>
          <h1 className="kb-hero__title">Motorcycle Knowledge Base</h1>
          <p className="kb-hero__sub">Search technical manuals, workshop guides, and parts catalogues</p>

          <div className="kb-search-bar">
            <div className="kb-vehicle-select">
              <select
                value={vehicle}
                onChange={e => setVehicle(e.target.value)}
              >
                {VEHICLES.map(v => (
                  <option key={v.value} value={v.value}>{v.label}</option>
                ))}
              </select>
            </div>
            <input
              className="kb-search-input"
              type="text"
              placeholder="Ask anything about your BSA motorcycle…"
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={searching}
              autoFocus
            />
            <button
              className="kb-search-btn"
              onClick={doSearch}
              disabled={searching || !query.trim()}
            >
              {searching ? (
                <span className="kb-spinner" />
              ) : (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <circle cx="11" cy="11" r="8" />
                  <path d="m21 21-4.35-4.35" />
                </svg>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Results area */}
      <div className="kb-results">
        {error && (
          <div className="error-msg kb-error">{error}</div>
        )}

        {searching && (
          <div className="kb-progress">
            <div className="kb-progress__steps">
              {PROGRESS_STEPS.map((step, i) => (
                <div
                  key={i}
                  className={`kb-progress__step ${i <= progressStep ? 'kb-progress__step--active' : ''} ${i === progressStep ? 'kb-progress__step--current' : ''}`}
                >
                  <span className="kb-progress__dot" />
                  <span className="kb-progress__text">{step.text}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {(answer || searching) && (
          <div className="kb-answer-card">
            <div className="kb-answer-header">
              <span className="kb-answer-label">Answer</span>
              {responseTime && (
                <span className="kb-response-time">{(responseTime / 1000).toFixed(1)}s</span>
              )}
            </div>
            <div className="kb-answer-text">
              {answer}
              {searching && !answer && <span className="loading-dots" />}
              {citations.length > 0 && answer && (
                <span className="citations-inline">
                  {citations.slice(0, 5).map((c, i) => (
                    <Citation key={i} cite={c} index={i} />
                  ))}
                </span>
              )}
            </div>

            {citations.length > 0 && (
              <div className="kb-citations">
                <div className="kb-citations__title">Sources</div>
                {citations.slice(0, 5).map((c, i) => (
                  <div key={i} className="kb-citation-row">
                    <span className="kb-citation-num">[{i + 1}]</span>
                    <span className="kb-citation-text">
                      {c.source || c.document || c.file || 'Manual'}
                      {c.page ? `, page ${c.page}` : ''}
                      {c.score ? ` · ${(c.score * 100).toFixed(0)}% match` : ''}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {showContact && !searching && (
          <ContactForm question={query} />
        )}

        {!searching && !hasResult && !error && (
          <div className="kb-suggestions">
            <div className="kb-suggestions__title">Popular questions</div>
            <div className="kb-suggestions__list">
              {[
                'What oil should I use in my Gold Star 650?',
                'How do I adjust the valve clearances?',
                'What are the torque specs for cylinder head bolts?',
                'How do I check the chain tension?',
                'What spark plug does the Bantam use?',
              ].map(q => (
                <button
                  key={q}
                  className="kb-suggestion"
                  onClick={() => { setQuery(q); }}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
