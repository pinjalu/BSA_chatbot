import React, { useState, useEffect, useCallback } from 'react';
import Header from '../components/Header.jsx';
import { useAuth } from '../hooks/useAuth.jsx';
import './Content.css';

function QARow({ item, onEdit, onDelete }) {
  const [editing, setEditing] = useState(false);
  const [q, setQ] = useState(item.question);
  const [a, setA] = useState(item.answer);
  const [saving, setSaving] = useState(false);

  function handleCancel() {
    setQ(item.question);
    setA(item.answer);
    setEditing(false);
  }

  async function handleSave() {
    if (!q.trim() || !a.trim()) return;
    setSaving(true);
    try {
      await onEdit(item.id, q.trim(), a.trim());
      setEditing(false);
    } finally {
      setSaving(false);
    }
  }

  if (editing) {
    return (
      <div className="qa-row qa-row--editing">
        <div className="form-group" style={{ marginBottom: 8 }}>
          <label>Question</label>
          <input value={q} onChange={e => setQ(e.target.value)} />
        </div>
        <div className="form-group" style={{ marginBottom: 10 }}>
          <label>Answer</label>
          <textarea value={a} onChange={e => setA(e.target.value)} rows={4} />
        </div>
        <div className="qa-row__actions">
          <button className="btn btn-primary btn-sm" onClick={handleSave} disabled={saving || !q.trim() || !a.trim()}>
            {saving ? 'Saving…' : 'Save'}
          </button>
          <button className="btn btn-secondary btn-sm" onClick={handleCancel} disabled={saving}>
            Cancel
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="qa-row">
      <div className="qa-row__q">{item.question}</div>
      <div className="qa-row__a">{item.answer}</div>
      <div className="qa-row__actions">
        {item.dealer_id && (
          <span className="badge badge-gray" style={{ marginRight: 'auto' }}>
            {item.dealer_id}
          </span>
        )}
        <button className="btn btn-secondary btn-sm" onClick={() => setEditing(true)}>Edit</button>
        <button className="btn btn-danger btn-sm" onClick={() => onDelete(item.id)}>Delete</button>
      </div>
    </div>
  );
}

export default function Content() {
  const { authHeader } = useAuth();
  const [items, setItems] = useState([]);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [showAdd, setShowAdd] = useState(false);
  const [newQ, setNewQ] = useState('');
  const [newA, setNewA] = useState('');
  const [adding, setAdding] = useState(false);

  const fetchItems = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const res = await fetch('/api/admin/content', { headers: authHeader });
      if (!res.ok) throw new Error(res.statusText);
      const data = await res.json();
      setItems(data.items || []);
    } catch (e) {
      setError('Failed to load content: ' + e.message);
    } finally {
      setLoading(false);
    }
  }, [authHeader]);

  useEffect(() => { fetchItems(); }, [fetchItems]);

  async function handleAdd(e) {
    e.preventDefault();
    if (!newQ.trim() || !newA.trim()) return;
    setAdding(true);
    try {
      const res = await fetch('/api/admin/content', {
        method: 'POST',
        headers: { ...authHeader, 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: newQ.trim(), answer: newA.trim() }),
      });
      if (!res.ok) {
        const d = await res.json();
        throw new Error(d.error || res.statusText);
      }
      setNewQ('');
      setNewA('');
      setShowAdd(false);
      await fetchItems();
    } catch (e) {
      setError('Failed to add: ' + e.message);
    } finally {
      setAdding(false);
    }
  }

  async function handleEdit(id, question, answer) {
    const res = await fetch(`/api/admin/content/${id}`, {
      method: 'PUT',
      headers: { ...authHeader, 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, answer }),
    });
    if (!res.ok) {
      const d = await res.json().catch(() => ({}));
      throw new Error(d.error || res.statusText);
    }
    await fetchItems();
  }

  async function handleDelete(id) {
    if (!confirm('Delete this Q&A entry?')) return;
    try {
      const res = await fetch(`/api/admin/content/${id}`, {
        method: 'DELETE',
        headers: authHeader,
      });
      if (!res.ok) throw new Error(res.statusText);
      await fetchItems();
    } catch (e) {
      setError('Failed to delete: ' + e.message);
    }
  }

  const filtered = items.filter(item =>
    !search || item.question.toLowerCase().includes(search.toLowerCase()) ||
    item.answer.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="page-layout">
      <Header />
      <div className="page-content">
        <div className="content-header">
          <h1 className="page-title">Q&A Content</h1>
          <button
            className="btn btn-primary"
            onClick={() => setShowAdd(v => !v)}
          >
            {showAdd ? '✕ Cancel' : '+ Add Q&A'}
          </button>
        </div>

        {showAdd && (
          <div className="card add-form">
            <div className="add-form__title">New Q&A Entry</div>
            <form onSubmit={handleAdd}>
              <div className="form-group" style={{ marginBottom: 12 }}>
                <label>Question</label>
                <input
                  value={newQ}
                  onChange={e => setNewQ(e.target.value)}
                  placeholder="Enter the question…"
                  autoFocus
                />
              </div>
              <div className="form-group" style={{ marginBottom: 14 }}>
                <label>Answer</label>
                <textarea
                  value={newA}
                  onChange={e => setNewA(e.target.value)}
                  placeholder="Enter the answer…"
                  rows={5}
                />
              </div>
              {error && <div className="error-msg" style={{ marginBottom: 10 }}>{error}</div>}
              <div style={{ display: 'flex', gap: 8 }}>
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={adding || !newQ.trim() || !newA.trim()}
                >
                  {adding ? 'Adding…' : 'Add Entry'}
                </button>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => { setShowAdd(false); setNewQ(''); setNewA(''); setError(''); }}
                >
                  Cancel
                </button>
              </div>
            </form>
          </div>
        )}

        <div className="content-search-row">
          <input
            className="content-search"
            placeholder="Search Q&As…"
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
          <span className="content-count">{filtered.length} entries</span>
        </div>

        {error && !showAdd && <div className="error-msg" style={{ marginBottom: 14 }}>{error}</div>}

        {loading ? (
          <div className="analytics-loading">Loading<span className="loading-dots" /></div>
        ) : filtered.length === 0 ? (
          <div className="empty-state card" style={{ padding: '50px 20px' }}>
            <p>{search ? 'No results for that search.' : 'No Q&A entries yet. Add one above.'}</p>
          </div>
        ) : (
          <div className="qa-list">
            {filtered.map(item => (
              <QARow
                key={item.id}
                item={item}
                onEdit={handleEdit}
                onDelete={handleDelete}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
