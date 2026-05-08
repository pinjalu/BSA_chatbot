import React, { useState, useEffect, useRef, useCallback } from 'react';
import Header from '../components/Header.jsx';
import { useAuth } from '../hooks/useAuth.jsx';
import './Pipeline.css';

const CATEGORIES = [
  { value: 'workshop_manual', label: 'Workshop Manual' },
  { value: 'owners_manual', label: "Owner's Manual" },
  { value: 'parts_catalogue', label: 'Parts Catalogue' },
  { value: 'accessories_catalog', label: 'Accessories Catalog' },
  { value: 'spec_sheet', label: 'Spec Sheet' },
  { value: 'wiring_diagram', label: 'Wiring Diagram' },
  { value: 'sop', label: 'SOP' },
];

const STAGE_LABELS = {
  extract: 'Extract Text',
  clean: 'Clean Text',
  chunk: 'Chunk',
  embed: 'Embed',
  upsert: 'Upsert Vectors',
  index: 'Index',
  verify: 'Verify',
  done: 'Complete',
};

function statusBadge(status) {
  const map = {
    pending: 'gray',
    running: 'blue',
    done: 'green',
    error: 'red',
    cancelled: 'orange',
  };
  return <span className={`badge badge-${map[status] || 'gray'}`}>{status}</span>;
}

function formatDuration(ms) {
  if (!ms && ms !== 0) return '—';
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  const m = Math.floor(ms / 60000);
  const s = Math.round((ms % 60000) / 1000);
  return `${m}m ${s}s`;
}

function formatAgo(ts) {
  if (!ts) return '—';
  const d = new Date(ts);
  const diff = Date.now() - d.getTime();
  if (diff < 60000) return 'just now';
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

function StageDot({ stage }) {
  const statusMap = { pending: 'dot-pending', running: 'dot-running', done: 'dot-done', error: 'dot-error', skipped: 'dot-skipped' };
  return (
    <div className={`stage-row ${stage.status === 'running' ? 'stage-row--running' : ''}`}>
      <span className={`stage-dot ${statusMap[stage.status] || 'dot-pending'}`} />
      <span className="stage-name">{STAGE_LABELS[stage.name] || stage.name}</span>
      <span className="stage-time">
        {stage.elapsed_ms != null ? formatDuration(stage.elapsed_ms) : ''}
      </span>
    </div>
  );
}

export default function Pipeline() {
  const { authHeader, token } = useAuth();
  const [jobs, setJobs] = useState([]);
  const [selectedJob, setSelectedJob] = useState(null);
  const [jobDetail, setJobDetail] = useState(null);
  const [logLines, setLogLines] = useState([]);
  const [dragging, setDragging] = useState(false);
  const [uploadFile, setUploadFile] = useState(null);
  const [category, setCategory] = useState('workshop_manual');
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState('');
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [retrying, setRetrying] = useState(false);
  const logRef = useRef(null);
  const sseRef = useRef(null);
  const pollRef = useRef(null);
  const fileInputRef = useRef(null);

  const fetchJobs = useCallback(async () => {
    try {
      const res = await fetch('/api/admin/pipeline/jobs', { headers: authHeader });
      if (!res.ok) return;
      const data = await res.json();
      setJobs(Array.isArray(data) ? data : (data.jobs || []));
    } catch {}
  }, [authHeader]);

  useEffect(() => {
    fetchJobs();
    pollRef.current = setInterval(fetchJobs, 6000);
    return () => clearInterval(pollRef.current);
  }, [fetchJobs]);

  // Auto-scroll log
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logLines]);

  const fetchJobDetail = useCallback(async (jobId) => {
    setLoadingDetail(true);
    try {
      const res = await fetch(`/api/admin/pipeline/jobs/${jobId}`, { headers: authHeader });
      if (!res.ok) return;
      const data = await res.json();
      setJobDetail(data);
    } catch {} finally {
      setLoadingDetail(false);
    }
  }, [authHeader]);

  function startSSE(jobId) {
    if (sseRef.current) { sseRef.current.close(); sseRef.current = null; }
    setLogLines([]);
    const es = new EventSource(`/api/admin/pipeline/jobs/${jobId}/stream?token=${encodeURIComponent(token)}`);
    sseRef.current = es;

    es.addEventListener('log', e => {
      try {
        const d = JSON.parse(e.data);
        setLogLines(prev => [...prev, d.message || d.text || e.data]);
      } catch {
        setLogLines(prev => [...prev, e.data]);
      }
    });
    es.addEventListener('status', e => {
      try {
        const d = JSON.parse(e.data);
        setJobDetail(prev => prev ? { ...prev, ...d } : d);
        fetchJobDetail(jobId);
      } catch {}
    });
    es.addEventListener('done', e => {
      es.close();
      sseRef.current = null;
      fetchJobDetail(jobId);
      fetchJobs();
    });
    es.addEventListener('error', e => {
      if (es.readyState === EventSource.CLOSED) {
        sseRef.current = null;
      }
    });
  }

  async function selectJob(job) {
    if (sseRef.current) { sseRef.current.close(); sseRef.current = null; }
    setSelectedJob(job.id);
    setLogLines([]);
    setJobDetail(null);
    await fetchJobDetail(job.id);
    if (job.status === 'running' || job.status === 'pending') {
      startSSE(job.id);
    }
  }

  // Cleanup SSE on unmount
  useEffect(() => {
    return () => { if (sseRef.current) sseRef.current.close(); };
  }, []);

  function handleDrop(e) {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f && f.name.toLowerCase().endsWith('.pdf')) {
      setUploadFile(f);
      setUploadError('');
    } else {
      setUploadError('Only PDF files are accepted.');
    }
  }

  async function handleUpload() {
    if (!uploadFile) return;
    setUploading(true);
    setUploadError('');
    try {
      const fd = new FormData();
      fd.append('file', uploadFile);
      fd.append('category', category);
      const res = await fetch('/api/admin/pipeline/upload', {
        method: 'POST',
        headers: authHeader,
        body: fd,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || res.statusText);
      setUploadFile(null);
      await fetchJobs();
      // Auto-select new job
      if (data.job_id || data.id) {
        const jobId = data.job_id || data.id;
        setSelectedJob(jobId);
        setLogLines([]);
        setJobDetail(null);
        await fetchJobDetail(jobId);
        startSSE(jobId);
      }
    } catch (e) {
      setUploadError('Upload failed: ' + e.message);
    } finally {
      setUploading(false);
    }
  }

  async function handleDelete(jobId, e) {
    e.stopPropagation();
    if (!confirm('Delete this job?')) return;
    try {
      await fetch(`/api/admin/pipeline/jobs/${jobId}`, {
        method: 'DELETE',
        headers: authHeader,
      });
      if (selectedJob === jobId) {
        setSelectedJob(null);
        setJobDetail(null);
        setLogLines([]);
        if (sseRef.current) { sseRef.current.close(); sseRef.current = null; }
      }
      await fetchJobs();
    } catch (e) {
      alert('Delete failed: ' + e.message);
    }
  }

  async function handleCancel(jobId) {
    try {
      await fetch(`/api/admin/pipeline/jobs/${jobId}/cancel`, {
        method: 'POST',
        headers: authHeader,
      });
      await fetchJobs();
      await fetchJobDetail(jobId);
    } catch (e) {
      alert('Cancel failed: ' + e.message);
    }
  }

  async function handleRetry(jobId, skipStage) {
    setRetrying(true);
    try {
      const fd = new FormData();
      if (skipStage) fd.append('skip', skipStage);
      const res = await fetch(`/api/admin/pipeline/jobs/${jobId}/retry`, {
        method: 'POST',
        headers: authHeader,
        body: fd,
      });
      if (!res.ok) throw new Error(res.statusText);
      setLogLines([]);
      await fetchJobDetail(jobId);
      startSSE(jobId);
      await fetchJobs();
    } catch (e) {
      alert('Retry failed: ' + e.message);
    } finally {
      setRetrying(false);
    }
  }

  const currentJob = jobs.find(j => j.id === selectedJob);
  const stages = jobDetail?.stages || jobDetail?.events || [];
  const errorStage = stages.find(s => s.status === 'error');

  return (
    <div className="page-layout">
      <Header />
      <div className="page-content pipeline-content">
        <div className="pipeline-layout">
          {/* Left column */}
          <div className="pipeline-left">
            {/* Upload dropzone */}
            <div className="card upload-card">
              <div className="upload-card__title">Upload PDF</div>
              <div
                className={`dropzone ${dragging ? 'dropzone--active' : ''} ${uploadFile ? 'dropzone--has-file' : ''}`}
                onDragOver={e => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={handleDrop}
                onClick={() => !uploadFile && fileInputRef.current?.click()}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf"
                  style={{ display: 'none' }}
                  onChange={e => {
                    const f = e.target.files[0];
                    if (f) { setUploadFile(f); setUploadError(''); }
                  }}
                />
                {uploadFile ? (
                  <div className="dropzone__file">
                    <span className="dropzone__file-icon">PDF</span>
                    <span className="dropzone__file-name">{uploadFile.name}</span>
                    <span className="dropzone__file-size">
                      {(uploadFile.size / 1024 / 1024).toFixed(1)} MB
                    </span>
                    <button
                      className="btn btn-secondary btn-sm"
                      onClick={e => { e.stopPropagation(); setUploadFile(null); }}
                    >
                      Remove
                    </button>
                  </div>
                ) : (
                  <div className="dropzone__empty">
                    <div className="dropzone__icon">⬆</div>
                    <div>Drop a PDF here or click to browse</div>
                  </div>
                )}
              </div>

              {uploadFile && (
                <div className="upload-controls">
                  <div className="form-group">
                    <label>Category</label>
                    <select value={category} onChange={e => setCategory(e.target.value)}>
                      {CATEGORIES.map(c => (
                        <option key={c.value} value={c.value}>{c.label}</option>
                      ))}
                    </select>
                  </div>
                  {uploadError && <div className="error-msg">{uploadError}</div>}
                  <button
                    className="btn btn-primary upload-start-btn"
                    onClick={handleUpload}
                    disabled={uploading}
                  >
                    {uploading ? 'Uploading…' : 'Start Pipeline'}
                  </button>
                </div>
              )}
              {uploadError && !uploadFile && (
                <div className="error-msg" style={{ marginTop: 8 }}>{uploadError}</div>
              )}
            </div>

            {/* Jobs table */}
            <div className="card jobs-card">
              <div className="jobs-card__title">Recent Jobs</div>
              {jobs.length === 0 ? (
                <div className="empty-state" style={{ padding: '30px 0' }}>
                  <p>No pipeline jobs yet</p>
                </div>
              ) : (
                <table className="jobs-table">
                  <thead>
                    <tr>
                      <th>File</th>
                      <th>Status</th>
                      <th>Started</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {jobs.map(job => (
                      <tr
                        key={job.id}
                        className={`job-row ${selectedJob === job.id ? 'job-row--active' : ''}`}
                        onClick={() => selectJob(job)}
                      >
                        <td className="job-filename" title={job.filename || job.file_name}>
                          {(job.filename || job.file_name || 'Unknown').replace(/\.pdf$/i, '')}
                        </td>
                        <td>{statusBadge(job.status)}</td>
                        <td className="job-time">{formatAgo(job.created_at || job.started_at)}</td>
                        <td>
                          <button
                            className="btn btn-danger btn-sm"
                            onClick={(e) => handleDelete(job.id, e)}
                            title="Delete job"
                          >
                            ✕
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>

          {/* Right column: job detail */}
          <div className="panel job-detail-panel">
            {!selectedJob ? (
              <div className="empty-state" style={{ flex: 1 }}>
                <p>Select a job to see details</p>
              </div>
            ) : loadingDetail ? (
              <div className="empty-state" style={{ flex: 1 }}>
                <p className="loading-dots">Loading</p>
              </div>
            ) : (
              <>
                <div className="panel-header">
                  <div>
                    <span className="job-detail-name">
                      {jobDetail?.filename || jobDetail?.file_name || currentJob?.filename || 'Job ' + selectedJob}
                    </span>
                    {jobDetail && (
                      <span style={{ marginLeft: 10 }}>{statusBadge(jobDetail.status)}</span>
                    )}
                  </div>
                  <div style={{ display: 'flex', gap: 6 }}>
                    {(jobDetail?.status === 'running' || jobDetail?.status === 'pending') && (
                      <button className="btn btn-secondary btn-sm" onClick={() => handleCancel(selectedJob)}>
                        Cancel
                      </button>
                    )}
                    {jobDetail?.status === 'error' && (
                      <>
                        <button
                          className="btn btn-primary btn-sm"
                          onClick={() => handleRetry(selectedJob)}
                          disabled={retrying}
                        >
                          {retrying ? 'Retrying…' : 'Retry'}
                        </button>
                        {errorStage && (
                          <button
                            className="btn btn-secondary btn-sm"
                            onClick={() => handleRetry(selectedJob, errorStage.name)}
                            disabled={retrying}
                            title={`Skip ${errorStage.name} and continue`}
                          >
                            Retry from next stage
                          </button>
                        )}
                      </>
                    )}
                  </div>
                </div>

                <div className="job-detail-body">
                  {/* Stage list */}
                  <div className="stages-section">
                    <div className="stages-title">Stages</div>
                    {stages.length === 0 ? (
                      <div className="empty-state" style={{ padding: '16px 0' }}>
                        <p>No stage data</p>
                      </div>
                    ) : (
                      <div className="stages-list">
                        {stages.map((s, i) => (
                          <StageDot key={s.name || i} stage={s} />
                        ))}
                      </div>
                    )}
                    {jobDetail?.duration_ms != null && (
                      <div className="job-duration">
                        Total: {formatDuration(jobDetail.duration_ms)}
                      </div>
                    )}
                  </div>

                  {/* Log console */}
                  <div className="log-section">
                    <div className="log-title">Log Output</div>
                    <div className="log-console" ref={logRef}>
                      {logLines.length === 0 ? (
                        <span className="log-empty">
                          {jobDetail?.status === 'running' || jobDetail?.status === 'pending'
                            ? 'Waiting for log output…'
                            : 'No log output available.'}
                        </span>
                      ) : (
                        logLines.map((line, i) => (
                          <div key={i} className="log-line">{line}</div>
                        ))
                      )}
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
