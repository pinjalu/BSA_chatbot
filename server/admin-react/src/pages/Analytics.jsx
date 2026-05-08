import React, { useState, useEffect } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import Header from '../components/Header.jsx';
import { useAuth } from '../hooks/useAuth.jsx';
import './Analytics.css';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

function StatCard({ label, value, color }) {
  return (
    <div className="stat-card" style={{ borderTopColor: color }}>
      <div className="stat-card__value" style={{ color }}>{value ?? '—'}</div>
      <div className="stat-card__label">{label}</div>
    </div>
  );
}

function formatDate(dateStr) {
  const d = new Date(dateStr + 'T00:00:00');
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

export default function Analytics() {
  const { authHeader } = useAuth();
  const [days, setDays] = useState(7);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    setLoading(true);
    setError('');
    fetch(`/api/admin/analytics?days=${days}`, { headers: authHeader })
      .then(r => r.ok ? r.json() : Promise.reject(r.statusText))
      .then(d => { setData(d); setLoading(false); })
      .catch(e => { setError('Failed to load analytics: ' + e); setLoading(false); });
  }, [days, authHeader]);

  const dailyData = data?.daily || [];
  // DB returns overview + handoffs separately; merge into a summary object
  const ov = data?.overview || {};
  const hf = data?.handoffs || {};
  const summary = {
    total: ov.total_questions ?? ov.total ?? null,
    answered: ov.answered ?? null,
    unanswered: ov.unanswered ?? null,
    handoffs: (parseInt(hf.open_count || 0) + parseInt(hf.resolved_count || 0)) || null,
  };

  const chartData = {
    labels: dailyData.map(d => formatDate(d.day || d.date)),
    datasets: [
      {
        label: 'Answered',
        data: dailyData.map(d => d.answered || 0),
        backgroundColor: '#2e7d32',
        borderRadius: 4,
        borderSkipped: false,
      },
      {
        label: 'Unanswered',
        data: dailyData.map(d => d.unanswered || 0),
        backgroundColor: '#b71c1c',
        borderRadius: 4,
        borderSkipped: false,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: 'top' },
      title: { display: false },
      tooltip: {
        callbacks: {
          label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y}`,
        },
      },
    },
    scales: {
      x: { grid: { display: false } },
      y: {
        beginAtZero: true,
        ticks: { stepSize: 1, precision: 0 },
        grid: { color: '#f0f0f0' },
      },
    },
  };

  const unanswered = data?.unanswered_queries || [];

  return (
    <div className="page-layout">
      <Header />
      <div className="page-content">
        <div className="analytics-header">
          <h1 className="page-title">Analytics</h1>
          <div className="day-toggle">
            {[7, 30, 90].map(d => (
              <button
                key={d}
                className={`day-btn ${days === d ? 'day-btn--active' : ''}`}
                onClick={() => setDays(d)}
              >
                {d}d
              </button>
            ))}
          </div>
        </div>

        {error && <div className="error-msg" style={{ marginBottom: 16 }}>{error}</div>}

        {loading ? (
          <div className="analytics-loading">Loading analytics<span className="loading-dots" /></div>
        ) : (
          <>
            <div className="stat-cards">
              <StatCard label="Total Questions" value={summary.total} color="#1565c0" />
              <StatCard label="Answered" value={summary.answered} color="#2e7d32" />
              <StatCard label="Unanswered" value={summary.unanswered} color="#b71c1c" />
              <StatCard label="Handoffs" value={summary.handoffs} color="#e65100" />
            </div>

            <div className="card chart-card">
              <div className="chart-title">Daily Questions — Last {days} Days</div>
              <div className="chart-wrap">
                <Bar data={chartData} options={chartOptions} />
              </div>
            </div>

            <div className="card unanswered-card">
              <div className="chart-title">Unanswered Queries</div>
              {unanswered.length === 0 ? (
                <div className="empty-state" style={{ padding: '30px 0' }}>
                  <p>No unanswered queries in this period</p>
                </div>
              ) : (
                <table className="unanswered-table">
                  <thead>
                    <tr>
                      <th>Question</th>
                      <th>Time</th>
                    </tr>
                  </thead>
                  <tbody>
                    {unanswered.map((q, i) => (
                      <tr key={i}>
                        <td>{q.question || q.content || '—'}</td>
                        <td className="unanswered-time">
                          {q.created_at
                            ? new Date(q.created_at).toLocaleString([], {
                                month: 'short', day: 'numeric',
                                hour: '2-digit', minute: '2-digit',
                              })
                            : '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
