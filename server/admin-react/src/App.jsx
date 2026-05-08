import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './hooks/useAuth.jsx';
import RequireAuth from './components/RequireAuth.jsx';
import Login from './pages/Login.jsx';
import Handoffs from './pages/Handoffs.jsx';
import Analytics from './pages/Analytics.jsx';
import Content from './pages/Content.jsx';
import Pipeline from './pages/Pipeline.jsx';
import KnowledgeBase from './pages/KnowledgeBase.jsx';

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter basename="/app">
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/kb" element={<KnowledgeBase />} />
          <Route
            path="/admin"
            element={<RequireAuth><Handoffs /></RequireAuth>}
          />
          <Route
            path="/analytics"
            element={<RequireAuth><Analytics /></RequireAuth>}
          />
          <Route
            path="/content"
            element={<RequireAuth><Content /></RequireAuth>}
          />
          <Route
            path="/pipeline"
            element={<RequireAuth><Pipeline /></RequireAuth>}
          />
          <Route path="/" element={<Navigate to="/admin" replace />} />
          <Route path="*" element={<Navigate to="/admin" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}
