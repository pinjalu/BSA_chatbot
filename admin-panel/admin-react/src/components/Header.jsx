import React from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth.jsx';
import './Header.css';

export default function Header() {
  const { logout } = useAuth();
  const navigate = useNavigate();

  function handleLogout() {
    logout();
    navigate('/login');
  }

  return (
    <header className="site-header">
      <div className="site-header__brand">
        <span className="site-header__logo">BSA</span>
        <span className="site-header__title">Admin</span>
      </div>
      <nav className="site-header__nav">
        <NavLink to="/admin" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
          Handoffs
        </NavLink>
        <NavLink to="/analytics" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
          Analytics
        </NavLink>
        <NavLink to="/content" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
          Content
        </NavLink>
        <NavLink to="/pipeline" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
          Pipeline
        </NavLink>
      </nav>
      <button className="site-header__logout btn btn-sm" onClick={handleLogout}>
        Sign out
      </button>
    </header>
  );
}
