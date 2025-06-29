import React, { useState, useEffect } from 'react';

const BackupManager = ({ onRestore }) => {
  const [backups, setBackups] = useState([]);
  const [loading, setLoading] = useState(false);

  const loadBackups = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/config/backup/list');
      const result = await response.json();
      setBackups(result.backups || []);
    } catch (error) {
      alert('Failed to load backups: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadBackups();
  }, []);

  const handleRestore = async (filename) => {
    if (!window.confirm(`Are you sure you want to restore from backup: ${filename}? This will overwrite current configuration.`)) return;
    setLoading(true);
    try {
      const response = await fetch('/api/config/restore', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ backup_file: filename })
      });
      const result = await response.json();
      if (result.status === 'success') {
        alert('✅ Configuration restored successfully!');
        if (onRestore) onRestore();
      } else {
        throw new Error(result.message || 'Restore failed');
      }
    } catch (error) {
      alert('❌ Restore failed: ' + error.message);
    } finally {
      setLoading(false);
      loadBackups();
    }
  };

  return (
    <div className="config-form">
      <div className="form-header">
        <h3>💾 Backup & Restore</h3>
        <p>Manage configuration backups and restore points</p>
      </div>
      <div className="form-section">
        <h4>🗂️ Available Backups</h4>
        {loading ? (
          <div className="loading-spinner"></div>
        ) : (
          <ul>
            {backups.length === 0 && <li>No backups found.</li>}
            {backups.map(b => (
              <li key={b.filename} style={{ marginBottom: 12 }}>
                <strong>{b.filename}</strong> <br/>
                <span style={{ fontSize: 12, color: '#888' }}>
                  Created: {new Date(b.created).toLocaleString()} | Size: {(b.size/1024).toFixed(1)} KB
                </span>
                <br/>
                <button className="btn btn-secondary" style={{ marginTop: 6 }} onClick={() => handleRestore(b.filename)} disabled={loading}>
                  ♻️ Restore
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};

export default BackupManager; 