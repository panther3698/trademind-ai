import React, { useState, useEffect } from 'react';
import TradingConfig from './TradingConfig';
import APIConfig from './APIConfig';
import SystemConfig from './SystemConfig';
import MLConfig from './MLConfig';
import BackupManager from './BackupManager';
import './ConfigurationManager.css';

const ConfigurationManager = () => {
  const [activeTab, setActiveTab] = useState('trading');
  const [configurations, setConfigurations] = useState({});
  const [loading, setLoading] = useState(false);
  const [saveStatus, setSaveStatus] = useState({});

  const tabs = [
    { key: 'trading', label: 'Trading', icon: '💰' },
    { key: 'apis', label: 'APIs', icon: '🔑' },
    { key: 'system', label: 'System', icon: '⚙️' },
    { key: 'ml', label: 'ML Models', icon: '🤖' },
    { key: 'backup', label: 'Backup', icon: '💾' }
  ];

  const loadAllConfigs = async () => {
    setLoading(true);
    try {
      const [trading, apis, system, ml] = await Promise.all([
        fetch('/api/config/trading').then(r => r.json()),
        fetch('/api/config/apis').then(r => r.json()),
        fetch('/api/config/system').then(r => r.json()),
        fetch('/api/config/ml').then(r => r.json())
      ]);
      
      setConfigurations({ trading, apis, system, ml });
    } catch (error) {
      console.error('Failed to load configurations:', error);
      alert('Failed to load configurations: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAllConfigs();
  }, []);

  const handleSave = async (configType, data) => {
    setSaveStatus(prev => ({ ...prev, [configType]: 'saving' }));
    
    try {
      const response = await fetch(`/api/config/${configType}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });
      
      const result = await response.json();
      
      if (result.status === 'success') {
        setSaveStatus(prev => ({ ...prev, [configType]: 'success' }));
        alert('✅ ' + result.message);
        await loadAllConfigs(); // Reload to get updated values
      } else {
        throw new Error(result.message || 'Save failed');
      }
    } catch (error) {
      setSaveStatus(prev => ({ ...prev, [configType]: 'error' }));
      alert('❌ Failed to save configuration: ' + error.message);
    }

    // Clear status after 3 seconds
    setTimeout(() => {
      setSaveStatus(prev => ({ ...prev, [configType]: null }));
    }, 3000);
  };

  const handleTestAPIs = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/config/test-apis', { method: 'POST' });
      const result = await response.json();
      
      // Defensive check for invalid response
      if (!result || typeof result.test_results !== 'object' || result.test_results === null) {
        alert('❌ API test failed: Invalid response from server.');
        return;
      }
      // Always show all expected services
      const expectedServices = {
        zerodha: "Zerodha Broker",
        telegram: "Telegram",
        news_apis: "News APIs"
      };
      let message = '🧪 API Test Results:\n\n';
      for (const key of Object.keys(expectedServices)) {
        const status = result.test_results[key];
        let statusText = "Not Configured";
        let emoji = "⚪";
        if (status) {
          if (status.status === "success" || status.status === "connected") {
            statusText = "OK";
            emoji = "✅";
          } else if (status.status === "failed" || status.status === "error") {
            statusText = "Error";
            emoji = "❌";
          } else if (status.status === "not_initialized") {
            statusText = "Not Initialized";
            emoji = "🟣";
          } else if (status.status === "not_configured") {
            statusText = "Not Configured";
            emoji = "⚪";
          } else {
            statusText = status.status;
            emoji = "⚠️";
          }
        }
        message += `${emoji} ${expectedServices[key]}: ${statusText}\n`;
        if (status && status.user) message += `   User: ${status.user}\n`;
        if (status && status.funds) message += `   Funds: ₹${status.funds.toLocaleString()}\n`;
        if (status && status.error) message += `   Error: ${status.error}\n`;
      }
      alert(message);
    } catch (error) {
      alert('❌ API test failed: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleBackup = async () => {
    try {
      const response = await fetch('/api/config/backup', { method: 'POST' });
      const result = await response.json();
      alert(`✅ Configuration backed up successfully!\nFile: ${result.backup_file}`);
    } catch (error) {
      alert('❌ Backup failed: ' + error.message);
    }
  };

  if (loading && Object.keys(configurations).length === 0) {
    return (
      <div className="config-loading">
        <div className="loading-spinner"></div>
        <p>Loading configurations...</p>
      </div>
    );
  }

  return (
    <div className="configuration-manager">
      <div className="config-header">
        <h2>🔧 TradeMind AI Configuration</h2>
        <div className="config-actions">
          <button onClick={handleTestAPIs} className="btn btn-info" disabled={loading}>
            🧪 Test All APIs
          </button>
          <button onClick={handleBackup} className="btn btn-secondary">
            💾 Backup Config
          </button>
          <button onClick={loadAllConfigs} className="btn btn-primary" disabled={loading}>
            🔄 Refresh
          </button>
        </div>
      </div>

      <div className="config-tabs">
        {tabs.map(tab => (
          <button
            key={tab.key}
            className={`tab-button ${activeTab === tab.key ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.key)}
          >
            <span className="tab-icon">{tab.icon}</span>
            <span className="tab-label">{tab.label}</span>
            {saveStatus[tab.key] && (
              <span className={`save-status ${saveStatus[tab.key]}`}>
                {saveStatus[tab.key] === 'saving' && '⏳'}
                {saveStatus[tab.key] === 'success' && '✅'}
                {saveStatus[tab.key] === 'error' && '❌'}
              </span>
            )}
          </button>
        ))}
      </div>

      <div className="config-content">
        {activeTab === 'trading' && (
          <TradingConfig 
            config={configurations.trading} 
            onSave={(data) => handleSave('trading', data)}
            loading={saveStatus.trading === 'saving'}
          />
        )}
        {activeTab === 'apis' && (
          <APIConfig 
            config={configurations.apis} 
            onSave={(data) => handleSave('apis', data)}
            loading={saveStatus.apis === 'saving'}
          />
        )}
        {activeTab === 'system' && (
          <SystemConfig 
            config={configurations.system} 
            onSave={(data) => handleSave('system', data)}
            loading={saveStatus.system === 'saving'}
          />
        )}
        {activeTab === 'ml' && (
          <MLConfig 
            config={configurations.ml} 
            onSave={(data) => handleSave('ml', data)}
            loading={saveStatus.ml === 'saving'}
          />
        )}
        {activeTab === 'backup' && (
          <BackupManager onRestore={loadAllConfigs} />
        )}
      </div>
    </div>
  );
};

export default ConfigurationManager; 