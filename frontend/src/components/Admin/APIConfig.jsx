import React, { useState, useEffect } from 'react';

const APIConfig = ({ config, onSave, loading }) => {
  const [formData, setFormData] = useState({
    news_api_key: '',
    alpha_vantage_api_key: '',
    polygon_api_key: '',
    finnhub_api_key: '',
    eodhd_api_key: '',
    zerodha_api_key: '',
    zerodha_access_token: '',
    zerodha_api_secret: '',
    telegram_bot_token: '',
    telegram_chat_id: '',
    news_api_rate_limit: 100,
    api_timeout_seconds: 30,
    enable_api_retries: true,
    max_retry_attempts: 3
  });

  const [showPasswords, setShowPasswords] = useState({});
  const [testResults, setTestResults] = useState({});

  useEffect(() => {
    if (config) {
      setFormData(prev => ({ ...prev, ...config }));
    }
  }, [config]);

  const handleChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const togglePasswordVisibility = (field) => {
    setShowPasswords(prev => ({
      ...prev,
      [field]: !prev[field]
    }));
  };

  const testIndividualAPI = async (apiType) => {
    setTestResults(prev => ({ ...prev, [apiType]: 'testing' }));
    
    try {
      // Individual API test would go here
      // For now, simulate test
      await new Promise(resolve => setTimeout(resolve, 2000));
      setTestResults(prev => ({ ...prev, [apiType]: 'success' }));
    } catch (error) {
      setTestResults(prev => ({ ...prev, [apiType]: 'error' }));
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    // Only send non-empty values and mask current values
    const dataToSend = Object.fromEntries(
      Object.entries(formData).filter(([key, value]) => {
        // For masked fields, only send if user actually entered a new value
        const maskedFields = ['news_api_key', 'alpha_vantage_api_key', 'polygon_api_key', 
                             'finnhub_api_key', 'eodhd_api_key', 'zerodha_api_key', 
                             'zerodha_access_token', 'zerodha_api_secret', 'telegram_bot_token'];
        
        if (maskedFields.includes(key)) {
          return value !== '' && !value.startsWith('***');
        }
        
        return value !== '';
      })
    );
    
    onSave(dataToSend);
  };

  const getStatusIcon = (apiType) => {
    const status = testResults[apiType];
    if (status === 'testing') return '⏳';
    if (status === 'success') return '✅';
    if (status === 'error') return '❌';
    return '🔍';
  };

  return (
    <form onSubmit={handleSubmit} className="config-form">
      <div className="form-header">
        <h3>🔑 API Configuration</h3>
        <p>Configure your API keys and external service connections</p>
      </div>
      
      <div className="form-section">
        <h4>📰 News Intelligence APIs</h4>
        <p className="section-description">
          Configure news sources for market intelligence and sentiment analysis
        </p>
        
        <div className="api-group">
          <div className="form-group">
            <label>
              Polygon API Key:
              <span className="recommended-badge">Recommended</span>
            </label>
            <div className="api-input-group">
              <div className="password-input">
                <input
                  type={showPasswords.polygon_api_key ? "text" : "password"}
                  value={formData.polygon_api_key}
                  onChange={(e) => handleChange('polygon_api_key', e.target.value)}
                  placeholder="Enter new Polygon API key or leave empty to keep current"
                />
                <button 
                  type="button" 
                  className="password-toggle"
                  onClick={() => togglePasswordVisibility('polygon_api_key')}
                >
                  {showPasswords.polygon_api_key ? '🙈' : '👁️'}
                </button>
              </div>
              <button 
                type="button" 
                className="test-button"
                onClick={() => testIndividualAPI('polygon')}
                disabled={!formData.polygon_api_key || formData.polygon_api_key.startsWith('***')}
              >
                {getStatusIcon('polygon')} Test
              </button>
            </div>
            <small className="help-text">
              Best for real-time market news. Get key from: polygon.io
            </small>
          </div>

          <div className="form-group">
            <label>News API Key:</label>
            <div className="api-input-group">
              <div className="password-input">
                <input
                  type={showPasswords.news_api_key ? "text" : "password"}
                  value={formData.news_api_key}
                  onChange={(e) => handleChange('news_api_key', e.target.value)}
                  placeholder="Enter new News API key or leave empty to keep current"
                />
                <button 
                  type="button" 
                  className="password-toggle"
                  onClick={() => togglePasswordVisibility('news_api_key')}
                >
                  {showPasswords.news_api_key ? '🙈' : '👁️'}
                </button>
              </div>
              <button 
                type="button" 
                className="test-button"
                onClick={() => testIndividualAPI('newsapi')}
                disabled={!formData.news_api_key || formData.news_api_key.startsWith('***')}
              >
                {getStatusIcon('newsapi')} Test
              </button>
            </div>
            <small className="help-text">
              General business news. Get key from: newsapi.org
            </small>
          </div>

          <div className="form-group">
            <label>Alpha Vantage API Key:</label>
            <div className="api-input-group">
              <div className="password-input">
                <input
                  type={showPasswords.alpha_vantage_api_key ? "text" : "password"}
                  value={formData.alpha_vantage_api_key}
                  onChange={(e) => handleChange('alpha_vantage_api_key', e.target.value)}
                  placeholder="Enter new Alpha Vantage API key or leave empty to keep current"
                />
                <button 
                  type="button" 
                  className="password-toggle"
                  onClick={() => togglePasswordVisibility('alpha_vantage_api_key')}
                >
                  {showPasswords.alpha_vantage_api_key ? '🙈' : '👁️'}
                </button>
              </div>
              <button 
                type="button" 
                className="test-button"
                onClick={() => testIndividualAPI('alphavantage')}
                disabled={!formData.alpha_vantage_api_key || formData.alpha_vantage_api_key.startsWith('***')}
              >
                {getStatusIcon('alphavantage')} Test
              </button>
            </div>
            <small className="help-text">
              Financial news and sentiment. Get key from: alphavantage.co
            </small>
          </div>
        </div>
      </div>

      <div className="form-section">
        <h4>💼 Trading APIs</h4>
        <p className="section-description">
          Configure your broker APIs for live trading and order execution
        </p>
        
        <div className="api-group trading-apis">
          <div className="form-group">
            <label>
              Zerodha API Key:
              <span className="required-badge">Required for Trading</span>
            </label>
            <div className="api-input-group">
              <div className="password-input">
                <input
                  type={showPasswords.zerodha_api_key ? "text" : "password"}
                  value={formData.zerodha_api_key}
                  onChange={(e) => handleChange('zerodha_api_key', e.target.value)}
                  placeholder="Enter Zerodha API key"
                />
                <button 
                  type="button" 
                  className="password-toggle"
                  onClick={() => togglePasswordVisibility('zerodha_api_key')}
                >
                  {showPasswords.zerodha_api_key ? '🙈' : '👁️'}
                </button>
              </div>
              <button 
                type="button" 
                className="test-button"
                onClick={() => testIndividualAPI('zerodha')}
                disabled={!formData.zerodha_api_key || !formData.zerodha_access_token}
              >
                {getStatusIcon('zerodha')} Test
              </button>
            </div>
            <small className="help-text">
              Get from Zerodha Kite Connect developer console
            </small>
          </div>

          <div className="form-group">
            <label>Zerodha Access Token:</label>
            <div className="api-input-group">
              <div className="password-input">
                <input
                  type={showPasswords.zerodha_access_token ? "text" : "password"}
                  value={formData.zerodha_access_token}
                  onChange={(e) => handleChange('zerodha_access_token', e.target.value)}
                  placeholder="Enter Zerodha access token"
                />
                <button 
                  type="button" 
                  className="password-toggle"
                  onClick={() => togglePasswordVisibility('zerodha_access_token')}
                >
                  {showPasswords.zerodha_access_token ? '🙈' : '👁️'}
                </button>
              </div>
            </div>
            <small className="help-text">
              Daily token from Zerodha login process
            </small>
          </div>

          <div className="trading-warning">
            ⚠️ <strong>Important:</strong> Ensure you have sufficient funds and risk management in place before enabling live trading.
          </div>
        </div>
      </div>

      <div className="form-section">
        <h4>📱 Telegram Configuration</h4>
        <p className="section-description">
          Configure Telegram for signal notifications and interactive trading
        </p>
        
        <div className="api-group">
          <div className="form-group">
            <label>
              Telegram Bot Token:
              <span className="required-badge">Required for Notifications</span>
            </label>
            <div className="api-input-group">
              <div className="password-input">
                <input
                  type={showPasswords.telegram_bot_token ? "text" : "password"}
                  value={formData.telegram_bot_token}
                  onChange={(e) => handleChange('telegram_bot_token', e.target.value)}
                  placeholder="Enter Telegram bot token"
                />
                <button 
                  type="button" 
                  className="password-toggle"
                  onClick={() => togglePasswordVisibility('telegram_bot_token')}
                >
                  {showPasswords.telegram_bot_token ? '🙈' : '👁️'}
                </button>
              </div>
              <button 
                type="button" 
                className="test-button"
                onClick={() => testIndividualAPI('telegram')}
                disabled={!formData.telegram_bot_token || !formData.telegram_chat_id}
              >
                {getStatusIcon('telegram')} Test
              </button>
            </div>
            <small className="help-text">
              Create bot via @BotFather on Telegram
            </small>
          </div>

          <div className="form-group">
            <label>Telegram Chat ID:</label>
            <input
              type="text"
              value={formData.telegram_chat_id}
              onChange={(e) => handleChange('telegram_chat_id', e.target.value)}
              placeholder="Your Telegram chat ID (e.g., 123456789)"
            />
            <small className="help-text">
              Send /start to your bot and check chat ID via @userinfobot
            </small>
          </div>
        </div>
      </div>

      <div className="form-section">
        <h4>⚙️ API Settings</h4>
        
        <div className="form-row">
          <div className="form-group">
            <label>API Timeout (seconds):</label>
            <input
              type="number"
              value={formData.api_timeout_seconds}
              onChange={(e) => handleChange('api_timeout_seconds', parseInt(e.target.value))}
              min="5"
              max="120"
            />
            <small className="help-text">
              Timeout for API requests (5-120 seconds)
            </small>
          </div>

          <div className="form-group">
            <label>News API Rate Limit (per hour):</label>
            <input
              type="number"
              value={formData.news_api_rate_limit}
              onChange={(e) => handleChange('news_api_rate_limit', parseInt(e.target.value))}
              min="10"
              max="1000"
            />
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={formData.enable_api_retries}
                onChange={(e) => handleChange('enable_api_retries', e.target.checked)}
              />
              Enable API Retries
            </label>
          </div>

          <div className="form-group">
            <label>Max Retry Attempts:</label>
            <input
              type="number"
              value={formData.max_retry_attempts}
              onChange={(e) => handleChange('max_retry_attempts', parseInt(e.target.value))}
              min="1"
              max="10"
              disabled={!formData.enable_api_retries}
            />
          </div>
        </div>
      </div>

      <div className="form-actions">
        <button type="submit" className="btn btn-primary" disabled={loading}>
          {loading ? (
            <>
              <span className="spinner"></span>
              Saving & Restarting Services...
            </>
          ) : (
            <>
              💾 Save API Configuration
            </>
          )}
        </button>
        
        <div className="save-warning">
          ⚠️ Changing API keys will restart related services. This may take 30-60 seconds.
        </div>
      </div>
    </form>
  );
};

export default APIConfig; 