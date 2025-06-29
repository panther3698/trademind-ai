import React, { useState, useEffect } from 'react';

const SystemConfig = ({ config, onSave, loading }) => {
  const [formData, setFormData] = useState({
    // Timing & Intervals
    signal_generation_interval: 300,
    news_monitoring_interval: 60,
    market_open_time: "09:15",
    market_close_time: "15:30",
    timezone: "Asia/Kolkata",
    // Feature Toggles
    enable_news_intelligence: true,
    enable_regime_detection: true,
    enable_interactive_trading: true,
    enable_order_execution: true,
    enable_demo_mode: false,
    enable_backtesting: true,
    enable_webhook_handler: true,
    enable_analytics_tracking: true,
    // Cache Settings
    cache_ttl_market_data: 5,
    cache_ttl_news_data: 60,
    cache_ttl_regime_data: 300,
    enable_cache_compression: true,
    max_cache_size_mb: 500,
    // Logging & Monitoring
    log_level: "INFO",
    enable_detailed_logging: true,
    enable_performance_monitoring: true,
    enable_error_reporting: true,
    max_log_file_size_mb: 100,
    log_retention_days: 30,
    // Performance Settings
    max_concurrent_api_calls: 10,
    request_timeout_seconds: 30,
    background_task_interval: 30,
    health_check_interval: 60,
    // Security Settings
    enable_api_rate_limiting: true,
    max_requests_per_minute: 60,
    enable_request_logging: true,
    session_timeout_minutes: 120,
    // Trading Hours & Holidays
    trading_days: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
    enable_holiday_detection: true,
    premarket_start_time: "08:00",
    postmarket_end_time: "16:00"
  });

  const [errors, setErrors] = useState({});
  const [activeSection, setActiveSection] = useState('timing');

  useEffect(() => {
    if (config) {
      setFormData(prev => ({ ...prev, ...config }));
    }
  }, [config]);

  const validateField = (field, value) => {
    const newErrors = { ...errors };
    switch (field) {
      case 'signal_generation_interval':
        if (value < 30 || value > 3600) {
          newErrors[field] = 'Signal interval must be between 30 seconds and 1 hour';
        } else {
          delete newErrors[field];
        }
        break;
      case 'news_monitoring_interval':
        if (value < 15 || value > 600) {
          newErrors[field] = 'News interval must be between 15 seconds and 10 minutes';
        } else {
          delete newErrors[field];
        }
        break;
      case 'max_cache_size_mb':
        if (value < 50 || value > 2048) {
          newErrors[field] = 'Cache size must be between 50MB and 2GB';
        } else {
          delete newErrors[field];
        }
        break;
      case 'session_timeout_minutes':
        if (value < 15 || value > 480) {
          newErrors[field] = 'Session timeout must be between 15 minutes and 8 hours';
        } else {
          delete newErrors[field];
        }
        break;
    }
    setErrors(newErrors);
  };

  const handleChange = (field, value) => {
    const numericFields = [
      'signal_generation_interval', 'news_monitoring_interval', 'cache_ttl_market_data',
      'cache_ttl_news_data', 'cache_ttl_regime_data', 'max_cache_size_mb',
      'max_log_file_size_mb', 'log_retention_days', 'max_concurrent_api_calls',
      'request_timeout_seconds', 'background_task_interval', 'health_check_interval',
      'max_requests_per_minute', 'session_timeout_minutes'
    ];
    const processedValue = numericFields.includes(field) 
      ? parseInt(value) || 0 
      : value;
    setFormData(prev => ({
      ...prev,
      [field]: processedValue
    }));
    if (numericFields.includes(field)) {
      validateField(field, processedValue);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    Object.keys(formData).forEach(field => {
      if (typeof formData[field] === 'number') {
        validateField(field, formData[field]);
      }
    });
    if (Object.keys(errors).length > 0) {
      alert('Please fix validation errors before saving');
      return;
    }
    onSave(formData);
  };

  const sections = [
    { key: 'timing', label: 'Timing & Intervals', icon: '⏰' },
    { key: 'features', label: 'Feature Toggles', icon: '🎛️' },
    { key: 'cache', label: 'Cache Settings', icon: '💾' },
    { key: 'logging', label: 'Logging & Monitoring', icon: '📊' },
    { key: 'performance', label: 'Performance', icon: '🚀' },
    { key: 'security', label: 'Security', icon: '🔒' },
    { key: 'schedule', label: 'Trading Schedule', icon: '📅' }
  ];

  const formatInterval = (seconds) => {
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
    return `${Math.floor(seconds / 3600)}h`;
  };

  const getFeatureStatus = () => {
    const features = [
      'enable_news_intelligence', 'enable_regime_detection', 'enable_interactive_trading',
      'enable_order_execution', 'enable_backtesting', 'enable_webhook_handler'
    ];
    const enabled = features.filter(f => formData[f]).length;
    return `${enabled}/${features.length} features enabled`;
  };

  return (
    <form onSubmit={handleSubmit} className="config-form">
      <div className="form-header">
        <h3>⚙️ System Configuration</h3>
        <p>Configure system behavior, performance settings, and operational parameters</p>
        <div className="system-overview">
          <div className="overview-metric">
            <span className="metric-label">Signal Interval:</span>
            <span className="metric-value">{formatInterval(formData.signal_generation_interval)}</span>
          </div>
          <div className="overview-metric">
            <span className="metric-label">Features:</span>
            <span className="metric-value">{getFeatureStatus()}</span>
          </div>
          <div className="overview-metric">
            <span className="metric-label">Cache Size:</span>
            <span className="metric-value">{formData.max_cache_size_mb}MB</span>
          </div>
        </div>
      </div>
      <div className="config-sections">
        <div className="section-tabs">
          {sections.map(section => (
            <button
              key={section.key}
              type="button"
              className={`section-tab ${activeSection === section.key ? 'active' : ''}`}
              onClick={() => setActiveSection(section.key)}
            >
              <span className="tab-icon">{section.icon}</span>
              <span className="tab-label">{section.label}</span>
            </button>
          ))}
        </div>
        <div className="section-content">
          {/* Timing & Intervals Section */}
          {activeSection === 'timing' && (
            <div className="form-section">
              <h4>⏰ Timing & Intervals</h4>
              <p className="section-description">
                Configure how frequently different system operations run
              </p>
              <div className="form-row">
                <div className="form-group">
                  <label>
                    Signal Generation Interval:
                    <span className="tooltip" title="How often to check for new trading signals">ℹ️</span>
                  </label>
                  <div className="input-group">
                    <input
                      type="number"
                      value={formData.signal_generation_interval}
                      onChange={(e) => handleChange('signal_generation_interval', e.target.value)}
                      min="30"
                      max="3600"
                      step="30"
                      className={errors.signal_generation_interval ? 'error' : ''}
                    />
                    <span className="input-suffix">seconds</span>
                  </div>
                  {errors.signal_generation_interval && (
                    <span className="error-message">{errors.signal_generation_interval}</span>
                  )}
                  <small className="help-text">
                    Recommended: 300s (5 min) for active trading, 600s (10 min) for conservative
                  </small>
                </div>
                <div className="form-group">
                  <label>
                    News Monitoring Interval:
                    <span className="tooltip" title="How often to check for new market news">ℹ️</span>
                  </label>
                  <div className="input-group">
                    <input
                      type="number"
                      value={formData.news_monitoring_interval}
                      onChange={(e) => handleChange('news_monitoring_interval', e.target.value)}
                      min="15"
                      max="600"
                      step="15"
                      className={errors.news_monitoring_interval ? 'error' : ''}
                    />
                    <span className="input-suffix">seconds</span>
                  </div>
                  {errors.news_monitoring_interval && (
                    <span className="error-message">{errors.news_monitoring_interval}</span>
                  )}
                  <small className="help-text">
                    Recommended: 60s for real-time news monitoring
                  </small>
                </div>
              </div>
              <div className="form-row">
                <div className="form-group">
                  <label>Background Task Interval:</label>
                  <div className="input-group">
                    <input
                      type="number"
                      value={formData.background_task_interval}
                      onChange={(e) => handleChange('background_task_interval', e.target.value)}
                      min="10"
                      max="300"
                      step="10"
                    />
                    <span className="input-suffix">seconds</span>
                  </div>
                  <small className="help-text">
                    Interval for system maintenance tasks
                  </small>
                </div>
                <div className="form-group">
                  <label>Health Check Interval:</label>
                  <div className="input-group">
                    <input
                      type="number"
                      value={formData.health_check_interval}
                      onChange={(e) => handleChange('health_check_interval', e.target.value)}
                      min="30"
                      max="600"
                      step="30"
                    />
                    <span className="input-suffix">seconds</span>
                  </div>
                  <small className="help-text">
                    How often to check system health
                  </small>
                </div>
              </div>
            </div>
          )}
          {/* Feature Toggles Section */}
          {activeSection === 'features' && (
            <div className="form-section">
              <h4>🎛️ Feature Toggles</h4>
              <p className="section-description">
                Enable or disable system features and modules
              </p>
              <div className="feature-grid">
                <div className="feature-group">
                  <h5>🤖 AI & Intelligence</h5>
                  <div className="toggle-group">
                    <label className="toggle-label">
                      <input
                        type="checkbox"
                        checked={formData.enable_news_intelligence}
                        onChange={(e) => handleChange('enable_news_intelligence', e.target.checked)}
                      />
                      <span className="toggle-slider"></span>
                      <span className="toggle-text">News Intelligence</span>
                    </label>
                    <small>Real-time news analysis and sentiment detection</small>
                    <label className="toggle-label">
                      <input
                        type="checkbox"
                        checked={formData.enable_regime_detection}
                        onChange={(e) => handleChange('enable_regime_detection', e.target.checked)}
                      />
                      <span className="toggle-slider"></span>
                      <span className="toggle-text">Regime Detection</span>
                    </label>
                    <small>Market regime analysis and strategy adaptation</small>
                  </div>
                </div>
                <div className="feature-group">
                  <h5>💼 Trading</h5>
                  <div className="toggle-group">
                    <label className="toggle-label">
                      <input
                        type="checkbox"
                        checked={formData.enable_interactive_trading}
                        onChange={(e) => handleChange('enable_interactive_trading', e.target.checked)}
                      />
                      <span className="toggle-slider"></span>
                      <span className="toggle-text">Interactive Trading</span>
                    </label>
                    <small>Telegram approval system for signals</small>
                    <label className="toggle-label">
                      <input
                        type="checkbox"
                        checked={formData.enable_order_execution}
                        onChange={(e) => handleChange('enable_order_execution', e.target.checked)}
                      />
                      <span className="toggle-slider"></span>
                      <span className="toggle-text">Order Execution</span>
                    </label>
                    <small>Automatic order placement via broker API</small>
                    <label className="toggle-label">
                      <input
                        type="checkbox"
                        checked={formData.enable_demo_mode}
                        onChange={(e) => handleChange('enable_demo_mode', e.target.checked)}
                      />
                      <span className="toggle-slider"></span>
                      <span className="toggle-text">Demo Mode</span>
                    </label>
                    <small>Paper trading for testing (disables real orders)</small>
                  </div>
                </div>
                <div className="feature-group">
                  <h5>🔧 System</h5>
                  <div className="toggle-group">
                    <label className="toggle-label">
                      <input
                        type="checkbox"
                        checked={formData.enable_backtesting}
                        onChange={(e) => handleChange('enable_backtesting', e.target.checked)}
                      />
                      <span className="toggle-slider"></span>
                      <span className="toggle-text">Backtesting Engine</span>
                    </label>
                    <small>Historical strategy testing capabilities</small>
                    <label className="toggle-label">
                      <input
                        type="checkbox"
                        checked={formData.enable_webhook_handler}
                        onChange={(e) => handleChange('enable_webhook_handler', e.target.checked)}
                      />
                      <span className="toggle-slider"></span>
                      <span className="toggle-text">Webhook Handler</span>
                    </label>
                    <small>Telegram webhook processing</small>
                    <label className="toggle-label">
                      <input
                        type="checkbox"
                        checked={formData.enable_analytics_tracking}
                        onChange={(e) => handleChange('enable_analytics_tracking', e.target.checked)}
                      />
                      <span className="toggle-slider"></span>
                      <span className="toggle-text">Analytics Tracking</span>
                    </label>
                    <small>Performance and usage analytics</small>
                  </div>
                </div>
              </div>
            </div>
          )}
          {/* Cache Settings Section */}
          {activeSection === 'cache' && (
            <div className="form-section">
              <h4>💾 Cache Settings</h4>
              <p className="section-description">
                Configure caching to optimize performance and reduce API calls
              </p>
              <div className="form-row">
                <div className="form-group">
                  <label>Market Data Cache TTL:</label>
                  <div className="input-group">
                    <input
                      type="number"
                      value={formData.cache_ttl_market_data}
                      onChange={(e) => handleChange('cache_ttl_market_data', e.target.value)}
                      min="1"
                      max="300"
                    />
                    <span className="input-suffix">seconds</span>
                  </div>
                  <small className="help-text">
                    How long to cache market data (recommended: 5s)
                  </small>
                </div>
                <div className="form-group">
                  <label>News Data Cache TTL:</label>
                  <div className="input-group">
                    <input
                      type="number"
                      value={formData.cache_ttl_news_data}
                      onChange={(e) => handleChange('cache_ttl_news_data', e.target.value)}
                      min="30"
                      max="3600"
                    />
                    <span className="input-suffix">seconds</span>
                  </div>
                  <small className="help-text">
                    How long to cache news data (recommended: 60s)
                  </small>
                </div>
              </div>
              <div className="form-row">
                <div className="form-group">
                  <label>Regime Data Cache TTL:</label>
                  <div className="input-group">
                    <input
                      type="number"
                      value={formData.cache_ttl_regime_data}
                      onChange={(e) => handleChange('cache_ttl_regime_data', e.target.value)}
                      min="60"
                      max="1800"
                    />
                    <span className="input-suffix">seconds</span>
                  </div>
                  <small className="help-text">
                    How long to cache regime analysis (recommended: 300s)
                  </small>
                </div>
                <div className="form-group">
                  <label>Max Cache Size:</label>
                  <div className="input-group">
                    <input
                      type="number"
                      value={formData.max_cache_size_mb}
                      onChange={(e) => handleChange('max_cache_size_mb', e.target.value)}
                      min="50"
                      max="2048"
                      className={errors.max_cache_size_mb ? 'error' : ''}
                    />
                    <span className="input-suffix">MB</span>
                  </div>
                  {errors.max_cache_size_mb && (
                    <span className="error-message">{errors.max_cache_size_mb}</span>
                  )}
                  <small className="help-text">
                    Maximum memory for caching (recommended: 500MB)
                  </small>
                </div>
              </div>
              <div className="form-row">
                <div className="form-group">
                  <label className="toggle-label">
                    <input
                      type="checkbox"
                      checked={formData.enable_cache_compression}
                      onChange={(e) => handleChange('enable_cache_compression', e.target.checked)}
                    />
                    <span className="toggle-slider"></span>
                    <span className="toggle-text">Enable Cache Compression</span>
                  </label>
                  <small className="help-text">
                    Compress cached data to save memory
                  </small>
                </div>
              </div>
            </div>
          )}
          {/* Logging & Monitoring Section */}
          {activeSection === 'logging' && (
            <div className="form-section">
              <h4>📊 Logging & Monitoring</h4>
              <p className="section-description">
                Configure system logging, monitoring, and debugging options
              </p>
              <div className="form-row">
                <div className="form-group">
                  <label>Log Level:</label>
                  <select
                    value={formData.log_level}
                    onChange={(e) => handleChange('log_level', e.target.value)}
                  >
                    <option value="DEBUG">DEBUG (Most detailed)</option>
                    <option value="INFO">INFO (Recommended)</option>
                    <option value="WARNING">WARNING (Important only)</option>
                    <option value="ERROR">ERROR (Errors only)</option>
                    <option value="CRITICAL">CRITICAL (Critical only)</option>
                  </select>
                  <small className="help-text">
                    Higher levels = less logging. INFO recommended for production.
                  </small>
                </div>
                <div className="form-group">
                  <label>Max Log File Size:</label>
                  <div className="input-group">
                    <input
                      type="number"
                      value={formData.max_log_file_size_mb}
                      onChange={(e) => handleChange('max_log_file_size_mb', e.target.value)}
                      min="10"
                      max="1000"
                    />
                    <span className="input-suffix">MB</span>
                  </div>
                  <small className="help-text">
                    When log files reach this size, they'll be rotated
                  </small>
                </div>
              </div>
              <div className="form-row">
                <div className="form-group">
                  <label>Log Retention Period:</label>
                  <div className="input-group">
                    <input
                      type="number"
                      value={formData.log_retention_days}
                      onChange={(e) => handleChange('log_retention_days', e.target.value)}
                      min="1"
                      max="365"
                    />
                    <span className="input-suffix">days</span>
                  </div>
                  <small className="help-text">
                    How long to keep old log files
                  </small>
                </div>
              </div>
              <div className="toggle-section">
                <label className="toggle-label">
                  <input
                    type="checkbox"
                    checked={formData.enable_detailed_logging}
                    onChange={(e) => handleChange('enable_detailed_logging', e.target.checked)}
                  />
                  <span className="toggle-slider"></span>
                  <span className="toggle-text">Enable Detailed Logging</span>
                </label>
                <small>Include detailed debug information in logs</small>
                <label className="toggle-label">
                  <input
                    type="checkbox"
                    checked={formData.enable_performance_monitoring}
                    onChange={(e) => handleChange('enable_performance_monitoring', e.target.checked)}
                  />
                  <span className="toggle-slider"></span>
                  <span className="toggle-text">Performance Monitoring</span>
                </label>
                <small>Track system performance metrics</small>
                <label className="toggle-label">
                  <input
                    type="checkbox"
                    checked={formData.enable_error_reporting}
                    onChange={(e) => handleChange('enable_error_reporting', e.target.checked)}
                  />
                  <span className="toggle-slider"></span>
                  <span className="toggle-text">Error Reporting</span>
                </label>
                <small>Automatically report critical errors</small>
              </div>
            </div>
          )}
          {/* Performance Section */}
          {activeSection === 'performance' && (
            <div className="form-section">
              <h4>🚀 Performance Settings</h4>
              <p className="section-description">
                Configure system performance and resource usage limits
              </p>
              <div className="form-row">
                <div className="form-group">
                  <label>Max Concurrent API Calls:</label>
                  <input
                    type="number"
                    value={formData.max_concurrent_api_calls}
                    onChange={(e) => handleChange('max_concurrent_api_calls', e.target.value)}
                    min="1"
                    max="50"
                  />
                  <small className="help-text">
                    Maximum simultaneous API requests (recommended: 10)
                  </small>
                </div>
                <div className="form-group">
                  <label>Request Timeout:</label>
                  <div className="input-group">
                    <input
                      type="number"
                      value={formData.request_timeout_seconds}
                      onChange={(e) => handleChange('request_timeout_seconds', e.target.value)}
                      min="5"
                      max="120"
                    />
                    <span className="input-suffix">seconds</span>
                  </div>
                  <small className="help-text">
                    How long to wait for API responses
                  </small>
                </div>
              </div>
            </div>
          )}
          {/* Security Section */}
          {activeSection === 'security' && (
            <div className="form-section">
              <h4>🔒 Security Settings</h4>
              <p className="section-description">
                Configure security features and access controls
              </p>
              <div className="form-row">
                <div className="form-group">
                  <label>Max Requests Per Minute:</label>
                  <input
                    type="number"
                    value={formData.max_requests_per_minute}
                    onChange={(e) => handleChange('max_requests_per_minute', e.target.value)}
                    min="10"
                    max="1000"
                  />
                  <small className="help-text">
                    Rate limiting for API endpoints
                  </small>
                </div>
                <div className="form-group">
                  <label>Session Timeout:</label>
                  <div className="input-group">
                    <input
                      type="number"
                      value={formData.session_timeout_minutes}
                      onChange={(e) => handleChange('session_timeout_minutes', e.target.value)}
                      min="15"
                      max="480"
                      className={errors.session_timeout_minutes ? 'error' : ''}
                    />
                    <span className="input-suffix">minutes</span>
                  </div>
                  {errors.session_timeout_minutes && (
                    <span className="error-message">{errors.session_timeout_minutes}</span>
                  )}
                  <small className="help-text">
                    How long sessions stay active
                  </small>
                </div>
              </div>
              <div className="toggle-section">
                <label className="toggle-label">
                  <input
                    type="checkbox"
                    checked={formData.enable_api_rate_limiting}
                    onChange={(e) => handleChange('enable_api_rate_limiting', e.target.checked)}
                  />
                  <span className="toggle-slider"></span>
                  <span className="toggle-text">API Rate Limiting</span>
                </label>
                <small>Protect against excessive API usage</small>
                <label className="toggle-label">
                  <input
                    type="checkbox"
                    checked={formData.enable_request_logging}
                    onChange={(e) => handleChange('enable_request_logging', e.target.checked)}
                  />
                  <span className="toggle-slider"></span>
                  <span className="toggle-text">Request Logging</span>
                </label>
                <small>Log all API requests for security monitoring</small>
              </div>
            </div>
          )}
          {/* Trading Schedule Section */}
          {activeSection === 'schedule' && (
            <div className="form-section">
              <h4>📅 Trading Schedule</h4>
              <p className="section-description">
                Configure trading hours and market schedule
              </p>
              <div className="form-row">
                <div className="form-group">
                  <label>Market Open Time:</label>
                  <input
                    type="time"
                    value={formData.market_open_time}
                    onChange={(e) => handleChange('market_open_time', e.target.value)}
                  />
                  <small className="help-text">
                    When regular trading begins (IST)
                  </small>
                </div>
                <div className="form-group">
                  <label>Market Close Time:</label>
                  <input
                    type="time"
                    value={formData.market_close_time}
                    onChange={(e) => handleChange('market_close_time', e.target.value)}
                  />
                  <small className="help-text">
                    When regular trading ends (IST)
                  </small>
                </div>
              </div>
              <div className="form-row">
                <div className="form-group">
                  <label>Pre-market Start:</label>
                  <input
                    type="time"
                    value={formData.premarket_start_time}
                    onChange={(e) => handleChange('premarket_start_time', e.target.value)}
                  />
                </div>
                <div className="form-group">
                  <label>Post-market End:</label>
                  <input
                    type="time"
                    value={formData.postmarket_end_time}
                    onChange={(e) => handleChange('postmarket_end_time', e.target.value)}
                  />
                </div>
              </div>
              <div className="form-group">
                <label>Trading Days:</label>
                <div className="day-selector">
                  {["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].map(day => (
                    <label key={day} className="day-checkbox">
                      <input
                        type="checkbox"
                        checked={formData.trading_days.includes(day)}
                        onChange={(e) => {
                          const days = e.target.checked
                            ? [...formData.trading_days, day]
                            : formData.trading_days.filter(d => d !== day);
                          handleChange('trading_days', days);
                        }}
                      />
                      <span>{day.slice(0, 3)}</span>
                    </label>
                  ))}
                </div>
                <small className="help-text">
                  Select active trading days
                </small>
              </div>
              <div className="toggle-section">
                <label className="toggle-label">
                  <input
                    type="checkbox"
                    checked={formData.enable_holiday_detection}
                    onChange={(e) => handleChange('enable_holiday_detection', e.target.checked)}
                  />
                  <span className="toggle-slider"></span>
                  <span className="toggle-text">Holiday Detection</span>
                </label>
                <small>Automatically detect and skip trading on market holidays</small>
              </div>
            </div>
          )}
        </div>
      </div>
      <div className="form-actions">
        <button type="submit" className="btn btn-primary" disabled={loading}>
          {loading ? (
            <>
              <span className="spinner"></span>
              Applying System Configuration...
            </>
          ) : (
            <>
              💾 Save System Configuration
            </>
          )}
        </button>
        <div className="save-warning">
          ⚠️ System configuration changes will be applied immediately and may affect system performance
        </div>
      </div>
    </form>
  );
};

export default SystemConfig; 