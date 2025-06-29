import React, { useState, useEffect } from 'react';

const TradingConfig = ({ config, onSave, loading }) => {
  const [formData, setFormData] = useState({
    max_capital_per_trade: 100000,
    default_stop_loss_percentage: 3.0,
    default_target_percentage: 5.0,
    max_signals_per_day: 10,
    min_confidence_threshold: 0.75,
    max_daily_loss_limit: 500000,
    max_position_size_percentage: 20.0,
    risk_reward_ratio_min: 1.5,
    enable_auto_trading: true,
    trading_mode: 'CONSERVATIVE'
  });

  const [errors, setErrors] = useState({});

  useEffect(() => {
    if (config) {
      setFormData(prev => ({ ...prev, ...config }));
    }
  }, [config]);

  const validateField = (field, value) => {
    const newErrors = { ...errors };
    
    switch (field) {
      case 'max_capital_per_trade':
        if (value < 1000 || value > 10000000) {
          newErrors[field] = 'Capital must be between ₹1,000 and ₹1,00,00,000';
        } else {
          delete newErrors[field];
        }
        break;
      case 'default_stop_loss_percentage':
        if (value < 0.5 || value > 15) {
          newErrors[field] = 'Stop loss must be between 0.5% and 15%';
        } else {
          delete newErrors[field];
        }
        break;
      case 'default_target_percentage':
        if (value < 1 || value > 50) {
          newErrors[field] = 'Target must be between 1% and 50%';
        } else if (value <= formData.default_stop_loss_percentage) {
          newErrors[field] = 'Target must be higher than stop loss';
        } else {
          delete newErrors[field];
        }
        break;
      case 'min_confidence_threshold':
        if (value < 0.1 || value > 1.0) {
          newErrors[field] = 'Confidence must be between 10% and 100%';
        } else {
          delete newErrors[field];
        }
        break;
    }
    
    setErrors(newErrors);
  };

  const handleChange = (field, value) => {
    const numericValue = ['max_capital_per_trade', 'default_stop_loss_percentage', 'default_target_percentage', 
                         'max_signals_per_day', 'min_confidence_threshold', 'max_daily_loss_limit',
                         'max_position_size_percentage', 'risk_reward_ratio_min'].includes(field) 
                         ? parseFloat(value) || 0 : value;
    
    setFormData(prev => ({
      ...prev,
      [field]: numericValue
    }));
    
    if (typeof numericValue === 'number') {
      validateField(field, numericValue);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    // Validate all fields
    Object.keys(formData).forEach(field => {
      if (typeof formData[field] === 'number') {
        validateField(field, formData[field]);
      }
    });
    
    // Check if there are any errors
    if (Object.keys(errors).length > 0) {
      alert('Please fix validation errors before saving');
      return;
    }
    
    onSave(formData);
  };

  const calculateRiskReward = () => {
    const rr = formData.default_target_percentage / formData.default_stop_loss_percentage;
    return rr.toFixed(2);
  };

  const getTradeSize = () => {
    // Example calculation for 1% risk
    const riskAmount = formData.max_capital_per_trade * 0.01;
    const stopLossAmount = formData.max_capital_per_trade * (formData.default_stop_loss_percentage / 100);
    const quantity = Math.floor(riskAmount / stopLossAmount);
    return quantity > 0 ? quantity : 1;
  };

  return (
    <form onSubmit={handleSubmit} className="config-form">
      <div className="form-header">
        <h3>💰 Trading Configuration</h3>
        <p>Configure your trading parameters, risk management, and position sizing</p>
      </div>
      
      <div className="form-section">
        <h4>💵 Capital Management</h4>
        
        <div className="form-row">
          <div className="form-group">
            <label>
              Max Capital Per Trade:
              <span className="tooltip" title="Maximum amount to invest in a single trade">ℹ️</span>
            </label>
            <div className="input-group">
              <span className="input-prefix">₹</span>
              <input
                type="number"
                value={formData.max_capital_per_trade}
                onChange={(e) => handleChange('max_capital_per_trade', e.target.value)}
                min="1000"
                max="10000000"
                step="1000"
                className={errors.max_capital_per_trade ? 'error' : ''}
              />
            </div>
            {errors.max_capital_per_trade && (
              <span className="error-message">{errors.max_capital_per_trade}</span>
            )}
            <small className="help-text">
              Recommended: ₹50,000 - ₹2,00,000 per trade
            </small>
          </div>

          <div className="form-group">
            <label>
              Max Daily Loss Limit:
              <span className="tooltip" title="Maximum loss allowed per day before stopping trading">ℹ️</span>
            </label>
            <div className="input-group">
              <span className="input-prefix">₹</span>
              <input
                type="number"
                value={formData.max_daily_loss_limit}
                onChange={(e) => handleChange('max_daily_loss_limit', e.target.value)}
                min="10000"
                max="5000000"
                step="10000"
              />
            </div>
            <small className="help-text">
              System will stop trading if daily loss exceeds this amount
            </small>
          </div>
        </div>
      </div>

      <div className="form-section">
        <h4>📊 Risk Management</h4>
        
        <div className="form-row">
          <div className="form-group">
            <label>
              Default Stop Loss (%):
              <span className="tooltip" title="Default stop loss percentage for all trades">ℹ️</span>
            </label>
            <div className="input-group">
              <input
                type="number"
                value={formData.default_stop_loss_percentage}
                onChange={(e) => handleChange('default_stop_loss_percentage', e.target.value)}
                min="0.5"
                max="15"
                step="0.1"
                className={errors.default_stop_loss_percentage ? 'error' : ''}
              />
              <span className="input-suffix">%</span>
            </div>
            {errors.default_stop_loss_percentage && (
              <span className="error-message">{errors.default_stop_loss_percentage}</span>
            )}
          </div>

          <div className="form-group">
            <label>
              Default Target (%):
              <span className="tooltip" title="Default target percentage for all trades">ℹ️</span>
            </label>
            <div className="input-group">
              <input
                type="number"
                value={formData.default_target_percentage}
                onChange={(e) => handleChange('default_target_percentage', e.target.value)}
                min="1"
                max="50"
                step="0.1"
                className={errors.default_target_percentage ? 'error' : ''}
              />
              <span className="input-suffix">%</span>
            </div>
            {errors.default_target_percentage && (
              <span className="error-message">{errors.default_target_percentage}</span>
            )}
          </div>
        </div>

        <div className="risk-metrics">
          <div className="metric">
            <span className="metric-label">Risk-Reward Ratio:</span>
            <span className="metric-value">1:{calculateRiskReward()}</span>
          </div>
          <div className="metric">
            <span className="metric-label">Suggested Position Size:</span>
            <span className="metric-value">{getTradeSize()} shares</span>
          </div>
        </div>
      </div>

      <div className="form-section">
        <h4>🎯 Signal Configuration</h4>
        
        <div className="form-row">
          <div className="form-group">
            <label>Max Signals Per Day:</label>
            <select
              value={formData.max_signals_per_day}
              onChange={(e) => handleChange('max_signals_per_day', parseInt(e.target.value))}
            >
              <option value={3}>3 signals (Conservative)</option>
              <option value={5}>5 signals (Balanced)</option>
              <option value={10}>10 signals (Active)</option>
              <option value={15}>15 signals (Aggressive)</option>
            </select>
          </div>

          <div className="form-group">
            <label>
              Min Confidence Threshold:
              <span className="tooltip" title="Minimum ML confidence required to generate a signal">ℹ️</span>
            </label>
            <div className="input-group">
              <input
                type="number"
                value={formData.min_confidence_threshold}
                onChange={(e) => handleChange('min_confidence_threshold', e.target.value)}
                min="0.1"
                max="1.0"
                step="0.01"
                className={errors.min_confidence_threshold ? 'error' : ''}
              />
              <span className="input-suffix">%</span>
            </div>
            {errors.min_confidence_threshold && (
              <span className="error-message">{errors.min_confidence_threshold}</span>
            )}
            <small className="help-text">
              Higher threshold = fewer but higher quality signals
            </small>
          </div>
        </div>
      </div>

      <div className="form-section">
        <h4>⚙️ Trading Mode</h4>
        
        <div className="form-row">
          <div className="form-group">
            <label>Trading Mode:</label>
            <select
              value={formData.trading_mode}
              onChange={(e) => handleChange('trading_mode', e.target.value)}
            >
              <option value="CONSERVATIVE">Conservative (Lower risk, stable returns)</option>
              <option value="MODERATE">Moderate (Balanced risk-reward)</option>
              <option value="AGGRESSIVE">Aggressive (Higher risk, higher returns)</option>
            </select>
          </div>

          <div className="form-group">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={formData.enable_auto_trading}
                onChange={(e) => handleChange('enable_auto_trading', e.target.checked)}
              />
              Enable Automatic Trading
            </label>
            <small className="help-text">
              When enabled, approved signals will be executed automatically
            </small>
          </div>
        </div>
      </div>

      <div className="form-actions">
        <button type="submit" className="btn btn-primary" disabled={loading}>
          {loading ? (
            <>
              <span className="spinner"></span>
              Saving...
            </>
          ) : (
            <>
              💾 Save Trading Configuration
            </>
          )}
        </button>
        
        <div className="save-warning">
          ⚠️ Changes will be applied immediately to live trading
        </div>
      </div>
    </form>
  );
};

export default TradingConfig; 