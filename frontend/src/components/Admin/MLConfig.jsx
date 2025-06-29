import React, { useState, useEffect } from 'react';

const MLConfig = ({ config, onSave, loading }) => {
  const [formData, setFormData] = useState({
    ml_confidence_threshold: 0.7,
    news_confidence_threshold: 0.6,
    regime_confidence_threshold: 0.55,
    sentiment_weight: 0.3,
    technical_weight: 0.4,
    news_weight: 0.3,
    volume_weight: 0.2,
    regime_adjustment_factor: 0.1,
    model_retrain_frequency_days: 7,
    ensemble_model_weights: {
      xgboost: 0.4,
      lightgbm: 0.3,
      catboost: 0.3
    },
    enable_dynamic_risk_scoring: true,
    risk_score_decay_factor: 0.1
  });

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

  const handleEnsembleChange = (model, value) => {
    setFormData(prev => ({
      ...prev,
      ensemble_model_weights: {
        ...prev.ensemble_model_weights,
        [model]: parseFloat(value) || 0
      }
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSave(formData);
  };

  return (
    <form onSubmit={handleSubmit} className="config-form">
      <div className="form-header">
        <h3>🤖 ML Model Configuration</h3>
        <p>Configure ML thresholds, feature weights, and ensemble settings</p>
      </div>
      <div className="form-section">
        <h4>🎯 Confidence Thresholds</h4>
        <div className="form-row">
          <div className="form-group">
            <label>ML Confidence Threshold:</label>
            <input
              type="number"
              value={formData.ml_confidence_threshold}
              onChange={e => handleChange('ml_confidence_threshold', parseFloat(e.target.value))}
              min="0.1"
              max="1.0"
              step="0.01"
            />
            <small className="help-text">Minimum confidence for ML signals (0.1-1.0)</small>
          </div>
          <div className="form-group">
            <label>News Confidence Threshold:</label>
            <input
              type="number"
              value={formData.news_confidence_threshold}
              onChange={e => handleChange('news_confidence_threshold', parseFloat(e.target.value))}
              min="0.1"
              max="1.0"
              step="0.01"
            />
          </div>
          <div className="form-group">
            <label>Regime Confidence Threshold:</label>
            <input
              type="number"
              value={formData.regime_confidence_threshold}
              onChange={e => handleChange('regime_confidence_threshold', parseFloat(e.target.value))}
              min="0.1"
              max="1.0"
              step="0.01"
            />
          </div>
        </div>
      </div>
      <div className="form-section">
        <h4>⚖️ Feature Weights</h4>
        <div className="form-row">
          <div className="form-group">
            <label>Sentiment Weight:</label>
            <input
              type="number"
              value={formData.sentiment_weight}
              onChange={e => handleChange('sentiment_weight', parseFloat(e.target.value))}
              min="0"
              max="1"
              step="0.01"
            />
          </div>
          <div className="form-group">
            <label>Technical Weight:</label>
            <input
              type="number"
              value={formData.technical_weight}
              onChange={e => handleChange('technical_weight', parseFloat(e.target.value))}
              min="0"
              max="1"
              step="0.01"
            />
          </div>
          <div className="form-group">
            <label>News Weight:</label>
            <input
              type="number"
              value={formData.news_weight}
              onChange={e => handleChange('news_weight', parseFloat(e.target.value))}
              min="0"
              max="1"
              step="0.01"
            />
          </div>
          <div className="form-group">
            <label>Volume Weight:</label>
            <input
              type="number"
              value={formData.volume_weight}
              onChange={e => handleChange('volume_weight', parseFloat(e.target.value))}
              min="0"
              max="1"
              step="0.01"
            />
          </div>
        </div>
      </div>
      <div className="form-section">
        <h4>🔗 Ensemble Model Weights</h4>
        <div className="form-row">
          {Object.keys(formData.ensemble_model_weights).map(model => (
            <div className="form-group" key={model}>
              <label>{model.toUpperCase()} Weight:</label>
              <input
                type="number"
                value={formData.ensemble_model_weights[model]}
                onChange={e => handleEnsembleChange(model, e.target.value)}
                min="0"
                max="1"
                step="0.01"
              />
            </div>
          ))}
        </div>
      </div>
      <div className="form-section">
        <h4>🧮 Model Settings</h4>
        <div className="form-row">
          <div className="form-group">
            <label>Regime Adjustment Factor:</label>
            <input
              type="number"
              value={formData.regime_adjustment_factor}
              onChange={e => handleChange('regime_adjustment_factor', parseFloat(e.target.value))}
              min="0"
              max="1"
              step="0.01"
            />
          </div>
          <div className="form-group">
            <label>Model Retrain Frequency (days):</label>
            <input
              type="number"
              value={formData.model_retrain_frequency_days}
              onChange={e => handleChange('model_retrain_frequency_days', parseInt(e.target.value))}
              min="1"
              max="30"
            />
          </div>
        </div>
      </div>
      <div className="form-section">
        <h4>🛡️ Risk Scoring</h4>
        <div className="form-row">
          <div className="form-group">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={formData.enable_dynamic_risk_scoring}
                onChange={e => handleChange('enable_dynamic_risk_scoring', e.target.checked)}
              />
              Enable Dynamic Risk Scoring
            </label>
          </div>
          <div className="form-group">
            <label>Risk Score Decay Factor:</label>
            <input
              type="number"
              value={formData.risk_score_decay_factor}
              onChange={e => handleChange('risk_score_decay_factor', parseFloat(e.target.value))}
              min="0"
              max="1"
              step="0.01"
            />
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
              💾 Save ML Configuration
            </>
          )}
        </button>
        <div className="save-warning">
          ⚠️ Changes will be applied immediately to ML models
        </div>
      </div>
    </form>
  );
};

export default MLConfig; 