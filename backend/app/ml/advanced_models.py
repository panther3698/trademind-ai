# backend/app/ml/advanced_models.py
"""
TradeMind AI - Advanced ML Models and Ensemble System
Multiple state-of-the-art models for maximum trading accuracy
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import joblib
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Advanced ML Libraries
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve

# Deep Learning (Optional - install with: pip install torch)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available. Install with: pip install torch")

# Time Series Models
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("⚠️ Statsmodels not available. Install with: pip install statsmodels")

logger = logging.getLogger(__name__)

class LightGBMModel:
    """
    LightGBM - Often outperforms XGBoost on financial data
    Faster training, better handling of categorical features
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        
        # Optimized parameters for financial data
        self.params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'min_child_weight': 0.001,
            'min_split_gain': 0.02,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'verbosity': -1,
            'force_row_wise': True
        }
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Train LightGBM model"""
        try:
            # Store feature columns
            self.feature_columns = list(X_train.columns)
            
            # Scale features
            self.scaler = RobustScaler()  # Better for outliers in financial data
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Create LightGBM datasets
            train_data = lgb.Dataset(X_train_scaled, label=y_train)
            val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)
            
            # Train model
            self.model = lgb.train(
                self.params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            # Evaluate
            val_pred = self.model.predict(X_val_scaled)
            val_accuracy = accuracy_score(y_val, (val_pred > 0.5).astype(int))
            val_auc = roc_auc_score(y_val, val_pred)
            
            return {
                "model_type": "LightGBM",
                "validation_accuracy": val_accuracy,
                "validation_auc": val_auc,
                "feature_importance": dict(zip(self.feature_columns, self.model.feature_importance())),
                "num_trees": self.model.num_trees()
            }
            
        except Exception as e:
            logger.error(f"LightGBM training failed: {e}")
            return {"error": str(e)}
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X[self.feature_columns])
        return self.model.predict(X_scaled)

class CatBoostModel:
    """
    CatBoost - Excellent for categorical features and missing values
    Built-in regularization, handles overfitting well
    """
    
    def __init__(self):
        self.model = None
        self.feature_columns = None
        
        # CatBoost parameters
        self.params = {
            'objective': 'Logloss',
            'eval_metric': 'AUC',
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 8,
            'l2_leaf_reg': 3,
            'random_seed': 42,
            'logging_level': 'Silent',
            'early_stopping_rounds': 50,
            'use_best_model': True
        }
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Train CatBoost model"""
        try:
            self.feature_columns = list(X_train.columns)
            
            # CatBoost handles missing values and doesn't need scaling
            self.model = cb.CatBoostClassifier(**self.params)
            
            # Train with validation
            self.model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                verbose=False
            )
            
            # Evaluate
            val_pred = self.model.predict_proba(X_val)[:, 1]
            val_accuracy = accuracy_score(y_val, (val_pred > 0.5).astype(int))
            val_auc = roc_auc_score(y_val, val_pred)
            
            return {
                "model_type": "CatBoost",
                "validation_accuracy": val_accuracy,
                "validation_auc": val_auc,
                "feature_importance": dict(zip(self.feature_columns, self.model.feature_importances_)),
                "num_trees": self.model.tree_count_
            }
            
        except Exception as e:
            logger.error(f"CatBoost training failed: {e}")
            return {"error": str(e)}
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        return self.model.predict_proba(X[self.feature_columns])[:, 1]

class LSTMModel:
    """
    LSTM Neural Network - Excellent for time series patterns
    Captures long-term dependencies in price movements
    """
    
    def __init__(self, sequence_length: int = 30):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.sequence_length = sequence_length
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, LSTM model disabled")
            return
        
        # Model architecture parameters
        self.hidden_size = 64
        self.num_layers = 2
        self.dropout = 0.2
        self.learning_rate = 0.001
        self.batch_size = 64
        self.epochs = 100
    
    def _create_sequences(self, data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(labels[i])
        
        return np.array(X), np.array(y)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Train LSTM model"""
        if not TORCH_AVAILABLE:
            return {"error": "PyTorch not available"}
        
        try:
            self.feature_columns = list(X_train.columns)
            
            # Scale features
            self.scaler = MinMaxScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Create sequences
            X_train_seq, y_train_seq = self._create_sequences(X_train_scaled, y_train.values)
            X_val_seq, y_val_seq = self._create_sequences(X_val_scaled, y_val.values)
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train_seq)
            y_train_tensor = torch.FloatTensor(y_train_seq)
            X_val_tensor = torch.FloatTensor(X_val_seq)
            y_val_tensor = torch.FloatTensor(y_val_seq)
            
            # Create data loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            
            # Define LSTM model
            class LSTMClassifier(nn.Module):
                def __init__(self, input_size, hidden_size, num_layers, dropout):
                    super(LSTMClassifier, self).__init__()
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                       batch_first=True, dropout=dropout)
                    self.fc = nn.Linear(hidden_size, 1)
                    self.sigmoid = nn.Sigmoid()
                
                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    output = self.fc(lstm_out[:, -1, :])  # Use last output
                    return self.sigmoid(output)
            
            # Initialize model
            input_size = X_train_seq.shape[2]
            self.model = LSTMClassifier(input_size, self.hidden_size, self.num_layers, self.dropout)
            
            # Training setup
            criterion = nn.BCELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            # Training loop
            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0
            
            for epoch in range(self.epochs):
                self.model.train()
                total_loss = 0
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                # Validation
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor).squeeze()
                    val_loss = criterion(val_outputs, y_val_tensor)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
            
            # Final evaluation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val_tensor).squeeze().numpy()
                val_accuracy = accuracy_score(y_val_seq, (val_pred > 0.5).astype(int))
                val_auc = roc_auc_score(y_val_seq, val_pred)
            
            return {
                "model_type": "LSTM",
                "validation_accuracy": val_accuracy,
                "validation_auc": val_auc,
                "sequence_length": self.sequence_length,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers
            }
            
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            return {"error": str(e)}
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with LSTM"""
        if not TORCH_AVAILABLE or self.model is None:
            raise ValueError("Model not available")
        
        # Scale and create sequences
        X_scaled = self.scaler.transform(X[self.feature_columns])
        
        if len(X_scaled) < self.sequence_length:
            # Pad with last available values if insufficient data
            padding = np.repeat(X_scaled[-1:], self.sequence_length - len(X_scaled), axis=0)
            X_scaled = np.vstack([padding, X_scaled])
        
        # Get the last sequence
        X_seq = X_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        X_tensor = torch.FloatTensor(X_seq)
        
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(X_tensor).squeeze().numpy()
        
        return np.array([prediction])

class EnsembleModel:
    """
    Advanced Ensemble combining multiple models for maximum accuracy
    Uses stacking, voting, and dynamic weighting
    """
    
    def __init__(self):
        self.models = {}
        self.stacking_model = None
        self.voting_model = None
        self.feature_columns = None
        self.model_weights = {}
        
        # Initialize individual models
        self.models['xgboost'] = None  # Will be set externally
        self.models['lightgbm'] = LightGBMModel()
        self.models['catboost'] = CatBoostModel()
        if TORCH_AVAILABLE:
            self.models['lstm'] = LSTMModel()
        
        # Classical models for diversity
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        )
        self.models['logistic'] = LogisticRegression(random_state=42, max_iter=1000)
        self.models['svm'] = SVC(probability=True, random_state=42)
    
    def train_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series, 
                      X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Train all models and create ensemble"""
        try:
            self.feature_columns = list(X_train.columns)
            individual_results = {}
            base_models = []
            model_predictions = {}
            
            logger.info("🤖 Training ensemble models...")
            
            # Train individual models
            for name, model in self.models.items():
                try:
                    logger.info(f"Training {name}...")
                    
                    if name in ['lightgbm', 'catboost', 'lstm']:
                        # Advanced models with custom training
                        result = model.train(X_train, y_train, X_val, y_val)
                        individual_results[name] = result
                        
                        # Get predictions for stacking
                        val_pred = model.predict(X_val)
                        model_predictions[name] = val_pred
                        
                    elif name in ['random_forest', 'logistic', 'svm']:
                        # Classical sklearn models
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_val_scaled = scaler.transform(X_val)
                        
                        model.fit(X_train_scaled, y_train)
                        val_pred = model.predict_proba(X_val_scaled)[:, 1]
                        
                        accuracy = accuracy_score(y_val, (val_pred > 0.5).astype(int))
                        auc = roc_auc_score(y_val, val_pred)
                        
                        individual_results[name] = {
                            "model_type": name,
                            "validation_accuracy": accuracy,
                            "validation_auc": auc
                        }
                        
                        model_predictions[name] = val_pred
                        base_models.append((name, model))
                        
                        # Store scaler
                        setattr(model, 'scaler', scaler)
                
                except Exception as e:
                    logger.error(f"Failed to train {name}: {e}")
                    continue
            
            # Create stacking ensemble
            if len(model_predictions) >= 3:
                logger.info("🔗 Creating stacking ensemble...")
                
                # Prepare stacking features
                stacking_features = np.column_stack(list(model_predictions.values()))
                
                # Train meta-learner
                self.stacking_model = LogisticRegression(random_state=42)
                self.stacking_model.fit(stacking_features, y_val)
                
                # Evaluate stacking
                stacking_pred = self.stacking_model.predict_proba(stacking_features)[:, 1]
                stacking_accuracy = accuracy_score(y_val, (stacking_pred > 0.5).astype(int))
                stacking_auc = roc_auc_score(y_val, stacking_pred)
                
                individual_results['stacking_ensemble'] = {
                    "model_type": "stacking_ensemble",
                    "validation_accuracy": stacking_accuracy,
                    "validation_auc": stacking_auc
                }
            
            # Calculate dynamic weights based on performance
            self._calculate_model_weights(individual_results)
            
            # Find best performing model
            best_model = max(individual_results.items(), 
                           key=lambda x: x[1].get('validation_auc', 0))
            
            ensemble_summary = {
                "ensemble_results": individual_results,
                "best_individual_model": best_model[0],
                "best_individual_auc": best_model[1].get('validation_auc', 0),
                "model_weights": self.model_weights,
                "ensemble_size": len(individual_results)
            }
            
            logger.info(f"✅ Ensemble training complete!")
            logger.info(f"🏆 Best model: {best_model[0]} (AUC: {best_model[1].get('validation_auc', 0):.3f})")
            
            return ensemble_summary
            
        except Exception as e:
            logger.error(f"Ensemble training failed: {e}")
            return {"error": str(e)}
    
    def _calculate_model_weights(self, results: Dict[str, Dict]):
        """Calculate dynamic weights based on model performance"""
        try:
            # Weight based on AUC scores
            auc_scores = {name: result.get('validation_auc', 0.5) 
                         for name, result in results.items() 
                         if not result.get('error')}
            
            if not auc_scores:
                return
            
            # Normalize weights (higher AUC = higher weight)
            total_auc = sum(auc_scores.values())
            self.model_weights = {name: auc / total_auc 
                                for name, auc in auc_scores.items()}
            
            # Boost best performers
            best_model = max(auc_scores.items(), key=lambda x: x[1])
            if best_model[1] > 0.7:  # If AUC > 0.7, give extra weight
                self.model_weights[best_model[0]] *= 1.2
                
                # Renormalize
                total_weight = sum(self.model_weights.values())
                self.model_weights = {name: weight / total_weight 
                                    for name, weight in self.model_weights.items()}
                
        except Exception as e:
            logger.error(f"Weight calculation failed: {e}")
            # Equal weights fallback
            model_count = len([r for r in results.values() if not r.get('error')])
            if model_count > 0:
                equal_weight = 1.0 / model_count
                self.model_weights = {name: equal_weight for name in results.keys() 
                                    if not results[name].get('error')}
    
    def predict_ensemble(self, X: pd.DataFrame) -> Dict[str, float]:
        """Make ensemble predictions"""
        try:
            predictions = {}
            
            # Get predictions from each model
            for name, model in self.models.items():
                try:
                    if name in ['lightgbm', 'catboost', 'lstm']:
                        pred = model.predict(X)
                    elif name in ['random_forest', 'logistic', 'svm']:
                        if hasattr(model, 'scaler'):
                            X_scaled = model.scaler.transform(X[self.feature_columns])
                            pred = model.predict_proba(X_scaled)[:, 1]
                        else:
                            continue
                    else:
                        continue
                    
                    predictions[name] = pred[0] if len(pred) > 0 else 0.5
                    
                except Exception as e:
                    logger.debug(f"Prediction failed for {name}: {e}")
                    continue
            
            if not predictions:
                return {"ensemble_prediction": 0.5, "individual_predictions": {}}
            
            # Weighted average
            weighted_pred = sum(pred * self.model_weights.get(name, 0) 
                              for name, pred in predictions.items())
            
            # Stacking prediction if available
            stacking_pred = 0.5
            if self.stacking_model and len(predictions) >= 3:
                try:
                    stacking_features = np.array(list(predictions.values())).reshape(1, -1)
                    stacking_pred = self.stacking_model.predict_proba(stacking_features)[0, 1]
                except Exception as e:
                    logger.debug(f"Stacking prediction failed: {e}")
            
            # Final ensemble (combine weighted and stacking)
            final_prediction = (weighted_pred * 0.7) + (stacking_pred * 0.3)
            
            return {
                "ensemble_prediction": final_prediction,
                "weighted_prediction": weighted_pred,
                "stacking_prediction": stacking_pred,
                "individual_predictions": predictions,
                "model_weights": self.model_weights
            }
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            return {"ensemble_prediction": 0.5, "error": str(e)}

class MarketRegimeClassifier:
    """
    Advanced market regime classification using Hidden Markov Models
    Identifies: Trending Bull, Trending Bear, Sideways, High Vol, Low Vol
    """
    
    def __init__(self):
        self.hmm_model = None
        self.regime_features = [
            'returns', 'volatility', 'volume_ratio', 'rsi', 'macd'
        ]
        
        try:
            from hmmlearn import hmm
            self.HMM_AVAILABLE = True
        except ImportError:
            self.HMM_AVAILABLE = False
            logger.warning("hmmlearn not available. Install with: pip install hmmlearn")
    
    def train_regime_model(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Train Hidden Markov Model for regime detection"""
        if not self.HMM_AVAILABLE:
            return {"error": "hmmlearn not available"}
        
        try:
            from hmmlearn import hmm
            
            # Prepare features
            features = self._prepare_regime_features(market_data)
            
            if len(features) < 100:
                return {"error": "Insufficient data for regime modeling"}
            
            # Train HMM with 5 hidden states (regimes)
            self.hmm_model = hmm.GaussianHMM(
                n_components=5,
                covariance_type="full",
                n_iter=100,
                random_state=42
            )
            
            self.hmm_model.fit(features)
            
            # Predict regimes for training data
            hidden_states = self.hmm_model.predict(features)
            
            # Analyze regime characteristics
            regime_analysis = self._analyze_regimes(features, hidden_states)
            
            return {
                "model_type": "HMM_RegimeClassifier",
                "n_regimes": 5,
                "regime_analysis": regime_analysis,
                "log_likelihood": self.hmm_model.score(features)
            }
            
        except Exception as e:
            logger.error(f"Regime model training failed: {e}")
            return {"error": str(e)}
    
    def _prepare_regime_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for regime classification"""
        try:
            features = []
            
            # Returns and volatility
            data['returns'] = data['close'].pct_change()
            data['volatility'] = data['returns'].rolling(20).std()
            data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
            
            # Technical indicators
            data['rsi'] = self._calculate_rsi(data['close'])
            data['macd'] = self._calculate_macd(data['close'])
            
            # Select features
            feature_data = data[self.regime_features].dropna()
            
            return feature_data.values
            
        except Exception as e:
            logger.error(f"Regime feature preparation failed: {e}")
            return np.array([])
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD"""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        return ema12 - ema26
    
    def _analyze_regimes(self, features: np.ndarray, states: np.ndarray) -> Dict:
        """Analyze characteristics of each regime"""
        regime_analysis = {}
        
        for state in range(5):
            state_mask = states == state
            state_features = features[state_mask]
            
            if len(state_features) > 0:
                regime_analysis[f"regime_{state}"] = {
                    "frequency": np.sum(state_mask) / len(states),
                    "avg_returns": np.mean(state_features[:, 0]),
                    "avg_volatility": np.mean(state_features[:, 1]),
                    "avg_volume_ratio": np.mean(state_features[:, 2]),
                    "avg_rsi": np.mean(state_features[:, 3]),
                    "description": self._classify_regime(state_features)
                }
        
        return regime_analysis
    
    def _classify_regime(self, features: np.ndarray) -> str:
        """Classify regime based on feature characteristics"""
        avg_returns = np.mean(features[:, 0])
        avg_volatility = np.mean(features[:, 1])
        avg_rsi = np.mean(features[:, 3])
        
        if avg_returns > 0.002 and avg_rsi > 60:
            return "Trending_Bullish"
        elif avg_returns < -0.002 and avg_rsi < 40:
            return "Trending_Bearish"
        elif avg_volatility > 0.02:
            return "High_Volatility"
        elif avg_volatility < 0.01:
            return "Low_Volatility"
        else:
            return "Sideways_Choppy"
    
    def predict_current_regime(self, recent_data: pd.DataFrame) -> Dict[str, Any]:
        """Predict current market regime"""
        if not self.HMM_AVAILABLE or self.hmm_model is None:
            return {"error": "Regime model not available"}
        
        try:
            features = self._prepare_regime_features(recent_data)
            
            if len(features) < 20:  # Need minimum data
                return {"error": "Insufficient recent data"}
            
            # Predict regime for recent period
            recent_features = features[-20:]  # Last 20 periods
            predicted_states = self.hmm_model.predict(recent_features)
            
            # Get most common recent regime
            current_regime = np.bincount(predicted_states).argmax()
            
            # Get regime probabilities
            state_probs = self.hmm_model.predict_proba(recent_features)
            current_probs = state_probs[-1]  # Latest probabilities
            
            return {
                "current_regime": int(current_regime),
                "regime_probabilities": current_probs.tolist(),
                "confidence": np.max(current_probs),
                "recent_regimes": predicted_states.tolist()
            }
            
        except Exception as e:
            logger.error(f"Regime prediction failed: {e}")
            return {"error": str(e)}

# Model factory for easy instantiation
class ModelFactory:
    """Factory for creating and managing different model types"""
    
    @staticmethod
    def create_model(model_type: str, **kwargs):
        """Create model instance by type"""
        models = {
            'xgboost': lambda: None,  # Implemented elsewhere
            'lightgbm': LightGBMModel,
            'catboost': CatBoostModel,
            'lstm': LSTMModel,
            'ensemble': EnsembleModel,
            'regime_classifier': MarketRegimeClassifier
        }
        
        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = models[model_type]
        if model_class is None:
            return None
        
        return model_class(**kwargs)
    
    @staticmethod
    def get_recommended_models() -> List[str]:
        """Get list of recommended models for ensemble"""
        base_models = ['lightgbm', 'catboost']
        
        if TORCH_AVAILABLE:
            base_models.append('lstm')
        
        return base_models

# Performance comparison utility
class ModelComparison:
    """Compare performance of different models"""
    
    @staticmethod
    def compare_models(results: Dict[str, Dict]) -> pd.DataFrame:
        """Create comparison dataframe of model results"""
        comparison_data = []
        
        for model_name, result in results.items():
            if result.get('error'):
                continue
                
            comparison_data.append({
                'Model': model_name,
                'Accuracy': result.get('validation_accuracy', 0),
                'AUC': result.get('validation_auc', 0),
                'Type': result.get('model_type', model_name)
            })
        
        df = pd.DataFrame(comparison_data)
        if len(df) > 0:
            df = df.sort_values('AUC', ascending=False)
        
        return df
    
    @staticmethod
    def recommend_best_ensemble(results: Dict[str, Dict], 
                               min_auc: float = 0.65) -> List[str]:
        """Recommend best models for ensemble based on performance"""
        good_models = []
        
        for model_name, result in results.items():
            auc = result.get('validation_auc', 0)
            if auc >= min_auc and not result.get('error'):
                good_models.append((model_name, auc))
        
        # Sort by AUC and return top models
        good_models.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 5 models or all good models if less than 5
        return [model[0] for model in good_models[:5]]


if __name__ == "__main__":
    # Example usage
    print("TradeMind AI - Advanced ML Models")
    print("Available models:")
    
    for model_type in ['lightgbm', 'catboost', 'lstm', 'ensemble']:
        try:
            model = ModelFactory.create_model(model_type)
            status = "✅ Available" if model is not None else "❌ Not available"
            print(f"  {model_type}: {status}")
        except Exception as e:
            print(f"  {model_type}: ❌ Error - {e}")
