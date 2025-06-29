"""
Advanced Ensemble Model Loader Patch
Enables AdvancedEnsembleModel to load XGBoost bridge models
"""

import joblib
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class AdvancedEnsembleModelPatch:
    """Patch for AdvancedEnsembleModel to load XGBoost bridges"""
    
    @staticmethod
    def load_xgboost_bridge(models_dir="models"):
        """Load XGBoost bridge into advanced ensemble structure"""
        try:
            models_path = Path(models_dir)
            bridge_file = models_path / "enhanced_ensemble_model_v2.0.pkl"
            
            if bridge_file.exists():
                bridge_data = joblib.load(bridge_file)
                
                if bridge_data.get('bridge_created'):
                    logger.info(f"SUCCESS: Loading XGBoost bridge: {bridge_data.get('source_model_id')}")
                    accuracy = bridge_data.get('individual_performance', {}).get('xgboost', {}).get('accuracy', 'N/A')
                    logger.info(f"Model accuracy: {accuracy}")
                    return bridge_data
                    
            return None
            
        except Exception as e:
            logger.error(f"Bridge loading failed: {e}")
            return None
    
    @staticmethod
    def predict_with_bridge(bridge_data, features):
        """Make predictions using the XGBoost bridge"""
        try:
            xgb_model = bridge_data['xgboost_model']
            scaler = bridge_data['scaler']
            feature_columns = bridge_data['feature_columns']
            
            # Prepare features
            feature_vector = []
            for col in feature_columns:
                feature_vector.append(features.get(col, 0.0))
            
            X = np.array(feature_vector).reshape(1, -1)
            
            if scaler:
                X = scaler.transform(X)
            
            # Get prediction
            if hasattr(xgb_model, 'predict_proba'):
                prediction = xgb_model.predict_proba(X)[0, 1]
            else:
                prediction = xgb_model.predict(X)[0]
            
            return float(prediction)
            
        except Exception as e:
            logger.error(f"Bridge prediction failed: {e}")
            return 0.5
