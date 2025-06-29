# ================================================================
# backend/app/ml/optimized_model_loader.py
# Optimized Model Loading with Performance Metrics
# ================================================================

import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ModelPerformanceMetrics:
    """Model performance metrics for tracking"""
    model_id: str
    load_time_ms: float
    prediction_time_ms: float
    accuracy: float
    total_predictions: int
    successful_predictions: int
    last_loaded: datetime
    model_size_mb: float

class OptimizedModelLoader:
    """
    Optimized model loader with performance tracking and streamlined XGBoost loading
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.loaded_models = {}
        self.performance_metrics = {}
        self.model_cache = {}
        
        # Performance tracking
        self.total_load_time = 0.0
        self.total_prediction_time = 0.0
        self.total_predictions = 0
        
        logger.info("✅ Optimized model loader initialized")
    
    def load_xgboost_model(self, model_name: str = "enhanced_ensemble_model_v2.0.pkl") -> Optional[Dict[str, Any]]:
        """
        Load XGBoost model with performance tracking
        
        Args:
            model_name: Name of the model file
            
        Returns:
            Model data dictionary or None if loading fails
        """
        start_time = time.time()
        
        try:
            model_path = self.models_dir / model_name
            
            if not model_path.exists():
                logger.warning(f"Model file not found: {model_path}")
                return None
            
            # Load model with performance tracking
            logger.info(f"🔄 Loading XGBoost model: {model_name}")
            model_data = joblib.load(model_path)
            
            # Calculate load time
            load_time_ms = (time.time() - start_time) * 1000
            self.total_load_time += load_time_ms
            
            # Get model size
            model_size_mb = model_path.stat().st_size / (1024 * 1024)
            
            # Extract model information
            model_id = model_data.get('source_model_id', 'unknown')
            accuracy = model_data.get('individual_performance', {}).get('xgboost', {}).get('accuracy', 0.0)
            
            # Store performance metrics
            self.performance_metrics[model_id] = ModelPerformanceMetrics(
                model_id=model_id,
                load_time_ms=load_time_ms,
                prediction_time_ms=0.0,
                accuracy=accuracy,
                total_predictions=0,
                successful_predictions=0,
                last_loaded=datetime.now(),
                model_size_mb=model_size_mb
            )
            
            # Cache the model
            self.model_cache[model_id] = model_data
            
            logger.info(f"✅ XGBoost model loaded successfully:")
            logger.info(f"  • Model ID: {model_id}")
            logger.info(f"  • Load time: {load_time_ms:.1f}ms")
            logger.info(f"  • Model size: {model_size_mb:.1f}MB")
            logger.info(f"  • Accuracy: {accuracy:.1%}")
            
            return model_data
            
        except Exception as e:
            load_time_ms = (time.time() - start_time) * 1000
            logger.error(f"❌ XGBoost model loading failed: {e} (took {load_time_ms:.1f}ms)")
            return None
    
    def predict_with_performance_tracking(self, model_id: str, features: Dict[str, float]) -> Tuple[float, float]:
        """
        Make prediction with performance tracking
        
        Args:
            model_id: ID of the loaded model
            features: Feature dictionary
            
        Returns:
            Tuple of (prediction, prediction_time_ms)
        """
        start_time = time.time()
        
        try:
            if model_id not in self.model_cache:
                logger.error(f"Model {model_id} not found in cache")
                return 0.5, 0.0
            
            model_data = self.model_cache[model_id]
            xgb_model = model_data['xgboost_model']
            scaler = model_data.get('scaler')
            feature_columns = model_data['feature_columns']
            
            # Prepare feature vector
            feature_vector = []
            for col in feature_columns:
                feature_vector.append(features.get(col, 0.0))
            
            X = np.array(feature_vector).reshape(1, -1)
            
            # Apply scaling if available
            if scaler:
                X = scaler.transform(X)
            
            # Make prediction
            if hasattr(xgb_model, 'predict_proba'):
                prediction = xgb_model.predict_proba(X)[0, 1]
            else:
                prediction = xgb_model.predict(X)[0]
            
            # Calculate prediction time
            prediction_time_ms = (time.time() - start_time) * 1000
            self.total_prediction_time += prediction_time_ms
            self.total_predictions += 1
            
            # Update performance metrics
            if model_id in self.performance_metrics:
                metrics = self.performance_metrics[model_id]
                metrics.prediction_time_ms = prediction_time_ms
                metrics.total_predictions += 1
                metrics.successful_predictions += 1
            
            return float(prediction), prediction_time_ms
            
        except Exception as e:
            prediction_time_ms = (time.time() - start_time) * 1000
            logger.error(f"❌ Prediction failed: {e} (took {prediction_time_ms:.1f}ms)")
            
            # Update failed prediction count
            if model_id in self.performance_metrics:
                self.performance_metrics[model_id].total_predictions += 1
            
            return 0.5, prediction_time_ms
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            summary = {
                "total_load_time_ms": self.total_load_time,
                "total_prediction_time_ms": self.total_prediction_time,
                "total_predictions": self.total_predictions,
                "average_load_time_ms": self.total_load_time / max(len(self.performance_metrics), 1),
                "average_prediction_time_ms": self.total_prediction_time / max(self.total_predictions, 1),
                "models_loaded": len(self.model_cache),
                "model_details": {}
            }
            
            # Add individual model metrics
            for model_id, metrics in self.performance_metrics.items():
                summary["model_details"][model_id] = {
                    "load_time_ms": metrics.load_time_ms,
                    "average_prediction_time_ms": metrics.prediction_time_ms / max(metrics.total_predictions, 1),
                    "accuracy": metrics.accuracy,
                    "total_predictions": metrics.total_predictions,
                    "successful_predictions": metrics.successful_predictions,
                    "success_rate": metrics.successful_predictions / max(metrics.total_predictions, 1),
                    "model_size_mb": metrics.model_size_mb,
                    "last_loaded": metrics.last_loaded.isoformat()
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate performance summary: {e}")
            return {"error": str(e)}
    
    def clear_cache(self):
        """Clear model cache to free memory"""
        try:
            self.model_cache.clear()
            logger.info("🧹 Model cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for the model loader"""
        try:
            return {
                "status": "healthy",
                "models_loaded": len(self.model_cache),
                "total_predictions": self.total_predictions,
                "average_prediction_time_ms": self.total_prediction_time / max(self.total_predictions, 1),
                "cache_size_mb": sum(m.model_size_mb for m in self.performance_metrics.values()),
                "last_activity": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "last_activity": datetime.now().isoformat()
            }

# Global instance for easy access
_optimized_loader = None

def get_optimized_loader() -> OptimizedModelLoader:
    """Get global optimized model loader instance"""
    global _optimized_loader
    if _optimized_loader is None:
        _optimized_loader = OptimizedModelLoader()
    return _optimized_loader 