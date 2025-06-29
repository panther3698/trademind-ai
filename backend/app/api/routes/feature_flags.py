"""
Feature Flags API Routes

Provides REST API endpoints for managing feature flags and configurations.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from app.core.config.feature_flags import (
    FeatureFlagsManager, FeatureCategory, SignalGenerationConfig, NewsSourceConfig,
    get_feature_flags_manager
)
from app.api.dependencies import get_initialized_service_manager
from app.core.services.service_manager import CorrectedServiceManager

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/api/feature-flags")
async def get_all_feature_flags(
    category: Optional[FeatureCategory] = Query(None, description="Filter by category"),
    service_manager: CorrectedServiceManager = Depends(get_initialized_service_manager)
):
    """Get all feature flags or filter by category"""
    try:
        feature_manager = get_feature_flags_manager()
        
        if category:
            flags = feature_manager.get_flags_by_category(category)
        else:
            flags = feature_manager.get_all_flags()
        
        # Convert to serializable format
        result = {}
        for name, flag in flags.items():
            result[name] = {
                "name": flag.name,
                "category": flag.category.value,
                "description": flag.description,
                "enabled": flag.enabled,
                "default_value": flag.default_value,
                "configurable": flag.configurable,
                "dependencies": flag.dependencies,
                "last_modified": flag.last_modified.isoformat() if flag.last_modified else None,
                "modified_by": flag.modified_by
            }
        
        return {
            "flags": result,
            "total_count": len(result),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get feature flags: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/feature-flags/{flag_name}")
async def get_feature_flag(
    flag_name: str,
    service_manager: CorrectedServiceManager = Depends(get_initialized_service_manager)
):
    """Get specific feature flag details"""
    try:
        feature_manager = get_feature_flags_manager()
        flag = feature_manager.get_flag(flag_name)
        
        if not flag:
            raise HTTPException(status_code=404, detail=f"Feature flag '{flag_name}' not found")
        
        return {
            "name": flag.name,
            "category": flag.category.value,
            "description": flag.description,
            "enabled": flag.enabled,
            "default_value": flag.default_value,
            "configurable": flag.configurable,
            "dependencies": flag.dependencies,
            "last_modified": flag.last_modified.isoformat() if flag.last_modified else None,
            "modified_by": flag.modified_by
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get feature flag {flag_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/feature-flags/{flag_name}")
async def set_feature_flag(
    flag_name: str,
    enabled: bool,
    modified_by: str = Query("api", description="User who made the change"),
    service_manager: CorrectedServiceManager = Depends(get_initialized_service_manager)
):
    """Set a feature flag value"""
    try:
        feature_manager = get_feature_flags_manager()
        success = feature_manager.set_flag(flag_name, enabled, modified_by)
        
        if not success:
            raise HTTPException(status_code=400, detail=f"Failed to set feature flag '{flag_name}'")
        
        return {
            "success": True,
            "flag_name": flag_name,
            "enabled": enabled,
            "modified_by": modified_by,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to set feature flag {flag_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/feature-flags/{flag_name}/status")
async def check_feature_flag_status(
    flag_name: str,
    service_manager: CorrectedServiceManager = Depends(get_initialized_service_manager)
):
    """Check if a feature flag is enabled"""
    try:
        feature_manager = get_feature_flags_manager()
        enabled = feature_manager.is_enabled(flag_name)
        
        return {
            "flag_name": flag_name,
            "enabled": enabled,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to check feature flag status {flag_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/feature-flags/audit-log")
async def get_audit_log(
    limit: int = Query(100, ge=1, le=1000, description="Number of entries to return"),
    service_manager: CorrectedServiceManager = Depends(get_initialized_service_manager)
):
    """Get feature flag audit log"""
    try:
        feature_manager = get_feature_flags_manager()
        audit_log = feature_manager.get_audit_log(limit)
        
        return {
            "audit_log": audit_log,
            "total_entries": len(audit_log),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get audit log: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/feature-flags/reset")
async def reset_feature_flags(
    modified_by: str = Query("api", description="User who made the change"),
    service_manager: CorrectedServiceManager = Depends(get_initialized_service_manager)
):
    """Reset all feature flags to default values"""
    try:
        feature_manager = get_feature_flags_manager()
        success = feature_manager.reset_to_defaults(modified_by)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to reset feature flags")
        
        return {
            "success": True,
            "message": "All feature flags reset to defaults",
            "modified_by": modified_by,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to reset feature flags: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/feature-flags/signal-config")
async def get_signal_config(
    service_manager: CorrectedServiceManager = Depends(get_initialized_service_manager)
):
    """Get current signal generation configuration"""
    try:
        feature_manager = get_feature_flags_manager()
        config = feature_manager.get_signal_config()
        
        return {
            "base_interval_seconds": config.base_interval_seconds,
            "market_hours_interval_seconds": config.market_hours_interval_seconds,
            "off_hours_interval_seconds": config.off_hours_interval_seconds,
            "min_confidence_threshold": config.min_confidence_threshold,
            "max_confidence_threshold": config.max_confidence_threshold,
            "news_boost_multiplier": config.news_boost_multiplier,
            "regime_boost_multiplier": config.regime_boost_multiplier,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get signal config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/feature-flags/signal-config")
async def update_signal_config(
    config_data: Dict[str, Any],
    modified_by: str = Query("api", description="User who made the change"),
    service_manager: CorrectedServiceManager = Depends(get_initialized_service_manager)
):
    """Update signal generation configuration"""
    try:
        feature_manager = get_feature_flags_manager()
        
        # Validate and create config
        config = SignalGenerationConfig(
            base_interval_seconds=config_data.get("base_interval_seconds", 30),
            market_hours_interval_seconds=config_data.get("market_hours_interval_seconds", 5),
            off_hours_interval_seconds=config_data.get("off_hours_interval_seconds", 300),
            min_confidence_threshold=config_data.get("min_confidence_threshold", 0.7),
            max_confidence_threshold=config_data.get("max_confidence_threshold", 0.95),
            news_boost_multiplier=config_data.get("news_boost_multiplier", 1.2),
            regime_boost_multiplier=config_data.get("regime_boost_multiplier", 1.1)
        )
        
        success = feature_manager.update_signal_config(config, modified_by)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to update signal config")
        
        return {
            "success": True,
            "message": "Signal generation configuration updated",
            "config": {
                "base_interval_seconds": config.base_interval_seconds,
                "market_hours_interval_seconds": config.market_hours_interval_seconds,
                "off_hours_interval_seconds": config.off_hours_interval_seconds,
                "min_confidence_threshold": config.min_confidence_threshold,
                "max_confidence_threshold": config.max_confidence_threshold,
                "news_boost_multiplier": config.news_boost_multiplier,
                "regime_boost_multiplier": config.regime_boost_multiplier
            },
            "modified_by": modified_by,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to update signal config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/feature-flags/news-config")
async def get_news_config(
    service_manager: CorrectedServiceManager = Depends(get_initialized_service_manager)
):
    """Get current news configuration"""
    try:
        feature_manager = get_feature_flags_manager()
        config = feature_manager.get_news_config()
        
        return {
            "enabled_sources": config.enabled_sources,
            "disabled_sources": config.disabled_sources,
            "max_articles_per_source": config.max_articles_per_source,
            "sentiment_weight": config.sentiment_weight,
            "breaking_news_weight": config.breaking_news_weight,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get news config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/feature-flags/news-config")
async def update_news_config(
    config_data: Dict[str, Any],
    modified_by: str = Query("api", description="User who made the change"),
    service_manager: CorrectedServiceManager = Depends(get_initialized_service_manager)
):
    """Update news configuration"""
    try:
        feature_manager = get_feature_flags_manager()
        
        # Validate and create config
        config = NewsSourceConfig(
            enabled_sources=config_data.get("enabled_sources", ["reuters", "bloomberg", "cnbc", "yahoo_finance"]),
            disabled_sources=config_data.get("disabled_sources", []),
            max_articles_per_source=config_data.get("max_articles_per_source", 50),
            sentiment_weight=config_data.get("sentiment_weight", 0.3),
            breaking_news_weight=config_data.get("breaking_news_weight", 0.7)
        )
        
        success = feature_manager.update_news_config(config, modified_by)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to update news config")
        
        return {
            "success": True,
            "message": "News configuration updated",
            "config": {
                "enabled_sources": config.enabled_sources,
                "disabled_sources": config.disabled_sources,
                "max_articles_per_source": config.max_articles_per_source,
                "sentiment_weight": config.sentiment_weight,
                "breaking_news_weight": config.breaking_news_weight
            },
            "modified_by": modified_by,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to update news config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/feature-flags/summary")
async def get_feature_flags_summary(
    service_manager: CorrectedServiceManager = Depends(get_initialized_service_manager)
):
    """Get a summary of all feature flags and configurations"""
    try:
        feature_manager = get_feature_flags_manager()
        
        # Get flags by category
        categories = {}
        for category in FeatureCategory:
            flags = feature_manager.get_flags_by_category(category)
            categories[category.value] = {
                "enabled_count": sum(1 for f in flags.values() if f.enabled),
                "total_count": len(flags),
                "flags": {name: {"enabled": flag.enabled, "description": flag.description} 
                         for name, flag in flags.items()}
            }
        
        # Get configurations
        signal_config = feature_manager.get_signal_config()
        news_config = feature_manager.get_news_config()
        
        return {
            "categories": categories,
            "signal_config": {
                "base_interval_seconds": signal_config.base_interval_seconds,
                "market_hours_interval_seconds": signal_config.market_hours_interval_seconds,
                "off_hours_interval_seconds": signal_config.off_hours_interval_seconds,
                "min_confidence_threshold": signal_config.min_confidence_threshold,
                "max_confidence_threshold": signal_config.max_confidence_threshold
            },
            "news_config": {
                "enabled_sources_count": len(news_config.enabled_sources),
                "disabled_sources_count": len(news_config.disabled_sources),
                "max_articles_per_source": news_config.max_articles_per_source
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get feature flags summary: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 