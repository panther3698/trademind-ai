from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, validator
from typing import Dict, Any, Optional, List
import json
import os
from datetime import datetime
from app.api.dependencies import get_service_manager
from app.core.availability import settings

router = APIRouter(prefix="/api/config", tags=["configuration"])

# Configuration Models
class TradingConfig(BaseModel):
    max_capital_per_trade: float = 100000
    default_stop_loss_percentage: float = 3.0
    default_target_percentage: float = 5.0
    max_signals_per_day: int = 10
    min_confidence_threshold: float = 0.75
    max_daily_loss_limit: float = 500000
    max_position_size_percentage: float = 20.0
    risk_reward_ratio_min: float = 1.5
    enable_auto_trading: bool = True
    trading_mode: str = "CONSERVATIVE"  # CONSERVATIVE, MODERATE, AGGRESSIVE
    
    @validator('max_capital_per_trade')
    def validate_capital(cls, v):
        if v < 1000 or v > 10000000:
            raise ValueError('Capital must be between ₹1,000 and ₹1,00,00,000')
        return v
    
    @validator('default_stop_loss_percentage')
    def validate_stop_loss(cls, v):
        if v < 0.5 or v > 15:
            raise ValueError('Stop loss must be between 0.5% and 15%')
        return v

class APIConfig(BaseModel):
    # News APIs
    news_api_key: Optional[str] = None
    alpha_vantage_api_key: Optional[str] = None
    eodhd_api_key: Optional[str] = None
    finnhub_api_key: Optional[str] = None
    polygon_api_key: Optional[str] = None
    
    # Trading APIs
    zerodha_api_key: Optional[str] = None
    zerodha_access_token: Optional[str] = None
    zerodha_api_secret: Optional[str] = None
    
    # Telegram
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    
    # API Settings
    news_api_rate_limit: int = 100
    api_timeout_seconds: int = 30
    enable_api_retries: bool = True
    max_retry_attempts: int = 3

class SystemConfig(BaseModel):
    signal_generation_interval: int = 300
    news_monitoring_interval: int = 60
    market_open_time: str = "09:15"
    market_close_time: str = "15:30"
    timezone: str = "Asia/Kolkata"
    
    # Feature Toggles
    enable_news_intelligence: bool = True
    enable_regime_detection: bool = True
    enable_interactive_trading: bool = True
    enable_order_execution: bool = True
    enable_demo_mode: bool = False
    
    # Cache Settings
    cache_ttl_market_data: int = 5
    cache_ttl_news_data: int = 60
    cache_ttl_regime_data: int = 300
    
    # Logging
    log_level: str = "INFO"
    enable_detailed_logging: bool = True

class MLConfig(BaseModel):
    # Confidence Thresholds
    ml_confidence_threshold: float = 0.70
    news_confidence_threshold: float = 0.60
    regime_confidence_threshold: float = 0.55
    
    # Feature Weights
    sentiment_weight: float = 0.3
    technical_weight: float = 0.4
    news_weight: float = 0.3
    volume_weight: float = 0.2
    
    # Model Settings
    regime_adjustment_factor: float = 0.1
    model_retrain_frequency_days: int = 7
    
    # Ensemble Weights
    ensemble_model_weights: Dict[str, float] = {
        "xgboost": 0.4,
        "lightgbm": 0.3,
        "catboost": 0.3
    }
    
    # Risk Scoring
    enable_dynamic_risk_scoring: bool = True
    risk_score_decay_factor: float = 0.1

class BackupConfig(BaseModel):
    auto_backup_enabled: bool = True
    backup_frequency_hours: int = 24
    max_backup_files: int = 30
    backup_location: str = "backups/"

# API Endpoints
@router.get("/trading")
async def get_trading_config(service_manager = Depends(get_service_manager)):
    """Get current trading configuration"""
    try:
        return {
            "max_capital_per_trade": getattr(service_manager, 'max_capital_per_trade', 100000),
            "default_stop_loss_percentage": getattr(service_manager, 'default_stop_loss_percentage', 3.0),
            "default_target_percentage": getattr(service_manager, 'default_target_percentage', 5.0),
            "max_signals_per_day": getattr(service_manager, 'max_signals_per_day', 10),
            "min_confidence_threshold": getattr(service_manager, 'min_confidence_threshold', 0.75),
            "max_daily_loss_limit": getattr(service_manager, 'max_daily_loss_limit', 500000),
            "max_position_size_percentage": getattr(service_manager, 'max_position_size_percentage', 20.0),
            "risk_reward_ratio_min": getattr(service_manager, 'risk_reward_ratio_min', 1.5),
            "enable_auto_trading": getattr(service_manager, 'enable_auto_trading', True),
            "trading_mode": getattr(service_manager, 'trading_mode', "CONSERVATIVE")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/trading")
async def update_trading_config(config: TradingConfig, service_manager = Depends(get_service_manager)):
    """Update trading configuration with real-time application"""
    try:
        # Apply to service manager
        service_manager.max_capital_per_trade = config.max_capital_per_trade
        service_manager.max_signals_per_day = config.max_signals_per_day
        service_manager.enable_auto_trading = config.enable_auto_trading
        
        # Update signal generator
        if service_manager.signal_generator:
            service_manager.signal_generator.max_capital_per_trade = config.max_capital_per_trade
            service_manager.signal_generator.base_min_ml_confidence = config.min_confidence_threshold
            service_manager.signal_generator.default_stop_loss_pct = config.default_stop_loss_percentage
            service_manager.signal_generator.default_target_pct = config.default_target_percentage
        
        # Save to persistent storage
        await save_config_to_storage("trading", config.dict())
        
        return {
            "status": "success", 
            "message": "Trading configuration updated and applied immediately",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update trading config: {str(e)}")

@router.get("/apis")
async def get_api_config():
    """Get current API configuration (with masked sensitive data)"""
    try:
        return {
            "news_api_key": mask_api_key(getattr(settings, 'news_api_key', '')),
            "alpha_vantage_api_key": mask_api_key(getattr(settings, 'alpha_vantage_api_key', '')),
            "polygon_api_key": mask_api_key(getattr(settings, 'polygon_api_key', '')),
            "finnhub_api_key": mask_api_key(getattr(settings, 'finnhub_api_key', '')),
            "eodhd_api_key": mask_api_key(getattr(settings, 'eodhd_api_key', '')),
            "zerodha_api_key": mask_api_key(getattr(settings, 'zerodha_api_key', '')),
            "zerodha_access_token": mask_api_key(getattr(settings, 'zerodha_access_token', '')),
            "telegram_bot_token": mask_api_key(getattr(settings, 'telegram_bot_token', '')),
            "telegram_chat_id": getattr(settings, 'telegram_chat_id', ''),
            "news_api_rate_limit": 100,
            "api_timeout_seconds": 30,
            "enable_api_retries": True,
            "max_retry_attempts": 3
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/apis")
async def update_api_config(config: APIConfig, background_tasks: BackgroundTasks, service_manager = Depends(get_service_manager)):
    """Update API configuration and restart affected services"""
    try:
        updated_services = []
        
        # Update settings
        if config.news_api_key:
            settings.news_api_key = config.news_api_key
            updated_services.append("news_intelligence")
        
        if config.polygon_api_key:
            settings.polygon_api_key = config.polygon_api_key
            updated_services.append("news_intelligence")
        
        if config.zerodha_api_key:
            settings.zerodha_api_key = config.zerodha_api_key
            updated_services.append("order_engine")
        
        if config.zerodha_access_token:
            settings.zerodha_access_token = config.zerodha_access_token
            updated_services.append("order_engine")
        
        if config.telegram_bot_token:
            settings.telegram_bot_token = config.telegram_bot_token
            updated_services.append("telegram_service")
        
        if config.telegram_chat_id:
            settings.telegram_chat_id = config.telegram_chat_id
            updated_services.append("telegram_service")
        
        # Restart affected services in background
        background_tasks.add_task(restart_services, service_manager, updated_services)
        
        # Save configuration
        await save_config_to_storage("apis", config.dict(exclude_none=True))
        
        return {
            "status": "success",
            "message": f"API configuration updated. Restarting services: {', '.join(updated_services)}",
            "updated_services": updated_services,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update API config: {str(e)}")

@router.get("/system")
async def get_system_config(service_manager = Depends(get_service_manager)):
    """Get current system configuration"""
    try:
        return {
            "signal_generation_interval": getattr(service_manager, 'signal_generation_interval', 300),
            "news_monitoring_interval": 60,
            "market_open_time": "09:15",
            "market_close_time": "15:30",
            "timezone": "Asia/Kolkata",
            "enable_news_intelligence": service_manager.system_health.get("news_intelligence", False),
            "enable_regime_detection": service_manager.system_health.get("regime_detection", False),
            "enable_interactive_trading": getattr(service_manager, 'interactive_trading_active', False),
            "enable_order_execution": service_manager.system_health.get("order_execution", False),
            "enable_demo_mode": getattr(service_manager, 'demo_mode', False),
            "cache_ttl_market_data": 5,
            "cache_ttl_news_data": 60,
            "cache_ttl_regime_data": 300,
            "log_level": "INFO",
            "enable_detailed_logging": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/system")
async def update_system_config(config: SystemConfig, service_manager = Depends(get_service_manager)):
    """Update system configuration with immediate effect"""
    try:
        # Apply interval changes
        service_manager.signal_generation_interval = config.signal_generation_interval
        
        # Update task manager intervals
        if hasattr(service_manager, 'task_manager'):
            service_manager.task_manager.update_intervals({
                "signal_generation": config.signal_generation_interval,
                "news_monitoring": config.news_monitoring_interval
            })
        
        # Update cache settings
        if hasattr(service_manager, 'cache_manager'):
            service_manager.cache_manager.update_ttl_settings({
                "market_data": config.cache_ttl_market_data,
                "news_data": config.cache_ttl_news_data,
                "regime_data": config.cache_ttl_regime_data
            })
        
        # Apply feature toggles
        if config.enable_news_intelligence != service_manager.system_health.get("news_intelligence", False):
            await toggle_news_intelligence(service_manager, config.enable_news_intelligence)
        
        if config.enable_interactive_trading != service_manager.interactive_trading_active:
            service_manager.interactive_trading_active = config.enable_interactive_trading
        
        await save_config_to_storage("system", config.dict())
        
        return {
            "status": "success",
            "message": "System configuration updated and applied",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update system config: {str(e)}")

@router.get("/ml")
async def get_ml_config(service_manager = Depends(get_service_manager)):
    """Get current ML configuration"""
    try:
        return {
            "ml_confidence_threshold": 0.70,
            "news_confidence_threshold": 0.60,
            "regime_confidence_threshold": 0.55,
            "sentiment_weight": 0.3,
            "technical_weight": 0.4,
            "news_weight": 0.3,
            "volume_weight": 0.2,
            "regime_adjustment_factor": 0.1,
            "model_retrain_frequency_days": 7,
            "ensemble_model_weights": {
                "xgboost": 0.4,
                "lightgbm": 0.3,
                "catboost": 0.3
            },
            "enable_dynamic_risk_scoring": True,
            "risk_score_decay_factor": 0.1
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ml")
async def update_ml_config(config: MLConfig, service_manager = Depends(get_service_manager)):
    """Update ML configuration with immediate model updates"""
    try:
        # Update signal generator ML parameters
        if service_manager.signal_generator:
            service_manager.signal_generator.base_min_ml_confidence = config.ml_confidence_threshold
            
            # Update feature weights
            if hasattr(service_manager.signal_generator, 'update_feature_weights'):
                service_manager.signal_generator.update_feature_weights({
                    "sentiment": config.sentiment_weight,
                    "technical": config.technical_weight,
                    "news": config.news_weight,
                    "volume": config.volume_weight
                })
            
            # Update ensemble weights
            if hasattr(service_manager.signal_generator, 'update_ensemble_weights'):
                service_manager.signal_generator.update_ensemble_weights(config.ensemble_model_weights)
        
        await save_config_to_storage("ml", config.dict())
        
        return {
            "status": "success",
            "message": "ML configuration updated and applied to models",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update ML config: {str(e)}")

@router.post("/test-apis")
async def test_all_apis(service_manager = Depends(get_service_manager)):
    """Test all configured APIs and return status"""
    try:
        test_results = {}
        # Use notification_service for all service checks
        notification_service = service_manager.notification_service
        order_engine = notification_service.order_engine
        telegram_service = notification_service.telegram_service
        news_intelligence = service_manager.news_intelligence

        # Zerodha Broker
        if not order_engine:
            reason = "Order engine not initialized. Check API key and access token."
            test_results["zerodha"] = {"status": "not_initialized", "error": reason}
        else:
            try:
                status = await order_engine.get_connection_status()
                test_results["zerodha"] = {
                    "status": "connected" if status.get("connected") else "failed",
                    "user": status.get("user_name", "Unknown"),
                    "funds": status.get("available_cash", 0)
                }
            except Exception as e:
                test_results["zerodha"] = {"status": "error", "error": str(e)}

        # Telegram
        if not telegram_service:
            reason = "Telegram service not initialized. Check bot token and chat ID."
            test_results["telegram"] = {"status": "not_initialized", "error": reason}
        else:
            try:
                test_msg = await telegram_service.send_message("🧪 API Test - TradeMind AI")
                test_results["telegram"] = {"status": "success" if test_msg else "failed"}
            except Exception as e:
                test_results["telegram"] = {"status": "error", "error": str(e)}

        # News APIs
        if not news_intelligence:
            reason = "News intelligence not initialized. Check news API keys."
            test_results["news_apis"] = {"status": "not_initialized", "error": reason}
        else:
            try:
                health = await news_intelligence.test_api_connections()
                test_results["news_apis"] = health
            except Exception as e:
                test_results["news_apis"] = {"status": "error", "error": str(e)}

        return {
            "status": "completed",
            "test_results": test_results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "test_results": {},
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.post("/backup")
async def create_configuration_backup():
    """Create backup of all configurations"""
    try:
        # Get all current configurations
        backup_data = {
            "trading": await get_trading_config(),
            "apis": await get_api_config(),
            "system": await get_system_config(),
            "ml": await get_ml_config(),
            "metadata": {
                "backup_date": datetime.now().isoformat(),
                "version": "1.0",
                "system": "TradeMind AI"
            }
        }
        
        # Create backup file
        os.makedirs("backups", exist_ok=True)
        backup_filename = f"trademind_config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        backup_path = f"backups/{backup_filename}"
        
        with open(backup_path, "w") as f:
            json.dump(backup_data, f, indent=2)
        
        return {
            "status": "success",
            "backup_file": backup_filename,
            "backup_path": backup_path,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/restore")
async def restore_configuration(backup_file: str, service_manager = Depends(get_service_manager)):
    """Restore configuration from backup file"""
    try:
        backup_path = f"backups/{backup_file}"
        
        if not os.path.exists(backup_path):
            raise HTTPException(status_code=404, detail="Backup file not found")
        
        with open(backup_path, "r") as f:
            backup_data = json.load(f)
        
        # Restore each configuration section
        restored_sections = []
        
        if "trading" in backup_data:
            await update_trading_config(TradingConfig(**backup_data["trading"]), service_manager)
            restored_sections.append("trading")
        
        if "system" in backup_data:
            await update_system_config(SystemConfig(**backup_data["system"]), service_manager)
            restored_sections.append("system")
        
        if "ml" in backup_data:
            await update_ml_config(MLConfig(**backup_data["ml"]), service_manager)
            restored_sections.append("ml")
        
        return {
            "status": "success",
            "message": f"Configuration restored from {backup_file}",
            "restored_sections": restored_sections,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/backup/list")
async def list_backup_files():
    """List available backup files"""
    try:
        backup_dir = "backups"
        if not os.path.exists(backup_dir):
            return {"backups": []}
        
        backup_files = []
        for file in os.listdir(backup_dir):
            if file.endswith('.json') and 'trademind_config_backup' in file:
                file_path = os.path.join(backup_dir, file)
                stat = os.stat(file_path)
                backup_files.append({
                    "filename": file,
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        # Sort by creation date (newest first)
        backup_files.sort(key=lambda x: x["created"], reverse=True)
        
        return {"backups": backup_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/debug-env")
async def debug_environment():
    """Debug endpoint to check environment variable loading"""
    import os
    from app.core.availability import settings
    
    # Get environment variables directly
    env_vars = {
        'SECRET_KEY': os.getenv('SECRET_KEY', 'NOT_FOUND')[:10] + '...' if os.getenv('SECRET_KEY') else 'NOT_FOUND',
        'ZERODHA_API_KEY': os.getenv('ZERODHA_API_KEY', 'NOT_FOUND')[:10] + '...' if os.getenv('ZERODHA_API_KEY') else 'NOT_FOUND',
        'TELEGRAM_BOT_TOKEN': os.getenv('TELEGRAM_BOT_TOKEN', 'NOT_FOUND')[:10] + '...' if os.getenv('TELEGRAM_BOT_TOKEN') else 'NOT_FOUND',
        'NEWS_API_KEY': os.getenv('NEWS_API_KEY', 'NOT_FOUND')[:10] + '...' if os.getenv('NEWS_API_KEY') else 'NOT_FOUND',
        'ALPHA_VANTAGE_API_KEY': os.getenv('ALPHA_VANTAGE_API_KEY', 'NOT_FOUND')[:10] + '...' if os.getenv('ALPHA_VANTAGE_API_KEY') else 'NOT_FOUND',
    }
    
    # Get settings values
    settings_vars = {
        'SECRET_KEY': settings.secret_key[:10] + '...' if settings.secret_key else 'NOT_LOADED',
        'ZERODHA_API_KEY': settings.zerodha_api_key[:10] + '...' if settings.zerodha_api_key else 'NOT_LOADED',
        'TELEGRAM_BOT_TOKEN': settings.telegram_bot_token[:10] + '...' if settings.telegram_bot_token else 'NOT_LOADED',
        'NEWS_API_KEY': settings.news_api_key[:10] + '...' if settings.news_api_key else 'NOT_LOADED',
        'ALPHA_VANTAGE_API_KEY': settings.alpha_vantage_api_key[:10] + '...' if settings.alpha_vantage_api_key else 'NOT_LOADED',
    }
    
    # Check if .env file exists
    import pathlib
    env_file_path = pathlib.Path('.env')
    env_file_exists = env_file_path.exists()
    env_file_absolute = env_file_path.absolute()
    
    return {
        "message": "Environment Debug Information",
        "env_file_exists": env_file_exists,
        "env_file_path": str(env_file_absolute),
        "current_working_directory": os.getcwd(),
        "environment_variables": env_vars,
        "settings_values": settings_vars,
        "config_available": bool(settings.secret_key),
        "timestamp": datetime.now().isoformat()
    }

# Helper Functions
def mask_api_key(api_key: str) -> str:
    """Mask API key for security (show only last 4 characters)"""
    if not api_key or len(api_key) < 4:
        return ""
    return "***" + api_key[-4:]

async def save_config_to_storage(config_type: str, config_data: Dict[str, Any]):
    """Save configuration to persistent storage"""
    try:
        os.makedirs("config", exist_ok=True)
        config_file = f"config/{config_type}_config.json"
        
        with open(config_file, "w") as f:
            json.dump({
                "config": config_data,
                "updated_at": datetime.now().isoformat()
            }, f, indent=2)
    except Exception as e:
        print(f"Failed to save config {config_type}: {e}")

async def restart_services(service_manager, service_names: List[str]):
    """Restart specified services with new configuration"""
    try:
        for service_name in service_names:
            if service_name == "news_intelligence" and service_manager.news_intelligence:
                await service_manager._initialize_news_intelligence()
            elif service_name == "order_engine" and service_manager.order_engine:
                await service_manager._initialize_order_engine()
            elif service_name == "telegram_service" and service_manager.telegram_service:
                await service_manager._initialize_enhanced_telegram()
    except Exception as e:
        print(f"Failed to restart services: {e}")

async def toggle_news_intelligence(service_manager, enable: bool):
    """Toggle news intelligence on/off"""
    try:
        if enable and not service_manager.system_health.get("news_intelligence"):
            await service_manager._initialize_news_intelligence()
        elif not enable and service_manager.system_health.get("news_intelligence"):
            if service_manager.news_intelligence:
                await service_manager.news_intelligence.close()
                service_manager.news_intelligence = None
                service_manager.system_health["news_intelligence"] = False
    except Exception as e:
        print(f"Failed to toggle news intelligence: {e}") 