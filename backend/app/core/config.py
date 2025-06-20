import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Application settings with comprehensive configuration including ML capabilities"""
    
    # Database
    database_url: str = Field(default="sqlite:///./trademind.db", env="DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # Security
    secret_key: str = Field(..., env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Telegram
    telegram_bot_token: Optional[str] = Field(default=None, env="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = Field(default=None, env="TELEGRAM_CHAT_ID")
    
    # Zerodha
    zerodha_api_key: Optional[str] = Field(default=None, env="ZERODHA_API_KEY")
    zerodha_secret: Optional[str] = Field(default=None, env="ZERODHA_SECRET")
    zerodha_access_token: Optional[str] = Field(default=None, env="ZERODHA_ACCESS_TOKEN")
    
    # News APIs
    news_api_key: Optional[str] = Field(default=None, env="NEWS_API_KEY")
    eodhd_api_key: Optional[str] = Field(default=None, env="EODHD_API_KEY")
    
    # Payment
    razorpay_key_id: Optional[str] = Field(default=None, env="RAZORPAY_KEY_ID")
    razorpay_secret: Optional[str] = Field(default=None, env="RAZORPAY_SECRET")
    
    # Email
    smtp_server: str = Field(default="smtp.gmail.com", env="SMTP_SERVER")
    smtp_port: int = Field(default=587, env="SMTP_PORT")
    smtp_username: Optional[str] = Field(default=None, env="SMTP_USERNAME")
    smtp_password: Optional[str] = Field(default=None, env="SMTP_PASSWORD")
    
    # System
    environment: str = Field(default="production", env="ENVIRONMENT")  # Updated to production default
    debug: bool = Field(default=True, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Trading System
    max_signals_per_day: int = Field(default=3, env="MAX_SIGNALS_PER_DAY")
    signal_generation_interval: int = Field(default=60, env="SIGNAL_GENERATION_INTERVAL")  # Updated to 60 seconds
    min_confidence_threshold: float = Field(default=0.65, env="MIN_CONFIDENCE_THRESHOLD")
    
    # Analytics
    track_performance: bool = Field(default=True, env="TRACK_PERFORMANCE")
    analytics_retention_days: int = Field(default=90, env="ANALYTICS_RETENTION_DAYS")
    
    # Development Mode Settings (for personal dashboard)
    development_mode: bool = Field(default=True, env="DEVELOPMENT_MODE")
    public_signals: bool = Field(default=False, env="PUBLIC_SIGNALS")
    require_subscription: bool = Field(default=False, env="REQUIRE_SUBSCRIPTION")
    personal_mode: bool = Field(default=True, env="PERSONAL_MODE")
    enable_testing_controls: bool = Field(default=True, env="ENABLE_TESTING_CONTROLS")
    
    # ML Configuration (NEW - from merged script)
    ml_confidence_threshold: float = Field(default=0.70, env="ML_CONFIDENCE_THRESHOLD")
    enable_finbert: bool = Field(default=True, env="ENABLE_FINBERT")
    enable_model_retraining: bool = Field(default=True, env="ENABLE_MODEL_RETRAINING")
    model_retrain_frequency_days: int = Field(default=7, env="MODEL_RETRAIN_FREQUENCY")
    
    # Stock Universe Configuration (NEW - from merged script)
    use_nifty_100: bool = Field(default=True, env="USE_NIFTY_100")
    max_stocks_per_scan: int = Field(default=100, env="MAX_STOCKS_PER_SCAN")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @property
    def is_telegram_configured(self) -> bool:
        """Check if Telegram is properly configured"""
        return bool(self.telegram_bot_token and self.telegram_chat_id)
    
    @property
    def is_zerodha_configured(self) -> bool:
        """Check if Zerodha is properly configured"""
        return bool(self.zerodha_api_key and self.zerodha_access_token)  # Updated to use access_token
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment.lower() == "production"
    
    @property
    def is_personal_mode(self) -> bool:
        """Check if running in personal development mode"""
        return self.development_mode and self.personal_mode
    
    @property
    def should_show_disclaimers(self) -> bool:
        """Check if disclaimers should be shown"""
        return not self.public_signals
    
    @property
    def is_ml_enabled(self) -> bool:
        """Check if ML features are enabled"""
        return bool(self.ml_confidence_threshold > 0)
    
    @property
    def effective_confidence_threshold(self) -> float:
        """Get the effective confidence threshold (use ML threshold if available, fallback to min_confidence)"""
        return max(self.ml_confidence_threshold, self.min_confidence_threshold)
    
    @property
    def is_finbert_enabled(self) -> bool:
        """Check if FinBERT sentiment analysis is enabled"""
        return self.enable_finbert and self.is_ml_enabled
    
    @property
    def should_retrain_models(self) -> bool:
        """Check if model retraining is enabled"""
        return self.enable_model_retraining and self.is_ml_enabled

# Global settings instance
settings = Settings()