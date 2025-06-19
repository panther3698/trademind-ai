import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
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
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Trading System
    max_signals_per_day: int = Field(default=3, env="MAX_SIGNALS_PER_DAY")
    signal_generation_interval: int = Field(default=45, env="SIGNAL_GENERATION_INTERVAL")
    min_confidence_threshold: float = Field(default=0.65, env="MIN_CONFIDENCE_THRESHOLD")
    
    # Analytics
    track_performance: bool = Field(default=True, env="TRACK_PERFORMANCE")
    analytics_retention_days: int = Field(default=90, env="ANALYTICS_RETENTION_DAYS")
    
    # Development Mode Settings (NEW - for personal dashboard)
    development_mode: bool = Field(default=True, env="DEVELOPMENT_MODE")
    public_signals: bool = Field(default=False, env="PUBLIC_SIGNALS")
    require_subscription: bool = Field(default=False, env="REQUIRE_SUBSCRIPTION")
    personal_mode: bool = Field(default=True, env="PERSONAL_MODE")
    enable_testing_controls: bool = Field(default=True, env="ENABLE_TESTING_CONTROLS")
    
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
        return bool(self.zerodha_api_key and self.zerodha_secret)
    
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

# Global settings instance
settings = Settings()