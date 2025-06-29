import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Application settings with comprehensive configuration including ML capabilities + News Intelligence - PRODUCTION MODE"""
    
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
    telegram_webhook_secret: Optional[str] = Field(default=None, env="TELEGRAM_WEBHOOK_SECRET")
    
    # Zerodha
    zerodha_api_key: Optional[str] = Field(default=None, env="ZERODHA_API_KEY")
    zerodha_secret: Optional[str] = Field(default=None, env="ZERODHA_SECRET")
    zerodha_access_token: Optional[str] = Field(default=None, env="ZERODHA_ACCESS_TOKEN")
    
    # News APIs
    news_api_key: Optional[str] = Field(default=None, env="NEWS_API_KEY")
    eodhd_api_key: Optional[str] = Field(default=None, env="EODHD_API_KEY")
    
    # Enhanced News Intelligence APIs
    alpha_vantage_api_key: Optional[str] = Field(default=None, env="ALPHA_VANTAGE_API_KEY")
    finnhub_api_key: Optional[str] = Field(default=None, env="FINNHUB_API_KEY")
    polygon_api_key: Optional[str] = Field(default=None, env="POLYGON_API_KEY")
    
    # News Processing Configuration
    news_sentiment_threshold: float = Field(default=0.3, env="NEWS_SENTIMENT_THRESHOLD")
    news_significance_threshold: float = Field(default=0.7, env="NEWS_SIGNIFICANCE_THRESHOLD")
    news_monitoring_interval: int = Field(default=900, env="NEWS_MONITORING_INTERVAL")  # 15 minutes
    # FIXED: Changed field name to match what's expected in error and environment
    max_news_articles_per_cycle: int = Field(default=500, env="MAX_NEWS_ARTICLES_PER_CYCLE")
    
    # Multi-language News Analysis
    # FIXED: Changed field names to match what's expected in error and environment
    enable_hindi_news_analysis: bool = Field(default=True, env="ENABLE_HINDI_NEWS_ANALYSIS")
    enable_gujarati_news_analysis: bool = Field(default=False, env="ENABLE_GUJARATI_NEWS_ANALYSIS")
    
    # News Sources Configuration
    enable_domestic_news: bool = Field(default=True, env="ENABLE_DOMESTIC_NEWS")
    enable_global_news: bool = Field(default=True, env="ENABLE_GLOBAL_NEWS")
    enable_realtime_alerts: bool = Field(default=True, env="ENABLE_REALTIME_ALERTS")
    enable_breaking_news_alerts: bool = Field(default=True, env="ENABLE_BREAKING_NEWS_ALERTS")
    
    # Payment
    razorpay_key_id: Optional[str] = Field(default=None, env="RAZORPAY_KEY_ID")
    razorpay_secret: Optional[str] = Field(default=None, env="RAZORPAY_SECRET")
    
    # Email
    smtp_server: str = Field(default="smtp.gmail.com", env="SMTP_SERVER")
    smtp_port: int = Field(default=587, env="SMTP_PORT")
    smtp_username: Optional[str] = Field(default=None, env="SMTP_USERNAME")
    smtp_password: Optional[str] = Field(default=None, env="SMTP_PASSWORD")
    
    # System - PRODUCTION DEFAULTS
    environment: str = Field(default="production", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")  # Changed to False for production
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Trading System - PRODUCTION SETTINGS
    max_signals_per_day: int = Field(default=3, env="MAX_SIGNALS_PER_DAY")
    signal_generation_interval: int = Field(default=60, env="SIGNAL_GENERATION_INTERVAL")
    min_confidence_threshold: float = Field(default=0.75, env="MIN_CONFIDENCE_THRESHOLD")  # Raised to 75%
    
    # Analytics
    track_performance: bool = Field(default=True, env="TRACK_PERFORMANCE")
    analytics_retention_days: int = Field(default=90, env="ANALYTICS_RETENTION_DAYS")
    
    # PRODUCTION MODE SETTINGS - Demo signals disabled by default
    development_mode: bool = Field(default=False, env="DEVELOPMENT_MODE")  # Changed to False
    enable_demo_signals: bool = Field(default=False, env="ENABLE_DEMO_SIGNALS")  # NEW - Demo signals disabled
    production_only: bool = Field(default=True, env="PRODUCTION_ONLY")  # NEW - Production only mode
    
    # Dashboard Settings (for personal use)
    public_signals: bool = Field(default=False, env="PUBLIC_SIGNALS")
    require_subscription: bool = Field(default=False, env="REQUIRE_SUBSCRIPTION")
    personal_mode: bool = Field(default=True, env="PERSONAL_MODE")
    enable_testing_controls: bool = Field(default=False, env="ENABLE_TESTING_CONTROLS")  # Changed to False
    
    # ML Configuration - PRODUCTION READY
    ml_confidence_threshold: float = Field(default=0.75, env="ML_CONFIDENCE_THRESHOLD")  # Raised to 75%
    enable_finbert: bool = Field(default=True, env="ENABLE_FINBERT")
    enable_model_retraining: bool = Field(default=True, env="ENABLE_MODEL_RETRAINING")
    model_retrain_frequency_days: int = Field(default=7, env="MODEL_RETRAIN_FREQUENCY")
    
    # Stock Universe Configuration
    use_nifty_100: bool = Field(default=True, env="USE_NIFTY_100")
    max_stocks_per_scan: int = Field(default=100, env="MAX_STOCKS_PER_SCAN")
    
    # ADDITIONAL PRODUCTION SETTINGS
    strict_risk_management: bool = Field(default=True, env="STRICT_RISK_MANAGEMENT")  # NEW
    fallback_analysis_threshold: float = Field(default=0.75, env="FALLBACK_ANALYSIS_THRESHOLD")  # NEW - Higher threshold for fallback
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    # ================================================================
    # EXISTING PROPERTIES (Unchanged)
    # ================================================================
    
    @property
    def is_telegram_configured(self) -> bool:
        """Check if Telegram is properly configured"""
        return bool(self.telegram_bot_token and self.telegram_chat_id)
    
    @property
    def is_zerodha_configured(self) -> bool:
        """Check if Zerodha is properly configured"""
        return bool(self.zerodha_api_key and self.zerodha_access_token)
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment.lower() == "production" or self.production_only
    
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
    
    # NEW PRODUCTION PROPERTIES
    @property
    def is_demo_mode(self) -> bool:
        """Check if demo signals are enabled (should be False in production)"""
        return self.enable_demo_signals and self.development_mode
    
    @property
    def is_strict_production(self) -> bool:
        """Check if running in strict production mode"""
        return self.production_only and not self.development_mode
    
    @property
    def should_use_fallback_analysis(self) -> bool:
        """Check if fallback analysis should be used (with strict thresholds in production)"""
        return self.is_production
    
    @property
    def production_confidence_threshold(self) -> float:
        """Get production-ready confidence threshold"""
        if self.is_strict_production:
            return max(0.75, self.effective_confidence_threshold)  # Minimum 75% in strict production
        return self.effective_confidence_threshold
    
    # ================================================================
    # NEWS INTELLIGENCE PROPERTIES (Updated to use correct field names)
    # ================================================================
    
    @property
    def is_news_intelligence_configured(self) -> bool:
        """Check if news intelligence is properly configured"""
        return bool(self.news_api_key or self.eodhd_api_key or self.alpha_vantage_api_key)
    
    @property
    def has_realtime_news_sources(self) -> bool:
        """Check if real-time news sources are available"""
        return bool(self.alpha_vantage_api_key or self.finnhub_api_key)
    
    @property
    def has_premium_news_sources(self) -> bool:
        """Check if premium news sources are available"""
        return bool(self.finnhub_api_key or self.polygon_api_key)
    
    @property
    def news_monitoring_enabled(self) -> bool:
        """Check if news monitoring should be enabled"""
        return (self.is_news_intelligence_configured and 
                self.enable_realtime_alerts and 
                self.is_production)
    
    @property
    def breaking_news_enabled(self) -> bool:
        """Check if breaking news alerts should be enabled"""
        return (self.enable_breaking_news_alerts and 
                self.has_realtime_news_sources and 
                self.is_telegram_configured)
    
    @property
    def multi_language_news_enabled(self) -> bool:
        """Check if multi-language news analysis is enabled"""
        return self.enable_hindi_news_analysis or self.enable_gujarati_news_analysis
    
    @property
    def news_sources_summary(self) -> dict:
        """Get a summary of available news sources"""
        return {
            "domestic_news": self.enable_domestic_news,
            "global_news": self.enable_global_news,
            "realtime_alerts": self.enable_realtime_alerts,
            "breaking_news": self.enable_breaking_news_alerts,
            "has_news_api": bool(self.news_api_key),
            "has_eodhd": bool(self.eodhd_api_key),
            "has_alpha_vantage": bool(self.alpha_vantage_api_key),
            "has_finnhub": bool(self.finnhub_api_key),
            "has_polygon": bool(self.polygon_api_key),
            "multi_language": self.multi_language_news_enabled
        }
    
    @property
    def effective_news_monitoring_interval(self) -> int:
        """Get effective news monitoring interval based on available sources"""
        if self.has_premium_news_sources:
            return min(self.news_monitoring_interval, 300)  # Max 5 minutes with premium sources
        elif self.has_realtime_news_sources:
            return min(self.news_monitoring_interval, 600)  # Max 10 minutes with real-time sources
        else:
            return max(self.news_monitoring_interval, 1800)  # Min 30 minutes with basic sources

    # COMPATIBILITY PROPERTIES (for backward compatibility with existing code)
    @property 
    def max_news_articles(self) -> int:
        """Backward compatibility property"""
        return self.max_news_articles_per_cycle
    
    @property
    def enable_hindi_analysis(self) -> bool:
        """Backward compatibility property"""
        return self.enable_hindi_news_analysis
        
    @property
    def enable_gujarati_analysis(self) -> bool:
        """Backward compatibility property"""
        return self.enable_gujarati_news_analysis

# Global settings instance
settings = Settings() 