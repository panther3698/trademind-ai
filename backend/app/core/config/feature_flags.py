"""
Feature Flags Configuration System for TradeMind AI

Provides runtime configuration management for easy system tuning and A/B testing.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

class FeatureCategory(Enum):
    """Feature categories for organization"""
    NEWS_INTELLIGENCE = "news_intelligence"
    REGIME_DETECTION = "regime_detection"
    TRADING = "trading"
    SIGNAL_GENERATION = "signal_generation"
    SYSTEM = "system"

@dataclass
class FeatureFlag:
    """Individual feature flag definition"""
    name: str
    category: FeatureCategory
    description: str
    enabled: bool
    default_value: bool
    configurable: bool = True
    validation_rules: Optional[Dict[str, Any]] = None
    dependencies: Optional[List[str]] = None
    last_modified: Optional[datetime] = None
    modified_by: Optional[str] = None

@dataclass
class SignalGenerationConfig:
    """Signal generation frequency and threshold configuration"""
    base_interval_seconds: int = 30
    market_hours_interval_seconds: int = 5
    off_hours_interval_seconds: int = 300
    min_confidence_threshold: float = 0.7
    max_confidence_threshold: float = 0.95
    news_boost_multiplier: float = 1.2
    regime_boost_multiplier: float = 1.1

@dataclass
class NewsSourceConfig:
    """News source configuration"""
    enabled_sources: List[str]
    disabled_sources: List[str]
    max_articles_per_source: int = 50
    sentiment_weight: float = 0.3
    breaking_news_weight: float = 0.7

class FeatureFlagsManager:
    """Centralized feature flags management system"""
    
    def __init__(self, config_file: str = "feature_flags.json"):
        self.config_file = Path(config_file)
        self.lock = threading.RLock()
        self.flags: Dict[str, FeatureFlag] = {}
        self.signal_config = SignalGenerationConfig()
        self.news_config = NewsSourceConfig(
            enabled_sources=["reuters", "bloomberg", "cnbc", "yahoo_finance"],
            disabled_sources=[]
        )
        self.audit_log: List[Dict[str, Any]] = []
        self._load_default_flags()
        self._load_config()
    
    def _load_default_flags(self):
        """Load default feature flags"""
        default_flags = [
            # News Intelligence Flags
            FeatureFlag(
                name="news_intelligence_enabled",
                category=FeatureCategory.NEWS_INTELLIGENCE,
                description="Enable news intelligence processing",
                enabled=True,
                default_value=True
            ),
            FeatureFlag(
                name="breaking_news_alerts",
                category=FeatureCategory.NEWS_INTELLIGENCE,
                description="Enable breaking news alerts",
                enabled=True,
                default_value=True
            ),
            FeatureFlag(
                name="news_sentiment_analysis",
                category=FeatureCategory.NEWS_INTELLIGENCE,
                description="Enable news sentiment analysis",
                enabled=True,
                default_value=True
            ),
            FeatureFlag(
                name="news_signal_integration",
                category=FeatureCategory.NEWS_INTELLIGENCE,
                description="Enable news-signal integration",
                enabled=True,
                default_value=True,
                dependencies=["news_intelligence_enabled"]
            ),
            
            # Regime Detection Flags
            FeatureFlag(
                name="regime_detection_enabled",
                category=FeatureCategory.REGIME_DETECTION,
                description="Enable market regime detection",
                enabled=True,
                default_value=True
            ),
            FeatureFlag(
                name="regime_adaptive_signals",
                category=FeatureCategory.REGIME_DETECTION,
                description="Enable regime-adaptive signal generation",
                enabled=True,
                default_value=True,
                dependencies=["regime_detection_enabled"]
            ),
            FeatureFlag(
                name="regime_notifications",
                category=FeatureCategory.REGIME_DETECTION,
                description="Enable regime change notifications",
                enabled=True,
                default_value=True,
                dependencies=["regime_detection_enabled"]
            ),
            
            # Trading Flags
            FeatureFlag(
                name="interactive_trading_enabled",
                category=FeatureCategory.TRADING,
                description="Enable interactive trading features",
                enabled=False,
                default_value=False
            ),
            FeatureFlag(
                name="order_execution_enabled",
                category=FeatureCategory.TRADING,
                description="Enable order execution",
                enabled=False,
                default_value=False,
                dependencies=["interactive_trading_enabled"]
            ),
            FeatureFlag(
                name="webhook_handler_enabled",
                category=FeatureCategory.TRADING,
                description="Enable webhook handler",
                enabled=False,
                default_value=False
            ),
            
            # Signal Generation Flags
            FeatureFlag(
                name="signal_generation_enabled",
                category=FeatureCategory.SIGNAL_GENERATION,
                description="Enable signal generation",
                enabled=True,
                default_value=True
            ),
            FeatureFlag(
                name="ml_signal_generation",
                category=FeatureCategory.SIGNAL_GENERATION,
                description="Enable ML-based signal generation",
                enabled=True,
                default_value=True,
                dependencies=["signal_generation_enabled"]
            ),
            FeatureFlag(
                name="technical_signal_generation",
                category=FeatureCategory.SIGNAL_GENERATION,
                description="Enable technical analysis signals",
                enabled=True,
                default_value=True,
                dependencies=["signal_generation_enabled"]
            ),
            
            # System Flags
            FeatureFlag(
                name="demo_signals_enabled",
                category=FeatureCategory.SYSTEM,
                description="Enable demo signal generation",
                enabled=False,
                default_value=False
            ),
            FeatureFlag(
                name="production_mode",
                category=FeatureCategory.SYSTEM,
                description="Enable production mode",
                enabled=True,
                default_value=True
            ),
            FeatureFlag(
                name="debug_logging",
                category=FeatureCategory.SYSTEM,
                description="Enable debug logging",
                enabled=False,
                default_value=False
            )
        ]
        
        for flag in default_flags:
            self.flags[flag.name] = flag
    
    def _load_config(self):
        """Load configuration from file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                
                # Load feature flags
                if 'flags' in config:
                    for flag_name, flag_data in config['flags'].items():
                        if flag_name in self.flags:
                            self.flags[flag_name].enabled = flag_data.get('enabled', self.flags[flag_name].default_value)
                            if 'last_modified' in flag_data and flag_data['last_modified'] is not None:
                                self.flags[flag_name].last_modified = datetime.fromisoformat(flag_data['last_modified'])
                            else:
                                self.flags[flag_name].last_modified = None
                            if 'modified_by' in flag_data:
                                self.flags[flag_name].modified_by = flag_data['modified_by']
                
                # Load signal generation config
                if 'signal_config' in config:
                    signal_data = config['signal_config']
                    self.signal_config = SignalGenerationConfig(
                        base_interval_seconds=signal_data.get('base_interval_seconds', 30),
                        market_hours_interval_seconds=signal_data.get('market_hours_interval_seconds', 5),
                        off_hours_interval_seconds=signal_data.get('off_hours_interval_seconds', 300),
                        min_confidence_threshold=signal_data.get('min_confidence_threshold', 0.7),
                        max_confidence_threshold=signal_data.get('max_confidence_threshold', 0.95),
                        news_boost_multiplier=signal_data.get('news_boost_multiplier', 1.2),
                        regime_boost_multiplier=signal_data.get('regime_boost_multiplier', 1.1)
                    )
                
                # Load news config
                if 'news_config' in config:
                    news_data = config['news_config']
                    self.news_config = NewsSourceConfig(
                        enabled_sources=news_data.get('enabled_sources', ["reuters", "bloomberg", "cnbc", "yahoo_finance"]),
                        disabled_sources=news_data.get('disabled_sources', []),
                        max_articles_per_source=news_data.get('max_articles_per_source', 50),
                        sentiment_weight=news_data.get('sentiment_weight', 0.3),
                        breaking_news_weight=news_data.get('breaking_news_weight', 0.7)
                    )
                
                logger.info(f"✅ Feature flags loaded from {self.config_file}")
            else:
                self._save_config()  # Create default config file
                logger.info(f"✅ Default feature flags config created at {self.config_file}")
                
        except Exception as e:
            logger.error(f"❌ Failed to load feature flags config: {e}")
    
    def _save_config(self):
        """Save configuration to file"""
        try:
            with self.lock:
                config = {
                    'flags': {},
                    'signal_config': asdict(self.signal_config),
                    'news_config': asdict(self.news_config),
                    'last_updated': datetime.now().isoformat()
                }
                
                for flag_name, flag in self.flags.items():
                    config['flags'][flag_name] = {
                        'enabled': flag.enabled,
                        'last_modified': flag.last_modified.isoformat() if flag.last_modified else None,
                        'modified_by': flag.modified_by
                    }
                
                with open(self.config_file, 'w') as f:
                    json.dump(config, f, indent=2)
                    
        except Exception as e:
            logger.error(f"❌ Failed to save feature flags config: {e}")
    
    def is_enabled(self, flag_name: str) -> bool:
        """Check if a feature flag is enabled"""
        with self.lock:
            if flag_name not in self.flags:
                logger.warning(f"⚠️ Unknown feature flag: {flag_name}")
                return False
            
            flag = self.flags[flag_name]
            
            # Check dependencies
            if flag.dependencies:
                for dep in flag.dependencies:
                    if not self.is_enabled(dep):
                        logger.debug(f"🔗 Feature {flag_name} disabled due to dependency {dep}")
                        return False
            
            return flag.enabled
    
    def set_flag(self, flag_name: str, enabled: bool, modified_by: str = "system") -> bool:
        """Set a feature flag value"""
        with self.lock:
            if flag_name not in self.flags:
                logger.error(f"❌ Unknown feature flag: {flag_name}")
                return False
            
            flag = self.flags[flag_name]
            if not flag.configurable:
                logger.error(f"❌ Feature flag {flag_name} is not configurable")
                return False
            
            old_value = flag.enabled
            flag.enabled = enabled
            flag.last_modified = datetime.now()
            flag.modified_by = modified_by
            
            # Validate flag combinations
            if not self._validate_flag_combinations():
                logger.error(f"❌ Invalid flag combination detected, reverting {flag_name}")
                flag.enabled = old_value
                flag.last_modified = None
                flag.modified_by = None
                return False
            
            # Log the change
            self._log_flag_change(flag_name, old_value, enabled, modified_by)
            
            # Save configuration
            self._save_config()
            
            logger.info(f"✅ Feature flag {flag_name} set to {enabled} by {modified_by}")
            return True
    
    def _validate_flag_combinations(self) -> bool:
        """Validate flag combinations for consistency"""
        # Production mode conflicts
        if self.is_enabled("production_mode") and self.is_enabled("demo_signals_enabled"):
            logger.error("❌ Cannot enable demo signals in production mode")
            return False
        
        # Interactive trading dependencies
        if self.is_enabled("order_execution_enabled") and not self.is_enabled("interactive_trading_enabled"):
            logger.error("❌ Order execution requires interactive trading to be enabled")
            return False
        
        # Signal generation dependencies
        if (self.is_enabled("ml_signal_generation") or self.is_enabled("technical_signal_generation")) and not self.is_enabled("signal_generation_enabled"):
            logger.error("❌ Signal generation methods require signal generation to be enabled")
            return False
        
        return True
    
    def _log_flag_change(self, flag_name: str, old_value: bool, new_value: bool, modified_by: str):
        """Log feature flag changes for audit trail"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'flag_name': flag_name,
            'old_value': old_value,
            'new_value': new_value,
            'modified_by': modified_by
        }
        
        self.audit_log.append(log_entry)
        
        # Keep only last 1000 entries
        if len(self.audit_log) > 1000:
            self.audit_log = self.audit_log[-1000:]
        
        logger.info(f"📝 Feature flag change: {flag_name} {old_value} → {new_value} by {modified_by}")
    
    def get_flag(self, flag_name: str) -> Optional[FeatureFlag]:
        """Get feature flag details"""
        with self.lock:
            return self.flags.get(flag_name)
    
    def get_all_flags(self) -> Dict[str, FeatureFlag]:
        """Get all feature flags"""
        with self.lock:
            return self.flags.copy()
    
    def get_flags_by_category(self, category: FeatureCategory) -> Dict[str, FeatureFlag]:
        """Get feature flags by category"""
        with self.lock:
            return {name: flag for name, flag in self.flags.items() if flag.category == category}
    
    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit log entries"""
        with self.lock:
            return self.audit_log[-limit:] if limit else self.audit_log.copy()
    
    def reset_to_defaults(self, modified_by: str = "system") -> bool:
        """Reset all flags to default values"""
        with self.lock:
            for flag_name, flag in self.flags.items():
                if flag.enabled != flag.default_value:
                    self.set_flag(flag_name, flag.default_value, modified_by)
            
            # Reset signal config
            self.signal_config = SignalGenerationConfig()
            
            # Reset news config
            self.news_config = NewsSourceConfig(
                enabled_sources=["reuters", "bloomberg", "cnbc", "yahoo_finance"],
                disabled_sources=[]
            )
            
            self._save_config()
            logger.info(f"✅ All feature flags reset to defaults by {modified_by}")
            return True
    
    def update_signal_config(self, config: SignalGenerationConfig, modified_by: str = "system") -> bool:
        """Update signal generation configuration"""
        with self.lock:
            old_config = self.signal_config
            self.signal_config = config
            self._save_config()
            
            logger.info(f"✅ Signal generation config updated by {modified_by}")
            return True
    
    def update_news_config(self, config: NewsSourceConfig, modified_by: str = "system") -> bool:
        """Update news configuration"""
        with self.lock:
            old_config = self.news_config
            self.news_config = config
            self._save_config()
            
            logger.info(f"✅ News config updated by {modified_by}")
            return True
    
    def get_signal_config(self) -> SignalGenerationConfig:
        """Get current signal generation configuration"""
        with self.lock:
            return self.signal_config
    
    def get_news_config(self) -> NewsSourceConfig:
        """Get current news configuration"""
        with self.lock:
            return self.news_config

# Global feature flags manager instance
_feature_flags_manager: Optional[FeatureFlagsManager] = None

def get_feature_flags_manager() -> FeatureFlagsManager:
    """Get the global feature flags manager instance"""
    global _feature_flags_manager
    if _feature_flags_manager is None:
        _feature_flags_manager = FeatureFlagsManager()
    return _feature_flags_manager

def is_feature_enabled(flag_name: str) -> bool:
    """Convenience function to check if a feature is enabled"""
    return get_feature_flags_manager().is_enabled(flag_name) 