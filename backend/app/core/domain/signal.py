# ================================================================
# Signal Domain Models
# ================================================================

"""
Signal domain models for TradeMind AI

This module defines the core signal-related data structures
used throughout the application.
"""

from enum import Enum
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

# ================================================================
# ENUMS
# ================================================================

class SignalType(str, Enum):
    """Signal types for trading decisions"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class SignalSource(str, Enum):
    """Sources of trading signals"""
    MANUAL = "MANUAL"
    ML_MODEL = "ML_MODEL"
    NEWS = "NEWS"
    TECHNICAL = "TECHNICAL"
    HYBRID = "HYBRID"

# ================================================================
# MODELS
# ================================================================

class Signal(BaseModel):
    """Trading signal model"""
    
    symbol: str = Field(..., description="Stock symbol")
    signal_type: SignalType = Field(..., description="Signal type (BUY/SELL/HOLD)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Signal confidence (0-1)")
    price: float = Field(..., gt=0, description="Current price")
    reasoning: str = Field(..., description="Signal reasoning")
    timestamp: datetime = Field(default_factory=datetime.now, description="Signal timestamp")
    source: SignalSource = Field(default=SignalSource.MANUAL, description="Signal source")
    
    # Optional fields
    target_price: Optional[float] = Field(None, gt=0, description="Target price")
    stop_loss: Optional[float] = Field(None, gt=0, description="Stop loss price")
    quantity: Optional[int] = Field(None, gt=0, description="Suggested quantity")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class SignalRequest(BaseModel):
    """Signal generation request model"""
    
    symbol: str = Field(..., description="Stock symbol")
    market_data: Dict[str, Any] = Field(..., description="Market data")
    news_sentiment: Optional[float] = Field(None, ge=-1.0, le=1.0, description="News sentiment")
    user_preferences: Optional[Dict[str, Any]] = Field(default_factory=dict, description="User preferences")

class SignalResponse(BaseModel):
    """Signal generation response model"""
    
    signal: Signal = Field(..., description="Generated signal")
    success: bool = Field(..., description="Generation success status")
    message: str = Field(..., description="Response message")
    processing_time: float = Field(..., description="Processing time in seconds")

# ================================================================
# VALIDATION
# ================================================================

def validate_signal_data(data: Dict[str, Any]) -> bool:
    """Validate signal data"""
    required_fields = ["symbol", "signal_type", "confidence", "price", "reasoning"]
    
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
    
    if not data["symbol"]:
        raise ValueError("Symbol cannot be empty")
    
    if not (0.0 <= data["confidence"] <= 1.0):
        raise ValueError("Confidence must be between 0 and 1")
    
    if data["price"] <= 0:
        raise ValueError("Price must be positive")
    
    return True 