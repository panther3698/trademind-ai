# backend/app/services/zerodha_order_engine.py
"""
TradeMind AI - Zerodha Kite Connect Order Execution Engine
Production-ready order management with proper error handling and logging
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import json

# Import Kite Connect
try:
    from kiteconnect import KiteConnect
    KITE_AVAILABLE = True
except ImportError:
    KITE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ Kite Connect not installed. Install with: pip install kiteconnect")

# Import performance monitoring
from app.core.performance_monitor import performance_monitor

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL_MARKET = "SL-M"
    SL_LIMIT = "SL"

class ExchangeType(Enum):
    NSE = "NSE"
    BSE = "BSE"
    NFO = "NFO"
    CDS = "CDS"
    MCX = "MCX"

@dataclass
class OrderResult:
    success: bool
    order_id: Optional[str] = None
    status: Optional[str] = None
    message: str = ""
    price: Optional[float] = None
    quantity: Optional[int] = None
    timestamp: Optional[datetime] = None
    error: Optional[str] = None

@dataclass
class TradeOrder:
    symbol: str
    action: str  # BUY/SELL
    quantity: int
    price: float
    order_type: OrderType
    exchange: ExchangeType = ExchangeType.NSE
    product: str = "CNC"  # CNC, MIS, NRML
    validity: str = "DAY"  # DAY, IOC
    stop_loss: Optional[float] = None
    target: Optional[float] = None

class ZerodhaOrderEngine:
    """
    Production-ready Zerodha order execution engine
    Handles order placement, modification, cancellation, and tracking
    """
    
    def __init__(self, api_key: str, access_token: str, enable_sandbox: bool = False):
        self.api_key = api_key
        self.access_token = access_token
        self.enable_sandbox = enable_sandbox
        self.kite = None
        self.is_connected = False
        
        # Order tracking
        self.pending_orders: Dict[str, TradeOrder] = {}
        self.completed_orders: Dict[str, OrderResult] = {}
        
        # Rate limiting
        self.order_count = 0
        self.last_order_time = datetime.now()
        self.max_orders_per_minute = 100  # Conservative limit
        
        # Cache for fund information
        self._cached_margins = None
        self._margins_cache_time = None
        self._cache_duration = 30  # Cache margins for 30 seconds
        
        # Initialize connection
        asyncio.create_task(self._initialize_connection())
    
    def _get_available_funds(self, profile: Dict[str, Any] = None, margins: Dict[str, Any] = None) -> float:
        """
        Enhanced fund detection with multiple fallback methods
        """
        if not margins:
            margins = self._get_margins_sync()
        
        available_cash = 0
        
        try:
            # Method 1: Equity segment available cash (most common)
            if margins.get("equity", {}).get("available", {}).get("cash"):
                available_cash = float(margins["equity"]["available"]["cash"])
                logger.debug(f"🔍 Found funds via equity.available.cash: ₹{available_cash:,.2f}")
                return available_cash
            
            # Method 2: Equity segment net cash
            if margins.get("equity", {}).get("net"):
                available_cash = float(margins["equity"]["net"])
                logger.debug(f"🔍 Found funds via equity.net: ₹{available_cash:,.2f}")
                return available_cash
            
            # Method 3: Check enabled cash (for accounts with different settings)
            if margins.get("equity", {}).get("enabled"):
                available_cash = float(margins["equity"]["enabled"])
                logger.debug(f"🔍 Found funds via equity.enabled: ₹{available_cash:,.2f}")
                return available_cash
            
            # Method 4: Check if funds are in commodity segment
            if margins.get("commodity", {}).get("available", {}).get("cash"):
                available_cash = float(margins["commodity"]["available"]["cash"])
                logger.debug(f"🔍 Found funds via commodity.available.cash: ₹{available_cash:,.2f}")
                return available_cash
            
            # Method 5: Profile meta cash (rarely used)
            if profile and profile.get('meta', {}).get('cash'):
                available_cash = float(profile['meta']['cash'])
                logger.debug(f"🔍 Found funds via profile.meta.cash: ₹{available_cash:,.2f}")
                return available_cash
            
            # Method 6: Check adhoc margin (additional margin provided)
            if margins.get("equity", {}).get("available", {}).get("adhoc_margin"):
                adhoc = float(margins["equity"]["available"]["adhoc_margin"])
                logger.debug(f"🔍 Found adhoc margin: ₹{adhoc:,.2f}")
                if adhoc > 0:
                    return adhoc
            
            # Log debug information if no funds found
            logger.warning("⚠️ No funds detected. Debugging margin structure:")
            if margins.get("equity"):
                logger.info(f"🔍 Equity margins: {json.dumps(margins['equity'], indent=2)}")
            if margins.get("commodity"):
                logger.info(f"🔍 Commodity margins: {json.dumps(margins['commodity'], indent=2)}")
                
        except (ValueError, TypeError) as e:
            logger.error(f"❌ Error parsing fund amounts: {e}")
        
        return 0
    
    async def _initialize_connection(self):
        """Initialize Kite Connect connection with enhanced fund detection"""
        if not KITE_AVAILABLE:
            logger.error("❌ Kite Connect library not available")
            return
        
        try:
            self.kite = KiteConnect(api_key=self.api_key)
            self.kite.set_access_token(self.access_token)
            
            # Test connection
            profile = self.kite.profile()
            self.is_connected = True
            
            # Get margins for fund detection
            margins = self._get_margins_sync()
            available_funds = self._get_available_funds(profile, margins)
            
            logger.info(f"✅ Zerodha connection established for user: {profile.get('user_name', 'Unknown')}")
            logger.info(f"💰 Available funds: ₹{available_funds:,.2f}")
            
            # Log debug info if funds are 0
            if available_funds == 0:
                logger.warning("⚠️ Available funds showing as ₹0.00. Debug information:")
                logger.info(f"🔍 Profile keys: {list(profile.keys())}")
                if margins:
                    logger.info(f"🔍 Margins keys: {list(margins.keys())}")
                    if margins.get("equity"):
                        logger.info(f"🔍 Equity structure: {json.dumps(margins['equity'], indent=2)}")
            
        except Exception as e:
            logger.error(f"❌ Zerodha connection failed: {e}")
            logger.error(f"🔍 Exception type: {type(e).__name__}")
            logger.error(f"🔍 Exception details: {str(e)}")
            self.is_connected = False
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        now = datetime.now()
        
        # Market hours: 9:15 AM to 3:30 PM on weekdays
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        now = datetime.now()
        
        # Reset counter if it's been more than a minute
        if now - self.last_order_time > timedelta(minutes=1):
            self.order_count = 0
        
        return self.order_count < self.max_orders_per_minute
    
    def _get_kite_order_type(self, order_type: OrderType) -> str:
        """Convert our OrderType to Kite Connect format"""
        mapping = {
            OrderType.MARKET: "MARKET",
            OrderType.LIMIT: "LIMIT",
            OrderType.SL_MARKET: "SL-M",
            OrderType.SL_LIMIT: "SL"
        }
        return mapping.get(order_type, "MARKET")
    
    def _get_kite_transaction_type(self, action: str) -> str:
        """Convert action to Kite Connect transaction type"""
        return "BUY" if action.upper() == "BUY" else "SELL"
    
    def _get_kite_exchange(self, exchange: ExchangeType) -> str:
        """Convert exchange type to Kite Connect format"""
        return exchange.value
    
    def _get_margins_sync(self) -> Dict[str, Any]:
        """Get margin information synchronously with caching"""
        if not self.is_connected:
            return {}
        
        # Check cache
        now = datetime.now()
        if (self._cached_margins and self._margins_cache_time and 
            (now - self._margins_cache_time).seconds < self._cache_duration):
            return self._cached_margins
        
        try:
            margins = self.kite.margins()
            self._cached_margins = margins
            self._margins_cache_time = now
            return margins
        except Exception as e:
            logger.error(f"❌ Failed to get margins: {e}")
            return {}
    
    async def place_order(self, 
                         symbol: str,
                         action: str,
                         quantity: int,
                         price: float,
                         order_type: OrderType = OrderType.MARKET,
                         exchange: ExchangeType = ExchangeType.NSE,
                         product: str = "CNC",
                         stop_loss: Optional[float] = None,
                         target: Optional[float] = None) -> OrderResult:
        """
        Place a trading order
        
        Args:
            symbol: Trading symbol (e.g., "RELIANCE", "INFY")
            action: "BUY" or "SELL"
            quantity: Number of shares
            price: Order price (ignored for MARKET orders)
            order_type: Type of order
            exchange: Exchange to place order on
            product: Product type (CNC, MIS, NRML)
            stop_loss: Stop loss price (optional)
            target: Target price (optional)
        
        Returns:
            OrderResult with success status and details
        """
        
        if not self.is_connected:
            return OrderResult(
                success=False,
                error="Not connected to Zerodha",
                message="Zerodha connection not established"
            )
        
        if not self._check_rate_limit():
            return OrderResult(
                success=False,
                error="Rate limit exceeded",
                message="Too many orders in the last minute"
            )
        
        # Validate market hours for regular orders
        if not self.is_market_open() and product != "CNC":
            return OrderResult(
                success=False,
                error="Market closed",
                message="Market is currently closed"
            )
        
        try:
            # Prepare order parameters
            order_params = {
                "tradingsymbol": symbol.upper(),
                "exchange": self._get_kite_exchange(exchange),
                "transaction_type": self._get_kite_transaction_type(action),
                "quantity": quantity,
                "order_type": self._get_kite_order_type(order_type),
                "product": product.upper(),
                "validity": "DAY"
            }
            
            # Add price for non-market orders
            if order_type != OrderType.MARKET:
                order_params["price"] = price
            
            # Add stop loss for SL orders
            if order_type in [OrderType.SL_MARKET, OrderType.SL_LIMIT] and stop_loss:
                order_params["trigger_price"] = stop_loss
            
            logger.info(f"📤 Placing order: {symbol} {action} {quantity} @ ₹{price}")
            
            # Place the order
            order_response = self.kite.place_order(**order_params)
            order_id = order_response.get("order_id")
            
            if order_id:
                # Update counters
                self.order_count += 1
                self.last_order_time = datetime.now()
                
                # Create trade order record
                trade_order = TradeOrder(
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    price=price,
                    order_type=order_type,
                    exchange=exchange,
                    product=product,
                    stop_loss=stop_loss,
                    target=target
                )
                
                self.pending_orders[order_id] = trade_order
                
                # Get order status
                order_status = await self.get_order_status(order_id)
                
                result = OrderResult(
                    success=True,
                    order_id=order_id,
                    status=order_status.get('status', 'PENDING'),
                    message=f"Order placed successfully",
                    price=price,
                    quantity=quantity,
                    timestamp=datetime.now()
                )
                
                logger.info(f"✅ Order placed successfully: {order_id}")
                
                # Place stop loss and target orders if specified
                if stop_loss or target:
                    await self._place_exit_orders(order_id, symbol, action, quantity, stop_loss, target, exchange, product)
                
                return result
            
            else:
                return OrderResult(
                    success=False,
                    error="Invalid response",
                    message="No order ID received from Zerodha"
                )
                
        except Exception as e:
            logger.error(f"❌ Order placement failed: {e}")
            return OrderResult(
                success=False,
                error=str(e),
                message=f"Order placement failed: {e}"
            )
    
    async def _place_exit_orders(self, 
                               parent_order_id: str,
                               symbol: str, 
                               original_action: str,
                               quantity: int,
                               stop_loss: Optional[float],
                               target: Optional[float],
                               exchange: ExchangeType,
                               product: str):
        """Place stop loss and target orders after main order execution"""
        try:
            # Wait a bit for the main order to execute
            await asyncio.sleep(2)
            
            # Check if main order was executed
            main_order_status = await self.get_order_status(parent_order_id)
            if main_order_status.get('status') != 'COMPLETE':
                logger.info(f"⏳ Main order {parent_order_id} not yet executed, skipping exit orders")
                return
            
            # Reverse action for exit orders
            exit_action = "SELL" if original_action.upper() == "BUY" else "BUY"
            
            # Place stop loss order
            if stop_loss:
                sl_result = await self.place_order(
                    symbol=symbol,
                    action=exit_action,
                    quantity=quantity,
                    price=stop_loss,
                    order_type=OrderType.SL_MARKET,
                    exchange=exchange,
                    product=product
                )
                
                if sl_result.success:
                    logger.info(f"✅ Stop loss order placed: {sl_result.order_id}")
                else:
                    logger.error(f"❌ Stop loss order failed: {sl_result.error}")
            
            # Place target order
            if target:
                target_result = await self.place_order(
                    symbol=symbol,
                    action=exit_action,
                    quantity=quantity,
                    price=target,
                    order_type=OrderType.LIMIT,
                    exchange=exchange,
                    product=product
                )
                
                if target_result.success:
                    logger.info(f"✅ Target order placed: {target_result.order_id}")
                else:
                    logger.error(f"❌ Target order failed: {target_result.error}")
                    
        except Exception as e:
            logger.error(f"❌ Exit orders placement failed: {e}")
    
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get status of a specific order"""
        if not self.is_connected:
            return {"status": "UNKNOWN", "error": "Not connected"}
        
        try:
            orders = self.kite.orders()
            for order in orders:
                if order.get("order_id") == order_id:
                    return order
            
            return {"status": "NOT_FOUND", "error": "Order not found"}
            
        except Exception as e:
            logger.error(f"❌ Failed to get order status: {e}")
            return {"status": "ERROR", "error": str(e)}
    
    async def cancel_order(self, order_id: str, variety: str = "regular") -> OrderResult:
        """Cancel a pending order"""
        if not self.is_connected:
            return OrderResult(success=False, error="Not connected")
        
        try:
            self.kite.cancel_order(variety=variety, order_id=order_id)
            
            logger.info(f"✅ Order cancelled: {order_id}")
            
            return OrderResult(
                success=True,
                order_id=order_id,
                message="Order cancelled successfully",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ Order cancellation failed: {e}")
            return OrderResult(
                success=False,
                order_id=order_id,
                error=str(e),
                message=f"Cancellation failed: {e}"
            )
    
    async def modify_order(self, 
                          order_id: str,
                          quantity: Optional[int] = None,
                          price: Optional[float] = None,
                          order_type: Optional[OrderType] = None,
                          variety: str = "regular") -> OrderResult:
        """Modify a pending order"""
        if not self.is_connected:
            return OrderResult(success=False, error="Not connected")
        
        try:
            modify_params = {"variety": variety, "order_id": order_id}
            
            if quantity:
                modify_params["quantity"] = quantity
            if price:
                modify_params["price"] = price
            if order_type:
                modify_params["order_type"] = self._get_kite_order_type(order_type)
            
            self.kite.modify_order(**modify_params)
            
            logger.info(f"✅ Order modified: {order_id}")
            
            return OrderResult(
                success=True,
                order_id=order_id,
                message="Order modified successfully",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ Order modification failed: {e}")
            return OrderResult(
                success=False,
                order_id=order_id,
                error=str(e),
                message=f"Modification failed: {e}"
            )
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        if not self.is_connected:
            return []
        
        try:
            positions = self.kite.positions()
            return positions.get("day", []) + positions.get("net", [])
        except Exception as e:
            logger.error(f"❌ Failed to get positions: {e}")
            return []
    
    async def get_holdings(self) -> List[Dict[str, Any]]:
        """Get current holdings"""
        if not self.is_connected:
            return []
        
        try:
            return self.kite.holdings()
        except Exception as e:
            logger.error(f"❌ Failed to get holdings: {e}")
            return []
    
    async def get_margins(self) -> Dict[str, Any]:
        """Get margin information (async wrapper)"""
        return self._get_margins_sync()
    
    async def get_order_history(self, days: int = 1) -> List[Dict[str, Any]]:
        """Get order history for specified days"""
        if not self.is_connected:
            return []
        
        try:
            return self.kite.orders()
        except Exception as e:
            logger.error(f"❌ Failed to get order history: {e}")
            return []
    
    async def get_connection_status(self) -> Dict[str, Any]:
        """Get connection and account status with enhanced fund detection"""
        if not self.is_connected:
            return {
                "connected": False,
                "error": "Not connected to Zerodha"
            }
        
        try:
            profile = self.kite.profile()
            margins = self._get_margins_sync()  # Use sync method instead of await
            
            # Get available funds using enhanced detection
            available_cash = self._get_available_funds(profile, margins)
            
            return {
                "connected": True,
                "user_name": profile.get("user_name"),
                "user_id": profile.get("user_id"),
                "email": profile.get("email"),
                "broker": profile.get("broker"),
                "available_cash": available_cash,
                "market_open": self.is_market_open(),
                "order_count_today": self.order_count,
                "rate_limit_remaining": self.max_orders_per_minute - self.order_count
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get connection status: {e}")
            return {
                "connected": False,
                "error": str(e)
            }
    
    def get_order_summary(self) -> Dict[str, Any]:
        """Get summary of orders placed through this engine"""
        return {
            "pending_orders": len(self.pending_orders),
            "completed_orders": len(self.completed_orders),
            "orders_today": self.order_count,
            "last_order_time": self.last_order_time.isoformat() if self.last_order_time else None,
            "rate_limit_remaining": self.max_orders_per_minute - self.order_count
        }


# Example usage and integration
async def example_usage():
    """Example of how to use the order engine"""
    
    # Initialize order engine
    order_engine = ZerodhaOrderEngine(
        api_key="your_api_key",
        access_token="your_access_token",
        enable_sandbox=True  # Use sandbox for testing
    )
    
    # Wait for connection
    await asyncio.sleep(2)
    
    # Check connection
    status = await order_engine.get_connection_status()
    print(f"Connection status: {status}")
    
    if status.get("connected"):
        # Place a market order
        result = await order_engine.place_order(
            symbol="RELIANCE",
            action="BUY",
            quantity=1,
            price=2500.0,  # Will be ignored for market orders
            order_type=OrderType.MARKET,
            product="CNC"
        )
        
        print(f"Order result: {result}")
        
        if result.success:
            # Get order status
            order_status = await order_engine.get_order_status(result.order_id)
            print(f"Order status: {order_status}")

if __name__ == "__main__":
    asyncio.run(example_usage())