from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from datetime import datetime
import logging
import json
from typing import List, Dict, Any

from app.api.dependencies import get_service_manager
from app.core.services.service_manager import CorrectedServiceManager

router = APIRouter()
logger = logging.getLogger(__name__)

class WebSocketManager:
    """WebSocket connection manager for real-time dashboard updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect_websocket(self, websocket: WebSocket, service_manager: CorrectedServiceManager):
        """Connect a WebSocket client"""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Send welcome message with enhanced features
        await websocket.send_json({
            "type": "connected",
            "production_mode": True,
            "demo_signals_disabled": True,
            "enhanced_features": {
                "interactive_trading": service_manager.interactive_trading_active,
                "order_execution": service_manager.system_health["order_execution"],
                "regime_detection": service_manager.system_health["regime_detection"],
                "backtesting": service_manager.system_health["backtesting"],
                "news_intelligence": service_manager.system_health["news_intelligence"],
                "breaking_news_alerts": service_manager.system_health["breaking_news_alerts"],
                "news_signal_integration": service_manager.system_health["news_signal_integration"]  # FIXED
            },
            "system_health": service_manager.system_health,
            "timestamp": datetime.now().isoformat()
        })
    
    async def disconnect_websocket(self, websocket: WebSocket):
        """Disconnect a WebSocket client"""
        try:
            self.active_connections.remove(websocket)
        except ValueError:
            pass
    
    async def broadcast_to_dashboard(self, message: Dict[str, Any]):
        """Broadcast message to all connected WebSocket clients"""
        if self.active_connections:
            disconnected = []
            for connection in self.active_connections:
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.append(connection)
            
            # Remove disconnected clients
            for conn in disconnected:
                await self.disconnect_websocket(conn)

# Create WebSocket manager instance
websocket_manager = WebSocketManager()

@router.websocket("/ws/signals")
async def websocket_endpoint(websocket: WebSocket):
    """Enhanced WebSocket endpoint with fixed news-signal integration updates"""
    # Get service manager for this connection
    service_manager = get_service_manager()
    
    await websocket_manager.connect_websocket(websocket, service_manager)
    
    try:
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "enhanced": True,
                        "production_mode": True,
                        "demo_signals_disabled": True,
                        "interactive_trading": service_manager.interactive_trading_active,
                        "news_intelligence": service_manager.system_health.get("news_intelligence", False),
                        "news_signal_integration": service_manager.system_health.get("news_signal_integration", False),  # FIXED
                        "timestamp": datetime.now().isoformat()
                    })
                
                elif message.get("type") == "get_status":
                    analytics_service = service_manager.get_analytics_service()
                    performance = analytics_service.get_performance_summary()
                    await websocket.send_json({
                        "type": "status_update",
                        "system_health": service_manager.system_health,
                        "performance": performance,
                        "interactive_trading": service_manager.interactive_trading_active,
                        "news_monitoring": service_manager.news_monitoring_active,
                        "news_signal_integration": service_manager.system_health.get("news_signal_integration", False),  # FIXED
                        "current_regime": str(service_manager.current_regime),
                        "regime_confidence": service_manager.regime_confidence,
                        "production_mode": True,
                        "demo_signals_disabled": True,
                        "timestamp": datetime.now().isoformat()
                    })
                
                elif message.get("type") == "get_news_status":
                    # FIXED: Real-time news status
                    news_stats = {}
                    if service_manager.news_intelligence:
                        news_stats = {
                            "news_intelligence_available": True,
                            "news_monitoring_active": service_manager.news_monitoring_active,
                            "articles_processed": 0  # Will be updated when news processing is implemented
                        }
                    
                    await websocket.send_json({
                        "type": "news_status_update",
                        "news_intelligence_active": service_manager.system_health.get("news_intelligence", False),
                        "news_signal_integration_active": service_manager.system_health.get("news_signal_integration", False),
                        "integration_stats": news_stats,
                        "timestamp": datetime.now().isoformat()
                    })
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        await websocket_manager.disconnect_websocket(websocket) 