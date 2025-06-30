import asyncio
from datetime import datetime, time as dt_time
from typing import List, Dict, Any, Union, Optional
import logging
import traceback
from app.core.config import settings
from utils.profiling import profile_timing
import time

logger = logging.getLogger(__name__)

class SignalService:
    def __init__(self, *, news_intelligence, signal_generator, analytics_service, signal_logger, telegram_service, regime_detector, backtest_engine, order_engine, webhook_handler, telegram_integration, enhanced_market_service, system_health, current_regime, regime_confidence, premarket_opportunities, priority_signals_queue, interactive_trading_active, signal_generation_active, premarket_analysis_active, priority_trading_active, news_monitoring_active, news_signal_integration=None):
        self.news_intelligence = news_intelligence
        self.signal_generator = signal_generator
        self.analytics_service = analytics_service
        self.signal_logger = signal_logger
        self.telegram_service = telegram_service
        self.regime_detector = regime_detector
        self.backtest_engine = backtest_engine
        self.order_engine = order_engine
        self.webhook_handler = webhook_handler
        self.telegram_integration = telegram_integration
        self.enhanced_market_service = enhanced_market_service
        self.system_health = system_health
        self.current_regime = current_regime
        self.regime_confidence = regime_confidence
        self.premarket_opportunities = premarket_opportunities
        self.priority_signals_queue = priority_signals_queue
        self.interactive_trading_active = interactive_trading_active
        self.signal_generation_active = signal_generation_active
        self.premarket_analysis_active = premarket_analysis_active
        self.priority_trading_active = priority_trading_active
        self.news_monitoring_active = news_monitoring_active
        self.news_signal_integration = news_signal_integration
        self.signal_generation_task = None
        self._last_signal_time = None

    def convert_signal_record_to_dict(self, signal_record) -> Dict[str, Any]:
        try:
            return {
                "signal_id": getattr(signal_record, 'signal_id', None),
                "symbol": getattr(signal_record, 'ticker', None),
                "action": getattr(signal_record, 'direction', None),
                "entry_price": getattr(signal_record, 'entry_price', None),
                "target_price": getattr(signal_record, 'target_price', None),
                "stop_loss": getattr(signal_record, 'stop_loss', None),
                "confidence": getattr(signal_record, 'ml_confidence', None),
                "technical_score": getattr(signal_record, 'technical_score', None),
                "sentiment_score": getattr(signal_record, 'sentiment_score', None),
                "final_score": getattr(signal_record, 'final_score', None),
                "risk_reward_ratio": getattr(signal_record, 'risk_reward_ratio', None),
                "quantity": getattr(signal_record, 'position_size_suggested', None),
                "capital_at_risk": getattr(signal_record, 'capital_at_risk', None),
                "model_version": getattr(signal_record, 'model_version', None),
                "signal_source": getattr(signal_record, 'signal_source', None),
                "notes": getattr(signal_record, 'notes', None),
                "timestamp": getattr(signal_record, 'timestamp', datetime.now()),
                "created_at": getattr(signal_record, 'timestamp', datetime.now()).isoformat(),
                "status": "active",
                "signal_type": "ML_PRODUCTION",
                "risk_level": "MEDIUM",
                "stock_universe": "NIFTY_100"
            }
        except Exception as e:
            logger.error(f"❌ Signal conversion failed: {e}")
            return {
                "symbol": getattr(signal_record, 'ticker', 'UNKNOWN'),
                "action": "BUY",
                "entry_price": getattr(signal_record, 'entry_price', 0.0),
                "target_price": getattr(signal_record, 'target_price', 0.0),
                "stop_loss": getattr(signal_record, 'stop_loss', 0.0),
                "confidence": getattr(signal_record, 'ml_confidence', 0.0),
                "quantity": 1,
                "timestamp": datetime.now(),
                "created_at": datetime.now().isoformat(),
                "status": "active"
            }

    async def generate_signals(self) -> List[Union[Dict, Any]]:
        signals = []
        start_time = time.time()
        logger.info("🚀 Starting signal generation process...")
        
        try:
            if self.signal_generator:
                logger.info("📊 Calling regime-aware signal generator...")
                signals = await self.signal_generator.generate_regime_aware_signals(
                    self.current_regime, self.regime_confidence
                )
                
                if signals:
                    logger.info(f"✅ Generated {len(signals)} ML signals")
                else:
                    logger.info("📊 No ML signals generated - all stocks filtered by confidence thresholds")
                
                # Add news-triggered signals if available
                if self.news_signal_integration:
                    logger.info("📰 Checking for news-triggered signals...")
                    news_signals = await self.get_news_triggered_signals()
                    if news_signals:
                        signals.extend(news_signals)
                        logger.info(f"📰 Added {len(news_signals)} news-triggered signals")
                    else:
                        logger.info("📰 No news-triggered signals found")
                
                # Fallback signal generation if no signals and market is open
                if not signals and self._is_market_open():
                    logger.warning("⚠️ No signals generated, attempting fallback signal generation...")
                    fallback_signals = await self._generate_fallback_signals()
                    if fallback_signals:
                        signals.extend(fallback_signals)
                        logger.info(f"🔄 Generated {len(fallback_signals)} fallback signals")
                
                # Log final results
                end_time = time.time()
                processing_time = end_time - start_time
                
                if signals:
                    logger.info(f"🎯 Signal generation completed in {processing_time:.2f}s: {len(signals)} total signals")
                    # Log each signal for monitoring
                    for i, signal in enumerate(signals):
                        if hasattr(signal, 'ticker'):
                            logger.info(f"📋 Signal {i+1}: {signal.ticker} | Confidence: {getattr(signal, 'ml_confidence', 0.0):.1%}")
                        elif isinstance(signal, dict):
                            logger.info(f"📋 Signal {i+1}: {signal.get('symbol', 'UNKNOWN')} | Confidence: {signal.get('confidence', 0.0):.1%}")
                else:
                    logger.info(f"📊 Signal generation completed in {processing_time:.2f}s: No signals generated")
                
            else:
                logger.warning("⚠️ Signal generator not available - no signals generated")
                
            return signals
            
        except Exception as e:
            end_time = time.time()
            processing_time = end_time - start_time
            logger.error(f"❌ Signal generation failed after {processing_time:.2f}s: {e}")
            return []
    
    def _is_market_open(self) -> bool:
        """Check if market is currently open"""
        try:
            current_time = datetime.now().time()
            market_open = dt_time(9, 15)  # 9:15 AM
            market_close = dt_time(15, 30)  # 3:30 PM
            
            return market_open <= current_time <= market_close
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return False
    
    async def _generate_fallback_signals(self) -> List[Union[Dict, Any]]:
        """Generate fallback signals without news intelligence"""
        try:
            logger.info("🔄 Generating fallback signals using basic technical analysis...")
            
            # Get basic market data
            if not self.enhanced_market_service:
                logger.warning("⚠️ Market service not available for fallback signals")
                return []
            
            # Get top Nifty stocks with basic analysis
            opportunities = await self.enhanced_market_service.get_top_opportunity_stocks(limit=10)
            
            fallback_signals = []
            for opportunity in opportunities:
                try:
                    symbol = opportunity.get("symbol")
                    if not symbol:
                        continue
                    
                    # Create basic signal
                    signal = {
                        "symbol": symbol,
                        "action": "BUY" if opportunity.get("gap_pct", 0) > 0 else "SELL",
                        "entry_price": opportunity.get("current_price", 0.0),
                        "target_price": opportunity.get("current_price", 0.0) * 1.02,  # 2% target
                        "stop_loss": opportunity.get("current_price", 0.0) * 0.98,  # 2% stop
                        "confidence": 0.6,  # Moderate confidence for fallback
                        "quantity": 1,
                        "timestamp": datetime.now(),
                        "created_at": datetime.now().isoformat(),
                        "status": "active",
                        "signal_source": "FALLBACK_TECHNICAL",
                        "notes": "Fallback signal generated without news intelligence"
                    }
                    
                    fallback_signals.append(signal)
                    logger.info(f"🔄 Fallback signal: {symbol} {signal['action']} @ ₹{signal['entry_price']}")
                    
                except Exception as e:
                    logger.error(f"Error creating fallback signal: {e}")
                    continue
            
            return fallback_signals
            
        except Exception as e:
            logger.error(f"❌ Fallback signal generation failed: {e}")
            return []

    async def get_news_triggered_signals(self) -> List[Dict]:
        try:
            if not self.news_signal_integration:
                return []
            stats = self.news_signal_integration.get_integration_stats()
            return []
        except Exception as e:
            logger.error(f"❌ News signal retrieval failed: {e}")
            return []

    async def process_signal(self, signal: Union[Dict, Any], is_priority: bool = False):
        try:
            if hasattr(signal, '__dict__'):
                signal_dict = self.convert_signal_record_to_dict(signal)
                logger.debug(f"🔄 Converted SignalRecord to Dict for processing: {signal_dict.get('symbol', 'UNKNOWN')}")
            elif isinstance(signal, dict):
                signal_dict = signal.copy()
            else:
                logger.error(f"❌ Unknown signal type: {type(signal)}")
                return
            signal_dict["is_priority_signal"] = is_priority
            signal_dict["processing_timestamp"] = datetime.now().isoformat()
            signal_dict["current_regime"] = str(self.current_regime)
            signal_dict["regime_confidence"] = self.regime_confidence
            signal_dict["production_signal"] = True
            signal_dict["demo_signal"] = False
            if signal_dict.get("is_news_signal"):
                signal_dict["enhanced_by_news"] = True
                signal_dict["news_triggered"] = True
                await self.analytics_service.track_news_triggered_signal()
            elif self.news_signal_integration and signal_dict.get("symbol"):
                symbol = signal_dict["symbol"]
                if hasattr(self.news_signal_integration, 'current_news_data'):
                    news_data = self.news_signal_integration.current_news_data
                    symbol_sentiment = news_data.get("sentiment_analysis", {}).get("symbol_sentiment", {}).get(symbol, 0.0)
                    if abs(symbol_sentiment) > 0.1:
                        signal_dict["news_sentiment"] = symbol_sentiment
                        signal_dict["enhanced_by_news"] = True
                        await self.analytics_service.track_enhanced_ml_signal()
            await self.analytics_service.track_signal_generated(signal_dict)
            telegram_success = False
            if self.telegram_service and self.telegram_service.is_configured():
                if self.interactive_trading_active:
                    quantity = signal_dict.get("quantity", 1)
                    telegram_success = await self.telegram_service.send_signal_with_approval(
                        signal_dict, quantity
                    )
                else:
                    telegram_success = await self.telegram_service.send_signal_notification(signal_dict)
                await self.analytics_service.track_telegram_sent(telegram_success, signal_dict)
            # Broadcast to dashboard (handled by WebSocket manager)
            # await websocket_manager.broadcast_to_dashboard({...})
            signal_type = "NEWS-TRIGGERED" if signal_dict.get("is_news_signal") else "ML-ENHANCED"
            logger.info(f"📈 PRODUCTION {signal_type} Signal processed: {signal_dict['symbol']} {signal_dict['action']} "
                       f"@ ₹{signal_dict['entry_price']} (Confidence: {signal_dict['confidence']:.1%})")
        except Exception as e:
            logger.error(f"❌ Signal processing failed: {e}")
            logger.error(traceback.format_exc())

    async def start_signal_generation(self):
        if self.signal_generation_active:
            return
        self.signal_generation_active = True
        self.signal_generation_task = asyncio.create_task(self.signal_generation_loop())
        logger.info("✅ Signal generation started (PRODUCTION MODE - No Demo Signals)")

    async def stop_signal_generation(self):
        self.signal_generation_active = False
        self.news_monitoring_active = False
        if self.signal_generation_task:
            self.signal_generation_task.cancel()
            try:
                await self.signal_generation_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 Signal generation and news monitoring stopped")

    async def signal_generation_loop(self):
        while self.signal_generation_active:
            try:
                await self.check_premarket_analysis_trigger()
                await self.check_priority_trading_trigger()
                await self.check_regular_signal_generation()
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Signal generation loop error: {e}")
                await asyncio.sleep(60)

    async def check_premarket_analysis_trigger(self):
        now = datetime.now()
        current_time = now.time()
        if (now.weekday() < 5 and 
            dt_time(8, 0) <= current_time <= dt_time(9, 15) and
            not self.premarket_analysis_active and
            self.enhanced_market_service):
            await self.run_premarket_analysis()

    async def check_priority_trading_trigger(self):
        now = datetime.now()
        current_time = now.time()
        if (now.weekday() < 5 and 
            dt_time(9, 15) <= current_time <= dt_time(9, 45) and
            not self.priority_trading_active):
            await self.run_priority_trading()

    async def check_regular_signal_generation(self):
        now = datetime.now()
        current_time = now.time()
        if (now.weekday() < 5 and 
            dt_time(9, 45) <= current_time <= dt_time(15, 30)):
            await self.run_regular_signal_generation()

    async def run_premarket_analysis(self):
        try:
            self.premarket_analysis_active = True
            logger.info("🌅 Running pre-market analysis...")
            if self.enhanced_market_service:
                analysis_result = await self.enhanced_market_service.run_premarket_analysis()
                self.premarket_opportunities = analysis_result.get("top_opportunities", [])
                await self.analytics_service.track_premarket_analysis(len(self.premarket_opportunities))
                if self.telegram_service and self.telegram_service.is_configured():
                    news_status = "✅ ACTIVE" if self.news_monitoring_active else "❌ DISABLED"
                    integration_status = "✅ ACTIVE" if self.system_health.get("news_signal_integration") else "❌ DISABLED"
                    summary_message = (
                        f"🌅 <b>PRE-MARKET ANALYSIS COMPLETE (PRODUCTION)</b>\n\n"
                        f"📊 Opportunities: {analysis_result.get('total_opportunities', 0)}\n"
                        f"🎯 Strong Buy: {analysis_result.get('strong_buy_count', 0)}\n"
                        f"📈 Buy: {analysis_result.get('buy_count', 0)}\n"
                        f"👀 Watch: {analysis_result.get('watch_count', 0)}\n"
                        f"📰 News Intelligence: {news_status}\n"
                        f"🔗 News-Signal Integration: {integration_status}\n\n"
                        f"🔥 Demo signals: DISABLED\n"
                        f"🤖 Production mode: {'ACTIVE' if self.interactive_trading_active else 'DISABLED'}"
                    )
                    await self.telegram_service.send_message(summary_message)
            logger.info(f"✅ Pre-market analysis: {len(self.premarket_opportunities)} opportunities")
        except Exception as e:
            logger.error(f"❌ Pre-market analysis failed: {e}")
        finally:
            current_time = datetime.now().time()
            if current_time > dt_time(9, 15):
                self.premarket_analysis_active = False

    async def run_priority_trading(self):
        try:
            self.priority_trading_active = True
            logger.info("⚡ Priority trading period active (PRODUCTION MODE)...")
            for i in range(3):
                signals = await self.generate_signals()
                for signal in signals:
                    await self.process_signal(signal, is_priority=True)
                if i < 2:
                    await asyncio.sleep(10)
        except Exception as e:
            logger.error(f"❌ Priority trading failed: {e}")
        finally:
            current_time = datetime.now().time()
            if current_time > dt_time(9, 45):
                self.priority_trading_active = False

    async def run_regular_signal_generation(self):
        try:
            last_signal_time = self._last_signal_time
            now = datetime.now()
            if (last_signal_time is None or 
                (now - last_signal_time).total_seconds() >= settings.signal_generation_interval):
                signals = await self.generate_signals()
                for signal in signals:
                    await self.process_signal(signal, is_priority=False)
                self._last_signal_time = now
        except Exception as e:
            logger.error(f"❌ Regular signal generation failed: {e}")

    @profile_timing("signal_generation_pipeline")
    async def generate_signal(self, request):
        signal_input = await self._fetch_market_data(request)
        news_features = await self._process_news_intelligence(signal_input)
        prediction = await self._run_ml_inference(signal_input, news_features)
        signal = self._create_signal_object(prediction, news_features)
        await self._postprocess_signal(signal)
        return signal

    @profile_timing("fetch_market_data")
    async def _fetch_market_data(self, request):
        # ... fetch from market data APIs ...
        pass

    @profile_timing("process_news_intelligence")
    async def _process_news_intelligence(self, signal_input):
        # ... call news intelligence system ...
        pass

    @profile_timing("run_ml_inference")
    async def _run_ml_inference(self, signal_input, news_features):
        # ... run ML model ...
        pass

    @profile_timing("postprocess_signal")
    async def _postprocess_signal(self, signal):
        # ... DB/logging ...
        pass 