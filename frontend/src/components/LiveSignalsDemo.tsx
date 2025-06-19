'use client';

import { useState, useEffect, useCallback } from 'react';

interface Signal {
  id: string;
  symbol: string;
  action: 'BUY' | 'SELL';
  entry_price: number;
  target_price: number;
  stop_loss: number;
  confidence: number;
  sentiment_score: number;
  created_at: string;
  status: string;
}

interface LiveSignalsDemoProps {
  onConnectionChange: (connected: boolean) => void;
}

export default function LiveSignalsDemo({ onConnectionChange }: LiveSignalsDemoProps) {
  const [signals, setSignals] = useState<Signal[]>([]);
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected'>('disconnected');

  const connectWebSocket = useCallback(() => {
    try {
      setConnectionStatus('connecting');
      const websocket = new WebSocket('ws://localhost:8000/ws/signals');
      
      websocket.onopen = () => {
        console.log('✅ Connected to TradeMind AI signals');
        setConnectionStatus('connected');
        onConnectionChange(true);
      };
      
      websocket.onmessage = (event) => {
        const message = JSON.parse(event.data);
        console.log('📡 Received:', message);
        
        if (message.type === 'new_signal') {
          setSignals(prev => [message.data, ...prev].slice(0, 5)); // Keep last 5 signals
        }
      };
      
      websocket.onclose = () => {
        console.log('❌ WebSocket connection closed');
        setConnectionStatus('disconnected');
        onConnectionChange(false);
        
        // Reconnect after 3 seconds
        setTimeout(connectWebSocket, 3000);
      };
      
      websocket.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        setConnectionStatus('disconnected');
        onConnectionChange(false);
      };
      
      setWs(websocket);
    } catch (error) {
      console.error('❌ Failed to connect:', error);
      setConnectionStatus('disconnected');
      onConnectionChange(false);
    }
  }, [onConnectionChange]);

  useEffect(() => {
    connectWebSocket();
    
    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, [connectWebSocket]);

  const formatCurrency = (amount: number) => `₹${amount.toLocaleString()}`;
  const formatPercentage = (value: number) => `${(value * 100).toFixed(1)}%`;
  const formatTime = (timestamp: string) => new Date(timestamp).toLocaleTimeString();

  return (
    <div className="bg-white rounded-xl shadow-lg border p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-2xl font-bold text-gray-900">Live Trading Signals</h3>
          <p className="text-gray-600 mt-1">Real-time AI-generated signals from our backend</p>
        </div>
        
        <div className="flex items-center space-x-3">
          <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm ${
            connectionStatus === 'connected' ? 'bg-green-100 text-green-800' :
            connectionStatus === 'connecting' ? 'bg-yellow-100 text-yellow-800' :
            'bg-red-100 text-red-800'
          }`}>
            <div className={`w-2 h-2 rounded-full ${
              connectionStatus === 'connected' ? 'bg-green-500' :
              connectionStatus === 'connecting' ? 'bg-yellow-500' :
              'bg-red-500'
            }`}></div>
            <span className="capitalize">{connectionStatus}</span>
          </div>
          
          {connectionStatus === 'disconnected' && (
            <button 
              onClick={connectWebSocket}
              className="bg-blue-600 hover:bg-blue-700 text-white px-3 py-1 rounded text-sm"
            >
              Reconnect
            </button>
          )}
        </div>
      </div>

      {/* Signals Display */}
      <div className="space-y-4">
        {signals.length > 0 ? (
          signals.map((signal) => (
            <div key={signal.id} className={`border rounded-lg p-4 ${
              signal.action === 'BUY' ? 'border-green-200 bg-green-50' : 'border-red-200 bg-red-50'
            }`}>
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-3">
                  <span className="text-lg font-bold text-gray-900">{signal.symbol}</span>
                  <span className={`px-2 py-1 rounded text-sm font-medium ${
                    signal.action === 'BUY' ? 'bg-green-600 text-white' : 'bg-red-600 text-white'
                  }`}>
                    {signal.action}
                  </span>
                </div>
                <div className="text-right">
                  <div className="text-sm text-gray-500">{formatTime(signal.created_at)}</div>
                  <div className="text-sm font-medium">
                    Confidence: {formatPercentage(signal.confidence)}
                  </div>
                </div>
              </div>
              
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div>
                  <span className="text-gray-500">Entry:</span>
                  <div className="font-semibold">{formatCurrency(signal.entry_price)}</div>
                </div>
                <div>
                  <span className="text-gray-500">Target:</span>
                  <div className="font-semibold text-green-600">{formatCurrency(signal.target_price)}</div>
                </div>
                <div>
                  <span className="text-gray-500">Stop Loss:</span>
                  <div className="font-semibold text-red-600">{formatCurrency(signal.stop_loss)}</div>
                </div>
              </div>
            </div>
          ))
        ) : (
          <div className="text-center py-12 text-gray-500">
            <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
              </svg>
            </div>
            <p className="text-lg">Waiting for signals...</p>
            <p className="text-sm">
              {connectionStatus === 'connected' 
                ? 'Connected to backend. Signals will appear here every 30 seconds.' 
                : 'Make sure your backend is running on localhost:8000'
              }
            </p>
          </div>
        )}
      </div>

      {/* Demo Notice */}
      <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <div className="flex items-start space-x-2">
          <svg className="w-5 h-5 text-blue-600 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <div>
            <h4 className="font-medium text-blue-900">Demo Mode</h4>
            <p className="text-sm text-blue-700">
              This is showing demo signals from your FastAPI backend. In production, these will be replaced 
              with real AI-generated signals using your 63.8% accuracy model.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}