'use client';
import React, { useState, useEffect } from 'react';
import Navigation from '@/components/Navigation';

export default function DashboardPage() {
  const [signals, setSignals] = useState([]);
  const [analytics, setAnalytics] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(new Date());

  // Check backend connection
  useEffect(() => {
    const checkConnection = async () => {
      try {
        const response = await fetch('http://localhost:8000/health');
        if (response.ok) {
          setIsConnected(true);
          const data = await response.json();
          setAnalytics(data.performance);
        }
      } catch (error) {
        setIsConnected(false);
        console.log('Backend not connected, using demo data');
      }
    };

    checkConnection();
    const interval = setInterval(checkConnection, 10000); // Check every 10 seconds

    return () => clearInterval(interval);
  }, []);

  // Mock data when backend is not connected
  useEffect(() => {
    if (!isConnected) {
      const interval = setInterval(() => {
        setLastUpdate(new Date());
        setAnalytics({
          daily: {
            signals_generated: 3,
            win_rate: 67.5,
            total_pnl: 1250.75,
            average_confidence: 72.3,
            system_uptime_hours: 18.5
          }
        });
      }, 5000);

      return () => clearInterval(interval);
    }
  }, [isConnected]);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation */}
      <Navigation />

      {/* Status Bar */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-12">
            <div className="flex items-center space-x-4">
              <h2 className="text-lg font-semibold text-gray-900">Research Dashboard</h2>
            </div>
            <div className="flex items-center space-x-4">
              <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm ${
                isConnected ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
              }`}>
                <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
                <span>{isConnected ? 'Backend Connected' : 'Demo Mode'}</span>
              </div>
              <div className="text-sm text-gray-500">
                {lastUpdate.toLocaleTimeString()}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Development Notice */}
        <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex items-start space-x-3">
            <svg className="w-6 h-6 text-blue-600 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <div>
              <h3 className="font-medium text-blue-900">Research & Development Mode</h3>
              <p className="text-sm text-blue-700 mt-1">
                Personal testing environment. No public trading advice provided. 
                All signals are for research and algorithm development only.
              </p>
            </div>
          </div>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white p-6 rounded-xl shadow-sm border">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Today's Signals</p>
                <p className="text-2xl font-bold text-gray-900">{analytics?.daily?.signals_generated || 0}</p>
              </div>
              <div className="bg-blue-100 rounded-lg p-3">
                <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
            </div>
          </div>

          <div className="bg-white p-6 rounded-xl shadow-sm border">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Algorithm Accuracy</p>
                <p className="text-2xl font-bold text-green-600">{analytics?.daily?.win_rate || 0}%</p>
              </div>
              <div className="bg-green-100 rounded-lg p-3">
                <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
            </div>
          </div>

          <div className="bg-white p-6 rounded-xl shadow-sm border">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Paper Trading P&L</p>
                <p className={`text-2xl font-bold ${(analytics?.daily?.total_pnl || 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  ₹{analytics?.daily?.total_pnl?.toLocaleString() || '0'}
                </p>
              </div>
              <div className="bg-purple-100 rounded-lg p-3">
                <svg className="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1" />
                </svg>
              </div>
            </div>
          </div>

          <div className="bg-white p-6 rounded-xl shadow-sm border">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Avg Confidence</p>
                <p className="text-2xl font-bold text-blue-600">{analytics?.daily?.average_confidence || 0}%</p>
              </div>
              <div className="bg-orange-100 rounded-lg p-3">
                <svg className="w-6 h-6 text-orange-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Recent Signals */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-xl shadow-sm border">
              <div className="p-6 border-b border-gray-200">
                <h2 className="text-lg font-semibold text-gray-900">Research Signals</h2>
                <p className="text-sm text-gray-500">AI-generated signals for personal analysis only</p>
              </div>
              <div className="p-6">
                <div className="space-y-4">
                  {[
                    { symbol: 'RELIANCE', action: 'BUY', price: 2485, target: 2550, confidence: 78, time: '10:30 AM', status: 'active' },
                    { symbol: 'TCS', action: 'SELL', price: 2510, target: 2450, confidence: 72, time: '11:45 AM', status: 'completed' },
                    { symbol: 'HDFC', action: 'BUY', price: 1670, target: 1720, confidence: 65, time: '2:15 PM', status: 'active' }
                  ].map((signal, index) => (
                    <div key={index} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                      <div className="flex items-center space-x-4">
                        <div className={`px-2 py-1 rounded text-xs font-medium ${
                          signal.action === 'BUY' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                        }`}>
                          {signal.action}
                        </div>
                        <div>
                          <div className="font-medium text-gray-900">{signal.symbol}</div>
                          <div className="text-sm text-gray-500">Entry: ₹{signal.price} → Target: ₹{signal.target}</div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-sm font-medium text-gray-900">{signal.confidence}% confidence</div>
                        <div className="text-xs text-gray-500">{signal.time}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* System Health & Controls */}
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-sm border">
              <div className="p-6 border-b border-gray-200">
                <h3 className="text-lg font-semibold text-gray-900">System Status</h3>
              </div>
              <div className="p-6 space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Backend API</span>
                  <span className="flex items-center space-x-2">
                    <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
                    <span className={`text-sm ${isConnected ? 'text-green-600' : 'text-red-600'}`}>
                      {isConnected ? 'Online' : 'Offline'}
                    </span>
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Signal Generator</span>
                  <span className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-sm text-green-600">Active</span>
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Analytics</span>
                  <span className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-sm text-green-600">Tracking</span>
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Uptime</span>
                  <span className="text-sm text-gray-900">{analytics?.daily?.system_uptime_hours?.toFixed(1) || 0}h</span>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-xl shadow-sm border">
              <div className="p-6 border-b border-gray-200">
                <h3 className="text-lg font-semibold text-gray-900">Research Tools</h3>
              </div>
              <div className="p-6 space-y-3">
                <a href="http://localhost:8000/docs" target="_blank" rel="noopener noreferrer"
                   className="block w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors text-center">
                  View API Documentation
                </a>
                <a href="http://localhost:8000/health" target="_blank" rel="noopener noreferrer"
                   className="block w-full border border-gray-300 text-gray-700 py-2 px-4 rounded-lg hover:bg-gray-50 transition-colors text-center">
                  System Health Check
                </a>
                <button className="w-full border border-gray-300 text-gray-700 py-2 px-4 rounded-lg hover:bg-gray-50 transition-colors">
                  Download Test Data
                </button>
              </div>
            </div>

            {/* Future Features */}
            <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl border border-blue-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-3">Development Roadmap</h3>
              <div className="space-y-2 text-sm text-gray-600">
                <div className="flex items-center space-x-2">
                  <svg className="w-4 h-4 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  <span>SEBI Research Analyst Registration</span>
                </div>
                <div className="flex items-center space-x-2">
                  <svg className="w-4 h-4 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  <span>Subscription & Payment Gateway</span>
                </div>
                <div className="flex items-center space-x-2">
                  <svg className="w-4 h-4 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  <span>Public Signal Distribution</span>
                </div>
                <div className="flex items-center space-x-2">
                  <svg className="w-4 h-4 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  <span>Mobile Application</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}