// ================================================================
// File 1: frontend/src/components/analytics/AnalyticsDashboard.tsx
// ================================================================

'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

interface DashboardAnalytics {
  daily: {
    signals_generated: number;
    signals_sent: number;
    win_rate: number;
    total_pnl: number;
    average_confidence: number;
    telegram_success_rate: number;
    system_uptime_hours: number;
    last_signal_time: string | null;
    status: string;
  };
  system_health: {
    telegram_configured: boolean;
    signal_generation_active: boolean;
    last_activity: string | null;
    error_rate: number;
  };
  configuration: {
    max_signals_per_day: number;
    min_confidence_threshold: number;
    signal_interval_seconds: number;
    telegram_enabled: boolean;
    zerodha_enabled: boolean;
  };
  recent_performance: {
    signals_today: number;
    success_rate: number;
    profit_loss: number;
    avg_confidence: number;
  };
}

export default function AnalyticsDashboard() {
  const [analytics, setAnalytics] = useState<DashboardAnalytics | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  useEffect(() => {
    fetchAnalytics();
    
    // Update every 30 seconds
    const interval = setInterval(fetchAnalytics, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchAnalytics = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/analytics/dashboard');
      const data = await response.json();
      setAnalytics(data);
      setLastUpdate(new Date());
      setLoading(false);
    } catch (error) {
      console.error('Failed to fetch analytics:', error);
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        <span className="ml-2 text-gray-600">Loading analytics...</span>
      </div>
    );
  }

  if (!analytics) {
    return (
      <div className="text-center py-8">
        <p className="text-red-600">Failed to load analytics data</p>
        <button 
          onClick={fetchAnalytics}
          className="mt-2 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">System Analytics</h2>
          <p className="text-gray-600">
            Last updated: {lastUpdate.toLocaleTimeString()}
          </p>
        </div>
        
        <div className="flex items-center space-x-2">
          <SystemStatus status={analytics.daily.status} />
          <button 
            onClick={fetchAnalytics}
            className="p-2 text-gray-500 hover:text-gray-700 rounded-full hover:bg-gray-100"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          </button>
        </div>
      </div>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Signals Today"
          value={analytics.daily.signals_generated}
          maxValue={analytics.configuration.max_signals_per_day}
          suffix={`/${analytics.configuration.max_signals_per_day}`}
          icon="📡"
          color="blue"
        />
        
        <MetricCard
          title="Win Rate"
          value={`${analytics.daily.win_rate}%`}
          icon="🎯"
          color={analytics.daily.win_rate >= 60 ? "green" : analytics.daily.win_rate >= 40 ? "yellow" : "red"}
          trend={analytics.daily.win_rate >= 65 ? "up" : analytics.daily.win_rate <= 45 ? "down" : "neutral"}
        />
        
        <MetricCard
          title="Today's P&L"
          value={`₹${analytics.daily.total_pnl.toLocaleString()}`}
          icon="💰"
          color={analytics.daily.total_pnl >= 0 ? "green" : "red"}
          trend={analytics.daily.total_pnl > 0 ? "up" : analytics.daily.total_pnl < 0 ? "down" : "neutral"}
        />
        
        <MetricCard
          title="Avg Confidence"
          value={`${analytics.daily.average_confidence}%`}
          icon="🎲"
          color={analytics.daily.average_confidence >= 75 ? "green" : "yellow"}
        />
      </div>

      {/* System Health */}
      <SystemHealthPanel health={analytics.system_health} />

      {/* Performance Details */}
      <PerformancePanel 
        daily={analytics.daily} 
        configuration={analytics.configuration}
      />

      {/* Service Status */}
      <ServiceStatusPanel 
        telegram={analytics.system_health.telegram_configured}
        zerodha={analytics.configuration.zerodha_enabled}
        uptime={analytics.daily.system_uptime_hours}
        errorRate={analytics.system_health.error_rate}
      />
    </div>
  );
}

// ================================================================
// Supporting Components
// ================================================================

const MetricCard: React.FC<{
  title: string;
  value: string | number;
  maxValue?: number;
  suffix?: string;
  icon: string;
  color: string;
  trend?: 'up' | 'down' | 'neutral';
}> = ({ title, value, maxValue, suffix, icon, color, trend }) => {
  const colorClasses = {
    blue: 'border-blue-200 bg-blue-50',
    green: 'border-green-200 bg-green-50',
    yellow: 'border-yellow-200 bg-yellow-50',
    red: 'border-red-200 bg-red-50',
  };

  const textColorClasses = {
    blue: 'text-blue-900',
    green: 'text-green-900',
    yellow: 'text-yellow-900',
    red: 'text-red-900',
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`p-6 rounded-lg border ${colorClasses[color as keyof typeof colorClasses]} relative overflow-hidden`}
    >
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-gray-600 mb-1">{title}</p>
          <div className="flex items-baseline space-x-1">
            <p className={`text-2xl font-bold ${textColorClasses[color as keyof typeof textColorClasses]}`}>
              {value}
            </p>
            {suffix && <span className="text-sm text-gray-500">{suffix}</span>}
          </div>
        </div>
        <div className="text-2xl">{icon}</div>
      </div>
      
      {trend && (
        <div className="mt-2 flex items-center">
          {trend === 'up' && (
            <svg className="w-4 h-4 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
            </svg>
          )}
          {trend === 'down' && (
            <svg className="w-4 h-4 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" />
            </svg>
          )}
        </div>
      )}
      
      {maxValue && (
        <div className="mt-3">
          <div className="bg-white/50 rounded-full h-2">
            <div 
              className="bg-current h-2 rounded-full transition-all duration-300"
              style={{ width: `${Math.min((Number(value) / maxValue) * 100, 100)}%` }}
            />
          </div>
        </div>
      )}
    </motion.div>
  );
};

const SystemStatus: React.FC<{ status: string }> = ({ status }) => {
  const statusConfig = {
    EXCELLENT: { color: 'green', text: 'Excellent', icon: '🟢' },
    GOOD: { color: 'blue', text: 'Good', icon: '🔵' },
    ACTIVE: { color: 'green', text: 'Active', icon: '🟢' },
    STARTING: { color: 'yellow', text: 'Starting', icon: '🟡' },
    ISSUES: { color: 'red', text: 'Issues', icon: '🔴' },
    ERROR: { color: 'red', text: 'Error', icon: '❌' },
  };

  const config = statusConfig[status as keyof typeof statusConfig] || statusConfig.ERROR;

  return (
    <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm bg-${config.color}-100 text-${config.color}-800`}>
      <span>{config.icon}</span>
      <span>System: {config.text}</span>
    </div>
  );
};

const SystemHealthPanel: React.FC<{ health: DashboardAnalytics['system_health'] }> = ({ health }) => (
  <div className="bg-white rounded-lg border p-6">
    <h3 className="text-lg font-semibold text-gray-900 mb-4">System Health</h3>
    
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <HealthIndicator
        label="Telegram"
        status={health.telegram_configured}
        icon="📱"
      />
      <HealthIndicator
        label="Signal Generation"
        status={health.signal_generation_active}
        icon="🤖"
      />
      <HealthIndicator
        label="Error Rate"
        status={health.error_rate < 5}
        icon="⚠️"
        detail={`${health.error_rate}%`}
      />
      <HealthIndicator
        label="Last Activity"
        status={!!health.last_activity}
        icon="⏰"
        detail={health.last_activity ? new Date(health.last_activity).toLocaleTimeString() : 'None'}
      />
    </div>
  </div>
);

const HealthIndicator: React.FC<{
  label: string;
  status: boolean;
  icon: string;
  detail?: string;
}> = ({ label, status, icon, detail }) => (
  <div className="text-center">
    <div className={`text-2xl mb-2 ${status ? 'text-green-600' : 'text-red-600'}`}>
      {icon}
    </div>
    <p className="text-sm font-medium text-gray-900">{label}</p>
    <p className={`text-xs ${status ? 'text-green-600' : 'text-red-600'}`}>
      {status ? 'Active' : 'Inactive'}
    </p>
    {detail && <p className="text-xs text-gray-500 mt-1">{detail}</p>}
  </div>
);

const PerformancePanel: React.FC<{
  daily: DashboardAnalytics['daily'];
  configuration: DashboardAnalytics['configuration'];
}> = ({ daily, configuration }) => (
  <div className="bg-white rounded-lg border p-6">
    <h3 className="text-lg font-semibold text-gray-900 mb-4">Performance Details</h3>
    
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      <div>
        <h4 className="font-medium text-gray-900 mb-3">Today's Activity</h4>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-600">Signals Generated:</span>
            <span className="font-medium">{daily.signals_generated}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Signals Sent:</span>
            <span className="font-medium">{daily.signals_sent}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Telegram Success:</span>
            <span className="font-medium">{daily.telegram_success_rate}%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Last Signal:</span>
            <span className="font-medium text-xs">
              {daily.last_signal_time 
                ? new Date(daily.last_signal_time).toLocaleTimeString()
                : 'None today'
              }
            </span>
          </div>
        </div>
      </div>
      
      <div>
        <h4 className="font-medium text-gray-900 mb-3">Configuration</h4>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-600">Max Signals/Day:</span>
            <span className="font-medium">{configuration.max_signals_per_day}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Min Confidence:</span>
            <span className="font-medium">{(configuration.min_confidence_threshold * 100).toFixed(0)}%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Signal Interval:</span>
            <span className="font-medium">{configuration.signal_interval_seconds}s</span>
          </div>
        </div>
      </div>
      
      <div>
        <h4 className="font-medium text-gray-900 mb-3">System Status</h4>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-600">Uptime:</span>
            <span className="font-medium">{daily.system_uptime_hours.toFixed(1)}h</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Status:</span>
            <span className={`font-medium ${
              daily.status === 'EXCELLENT' ? 'text-green-600' :
              daily.status === 'GOOD' ? 'text-blue-600' :
              daily.status === 'ACTIVE' ? 'text-green-600' :
              'text-yellow-600'
            }`}>
              {daily.status}
            </span>
          </div>
        </div>
      </div>
    </div>
  </div>
);

const ServiceStatusPanel: React.FC<{
  telegram: boolean;
  zerodha: boolean;
  uptime: number;
  errorRate: number;
}> = ({ telegram, zerodha, uptime, errorRate }) => (
  <div className="bg-white rounded-lg border p-6">
    <h3 className="text-lg font-semibold text-gray-900 mb-4">Service Status</h3>
    
    <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
      <div className="text-center">
        <div className={`text-3xl mb-2 ${telegram ? 'text-green-500' : 'text-red-500'}`}>
          📱
        </div>
        <p className="text-sm font-medium">Telegram Bot</p>
        <p className={`text-xs ${telegram ? 'text-green-600' : 'text-red-600'}`}>
          {telegram ? 'Connected' : 'Not Configured'}
        </p>
      </div>
      
      <div className="text-center">
        <div className={`text-3xl mb-2 ${zerodha ? 'text-green-500' : 'text-yellow-500'}`}>
          💹
        </div>
        <p className="text-sm font-medium">Zerodha API</p>
        <p className={`text-xs ${zerodha ? 'text-green-600' : 'text-yellow-600'}`}>
          {zerodha ? 'Ready' : 'Not Configured'}
        </p>
      </div>
      
      <div className="text-center">
        <div className="text-3xl mb-2 text-blue-500">⏰</div>
        <p className="text-sm font-medium">Uptime</p>
        <p className="text-xs text-blue-600">{uptime.toFixed(1)} hours</p>
      </div>
      
      <div className="text-center">
        <div className={`text-3xl mb-2 ${errorRate < 5 ? 'text-green-500' : 'text-red-500'}`}>
          📊
        </div>
        <p className="text-sm font-medium">Error Rate</p>
        <p className={`text-xs ${errorRate < 5 ? 'text-green-600' : 'text-red-600'}`}>
          {errorRate.toFixed(1)}%
        </p>
      </div>
    </div>
  </div>
);

// ================================================================
// File 2: Update frontend/src/app/page.tsx to include analytics
// ================================================================

// Add this import to your existing page.tsx:
// import AnalyticsDashboard from '@/components/analytics/AnalyticsDashboard';

// Add this section after your LiveSignalsDemo component:
/*
        {/* Analytics Dashboard *\/}
        <div className="mt-16">
          <AnalyticsDashboard />
        </div>
*/