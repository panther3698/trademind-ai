'use client';
import React from 'react';
import ConfigurationManager from '@/components/Admin/ConfigurationManager';
import AdminAuth from '@/components/AdminAuth';

export default function ConfigurationPage() {
  return (
    <AdminAuth>
      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <header className="bg-white shadow-sm border-b">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center h-16">
              <div className="flex items-center space-x-3">
                <div className="bg-blue-600 rounded-lg p-2">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                </div>
                <div>
                  <h1 className="text-xl font-bold text-gray-900">TradeMind AI</h1>
                  <p className="text-sm text-gray-500">System Configuration</p>
                </div>
              </div>
              <div className="flex items-center space-x-4">
                <a href="/dashboard" className="text-blue-600 hover:text-blue-700 transition-colors text-sm">
                  ← Back to Dashboard
                </a>
                <a href="/" className="text-gray-600 hover:text-gray-700 transition-colors text-sm">
                  ← Back to Home
                </a>
              </div>
            </div>
          </div>
        </header>

        {/* Breadcrumb */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <nav className="flex items-center space-x-2 text-sm text-gray-500">
            <a href="/" className="hover:text-blue-600 transition-colors">Home</a>
            <span>›</span>
            <a href="/dashboard" className="hover:text-blue-600 transition-colors">Dashboard</a>
            <span>›</span>
            <span className="text-gray-900 font-medium">Configuration</span>
          </nav>
        </div>

        {/* Configuration Manager */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <ConfigurationManager />
        </div>
      </div>
    </AdminAuth>
  );
} 