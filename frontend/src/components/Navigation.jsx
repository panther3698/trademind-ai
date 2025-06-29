'use client';
import React, { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

export default function Navigation() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const pathname = usePathname();

  const navItems = [
    { href: '/', label: '🏠 Home', icon: '🏠' },
    { href: '/dashboard', label: '📊 Dashboard', icon: '📊' },
    { href: '/configuration', label: '🔧 Configuration', icon: '🔧' },
  ];

  const isActive = (href) => {
    if (href === '/') {
      return pathname === '/';
    }
    return pathname.startsWith(href);
  };

  return (
    <nav className="bg-white/80 backdrop-blur-md border-b border-gray-200 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo and Brand */}
          <div className="flex items-center space-x-2">
            <div className="bg-blue-600 rounded-lg p-2">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
              </svg>
            </div>
            <span className="text-xl font-bold text-gray-900">TradeMind AI</span>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-6">
            {navItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  isActive(item.href)
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-600 hover:text-blue-600 hover:bg-gray-50'
                }`}
              >
                <span>{item.icon}</span>
                <span>{item.label.replace(/^[^\s]*\s/, '')}</span>
              </Link>
            ))}
            
            {/* External Links */}
            <a 
              href="http://localhost:8000/docs" 
              target="_blank" 
              rel="noopener noreferrer" 
              className="text-gray-600 hover:text-blue-600 transition-colors"
            >
              API
            </a>
            <a 
              href="https://github.com/panther3698/trademind-ai" 
              target="_blank" 
              rel="noopener noreferrer" 
              className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
            >
              View Code
            </a>
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden">
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="inline-flex items-center justify-center p-2 rounded-md text-gray-600 hover:text-blue-600 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-blue-500"
            >
              <span className="sr-only">Open main menu</span>
              {mobileMenuOpen ? (
                <svg className="block h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              ) : (
                <svg className="block h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Navigation Menu */}
      {mobileMenuOpen && (
        <div className="md:hidden">
          <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3 bg-white border-t border-gray-200">
            {navItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center space-x-3 px-3 py-2 rounded-md text-base font-medium transition-colors ${
                  isActive(item.href)
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-600 hover:text-blue-600 hover:bg-gray-50'
                }`}
                onClick={() => setMobileMenuOpen(false)}
              >
                <span className="text-lg">{item.icon}</span>
                <span>{item.label}</span>
              </Link>
            ))}
            
            {/* Mobile External Links */}
            <div className="pt-4 border-t border-gray-200">
              <a 
                href="http://localhost:8000/docs" 
                target="_blank" 
                rel="noopener noreferrer" 
                className="flex items-center space-x-3 px-3 py-2 rounded-md text-base font-medium text-gray-600 hover:text-blue-600 hover:bg-gray-50 transition-colors"
                onClick={() => setMobileMenuOpen(false)}
              >
                <span>📚</span>
                <span>API Documentation</span>
              </a>
              <a 
                href="https://github.com/panther3698/trademind-ai" 
                target="_blank" 
                rel="noopener noreferrer" 
                className="flex items-center space-x-3 px-3 py-2 rounded-md text-base font-medium text-gray-600 hover:text-blue-600 hover:bg-gray-50 transition-colors"
                onClick={() => setMobileMenuOpen(false)}
              >
                <span>🐙</span>
                <span>View Code</span>
              </a>
            </div>
          </div>
        </div>
      )}
    </nav>
  );
} 