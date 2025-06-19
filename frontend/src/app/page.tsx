import React from 'react';

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50">
      {/* Navigation */}
      <nav className="bg-white/80 backdrop-blur-md border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-2">
              <div className="bg-blue-600 rounded-lg p-2">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                </svg>
              </div>
              <span className="text-xl font-bold text-gray-900">TradeMind AI</span>
            </div>
            <div className="flex items-center space-x-6">
              <a href="/dashboard" className="text-gray-600 hover:text-blue-600 transition-colors">
                Dashboard
              </a>
              <a href="http://localhost:8000/docs" target="_blank" rel="noopener noreferrer" className="text-gray-600 hover:text-blue-600 transition-colors">
                API
              </a>
              <a href="https://github.com/panther3698/trademind-ai" target="_blank" rel="noopener noreferrer" className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors">
                View Code
              </a>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-20 pb-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <div className="mb-8">
            <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-yellow-100 text-yellow-800 mb-6">
              🔬 Research & Development Platform
            </span>
            <h1 className="text-5xl md:text-6xl font-bold text-gray-900 mb-6">
              AI Trading Signal
              <span className="block text-blue-600">Research Platform</span>
            </h1>
            <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
              A comprehensive AI system for analyzing Indian stock markets. 
              This platform generates trading signals for research and backtesting purposes only.
              <strong className="block mt-2 text-gray-800">Not for public trading advice or investment recommendations.</strong>
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <a href="/dashboard" 
                className="bg-blue-600 text-white px-8 py-4 rounded-lg text-lg font-semibold hover:bg-blue-700 transition-colors text-center">
                View Research Dashboard
              </a>
              <a href="https://github.com/panther3698/trademind-ai" target="_blank" rel="noopener noreferrer"
                className="border-2 border-blue-600 text-blue-600 px-8 py-4 rounded-lg text-lg font-semibold hover:bg-blue-50 transition-colors text-center">
                Explore Code
              </a>
            </div>
          </div>
        </div>
      </section>

      {/* Research Stats */}
      <section className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Research Performance Metrics</h2>
            <p className="text-gray-600">Algorithm testing and backtesting results</p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div className="text-center p-6 bg-green-50 rounded-xl">
              <div className="text-3xl font-bold text-green-600 mb-2">65.3%</div>
              <div className="text-gray-600">Backtesting Accuracy</div>
            </div>
            <div className="text-center p-6 bg-blue-50 rounded-xl">
              <div className="text-3xl font-bold text-blue-600 mb-2">2-3</div>
              <div className="text-gray-600">Signals per Day</div>
            </div>
            <div className="text-center p-6 bg-purple-50 rounded-xl">
              <div className="text-3xl font-bold text-purple-600 mb-2">100+</div>
              <div className="text-gray-600">Stocks Analyzed</div>
            </div>
            <div className="text-center p-6 bg-orange-50 rounded-xl">
              <div className="text-3xl font-bold text-orange-600 mb-2">45s</div>
              <div className="text-gray-600">Signal Generation</div>
            </div>
          </div>
        </div>
      </section>

      {/* Technology Stack */}
      <section className="py-16 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Technology & Features</h2>
            <p className="text-gray-600">Modern architecture for financial data analysis</p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="bg-white p-8 rounded-xl shadow-sm">
              <div className="bg-blue-100 rounded-lg p-3 w-12 h-12 flex items-center justify-center mb-6">
                <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-4">AI Signal Generation</h3>
              <p className="text-gray-600">Machine learning models for market pattern recognition, technical analysis, and sentiment scoring from news data.</p>
            </div>

            <div className="bg-white p-8 rounded-xl shadow-sm">
              <div className="bg-green-100 rounded-lg p-3 w-12 h-12 flex items-center justify-center mb-6">
                <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-4">Real-Time Processing</h3>
              <p className="text-gray-600">FastAPI backend with WebSocket connections for live market data processing and signal generation.</p>
            </div>

            <div className="bg-white p-8 rounded-xl shadow-sm">
              <div className="bg-purple-100 rounded-lg p-3 w-12 h-12 flex items-center justify-center mb-6">
                <svg className="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2-2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-4">Analytics Dashboard</h3>
              <p className="text-gray-600">Comprehensive performance tracking, signal accuracy monitoring, and backtesting result analysis.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Disclaimer Section */}
      <section className="py-16 bg-yellow-50 border-t border-yellow-200">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <div className="bg-yellow-100 rounded-lg p-6">
            <h3 className="text-xl font-semibold text-yellow-900 mb-4">Important Disclaimer</h3>
            <div className="text-yellow-800 space-y-2">
              <p><strong>This is a research and development platform.</strong></p>
              <p>• No trading advice or investment recommendations are provided</p>
              <p>• All signals are for backtesting and algorithm development only</p>
              <p>• Not registered as a SEBI Research Analyst</p>
              <p>• Past performance does not guarantee future results</p>
              <p>• Always consult qualified financial advisors before trading</p>
            </div>
          </div>
        </div>
      </section>

      {/* Future Plans */}
      <section className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-bold text-gray-900 mb-8">Future Development Roadmap</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="p-6 border border-gray-200 rounded-lg">
              <div className="text-2xl mb-3">📋</div>
              <h3 className="font-semibold text-gray-900 mb-2">SEBI Registration</h3>
              <p className="text-sm text-gray-600">Research Analyst certification for compliance</p>
            </div>
            <div className="p-6 border border-gray-200 rounded-lg">
              <div className="text-2xl mb-3">💰</div>
              <h3 className="font-semibold text-gray-900 mb-2">Subscription Model</h3>
              <p className="text-sm text-gray-600">Tiered pricing for different user types</p>
            </div>
            <div className="p-6 border border-gray-200 rounded-lg">
              <div className="text-2xl mb-3">📱</div>
              <h3 className="font-semibold text-gray-900 mb-2">Mobile App</h3>
              <p className="text-sm text-gray-600">iOS and Android applications</p>
            </div>
            <div className="p-6 border border-gray-200 rounded-lg">
              <div className="text-2xl mb-3">🔗</div>
              <h3 className="font-semibold text-gray-900 mb-2">API Platform</h3>
              <p className="text-sm text-gray-600">Developer API for integration</p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <div className="flex items-center justify-center space-x-2 mb-4">
              <div className="bg-blue-600 rounded-lg p-2">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                </svg>
              </div>
              <span className="text-xl font-bold">TradeMind AI</span>
            </div>
            <p className="text-gray-400 mb-8">
              Research platform for AI-powered market analysis. 
              Built for learning and algorithm development.
            </p>
            <div className="flex justify-center space-x-6">
              <a href="https://github.com/panther3698/trademind-ai" target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-white transition-colors">
                <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                </svg>
              </a>
            </div>
            <div className="mt-8 pt-8 border-t border-gray-800 text-center text-gray-400">
              <p>&copy; 2025 TradeMind AI Research Platform. For educational and research purposes only.</p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}