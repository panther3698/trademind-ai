'use client';
import React, { useState, useEffect } from 'react';

export default function AdminAuth({ children }) {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  // Check if already authenticated (stored in sessionStorage)
  useEffect(() => {
    const authStatus = sessionStorage.getItem('trademind_admin_auth');
    if (authStatus === 'true') {
      setIsAuthenticated(true);
    }
  }, []);

  const handleAuth = () => {
    // Simple password check - replace with your auth system
    const validPasswords = ['admin123', 'trademind2024', 'config2024'];
    
    if (validPasswords.includes(password)) {
      setIsAuthenticated(true);
      sessionStorage.setItem('trademind_admin_auth', 'true');
      setError('');
    } else {
      setError('Invalid password. Please try again.');
      setPassword('');
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleAuth();
    }
  };

  if (isAuthenticated) {
    return children;
  }

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        <div className="text-center">
          <div className="mx-auto h-12 w-12 bg-blue-600 rounded-lg flex items-center justify-center">
            <svg className="h-8 w-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
            </svg>
          </div>
          <h2 className="mt-6 text-3xl font-extrabold text-gray-900">
            🔐 Admin Access Required
          </h2>
          <p className="mt-2 text-sm text-gray-600">
            Enter the admin password to access configuration settings
          </p>
        </div>
        
        <div className="mt-8 space-y-6">
          <div className="space-y-4">
            <div>
              <label htmlFor="password" className="sr-only">
                Admin Password
              </label>
              <input
                id="password"
                name="password"
                type="password"
                required
                className="appearance-none rounded-lg relative block w-full px-3 py-3 border border-gray-300 placeholder-gray-500 text-gray-900 focus:outline-none focus:ring-blue-500 focus:border-blue-500 focus:z-10 sm:text-sm"
                placeholder="Enter admin password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                onKeyPress={handleKeyPress}
              />
            </div>
            
            {error && (
              <div className="text-red-600 text-sm text-center">
                {error}
              </div>
            )}
          </div>

          <div>
            <button
              onClick={handleAuth}
              className="group relative w-full flex justify-center py-3 px-4 border border-transparent text-sm font-medium rounded-lg text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors"
            >
              🔓 Unlock Configuration
            </button>
          </div>

          <div className="text-center">
            <a 
              href="/dashboard" 
              className="text-sm text-blue-600 hover:text-blue-500 transition-colors"
            >
              ← Back to Dashboard
            </a>
          </div>
        </div>

        <div className="mt-8 text-center">
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
            <h3 className="text-sm font-medium text-yellow-800 mb-2">
              💡 Default Passwords
            </h3>
            <p className="text-xs text-yellow-700">
              Try: <code className="bg-yellow-100 px-1 rounded">admin123</code>, 
              <code className="bg-yellow-100 px-1 rounded ml-1">trademind2024</code>, or 
              <code className="bg-yellow-100 px-1 rounded ml-1">config2024</code>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
} 