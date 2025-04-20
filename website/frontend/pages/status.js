import { useState } from 'react';

export default function Status() {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const checkStatus = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('http://localhost:8000/status');
      if (!res.ok) throw new Error('Failed to fetch status');
      const data = await res.json();
      setStatus(data.status);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100 flex flex-col items-center justify-center">
      <nav className="absolute top-4 left-8">
        <a href="/" className="underline text-blue-500">Home</a>
      </nav>
      <h1 className="text-2xl font-bold mb-4">Backend Status</h1>
      <button
        className="rounded bg-blue-600 text-white py-2 px-4 font-semibold hover:bg-blue-700 disabled:opacity-60 mb-4"
        onClick={checkStatus}
        disabled={loading}
      >
        {loading ? 'Checking...' : 'Check Status'}
      </button>
      {status && <div className="text-green-500">Status: {status}</div>}
      {error && <div className="text-red-500">{error}</div>}
    </div>
  );
}
