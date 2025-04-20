import { useState, useEffect } from 'react';

export default function Home() {
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [videoUrl, setVideoUrl] = useState(null);
  const [codeOutput, setCodeOutput] = useState(null);
  const [error, setError] = useState(null);
  const [modelType, setModelType] = useState('base-model'); 
  const [videoError, setVideoError] = useState(null);
  const [showingVideo, setShowingVideo] = useState(false);
  const [showingCode, setShowingCode] = useState(false);
  const [pendingVideo, setPendingVideo] = useState(null);
  const [pendingCode, setPendingCode] = useState(null);
  const [pendingVideoError, setPendingVideoError] = useState(null);

  useEffect(() => {
    if (pendingCode) {
      setShowingCode(true);
      const timer = setTimeout(() => {
        setCodeOutput(pendingCode);
        setShowingCode(false);
        setPendingCode(null);
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [pendingCode]);

  useEffect(() => {
    if (pendingVideo) {
      setShowingVideo(true);
      const timer = setTimeout(() => {
        setVideoUrl(pendingVideo);
        setShowingVideo(false);
        setPendingVideo(null);
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [pendingVideo]);

  useEffect(() => {
    if (pendingVideoError) {
      setShowingVideo(true);
      const timer = setTimeout(() => {
        setVideoError(pendingVideoError);
        setShowingVideo(false);
        setPendingVideoError(null);
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [pendingVideoError]);

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);
    setVideoUrl(null);
    setCodeOutput(null);
    setVideoError(null);
    setShowingVideo(false);
    setShowingCode(false);
    setPendingVideo(null);
    setPendingCode(null);
    setPendingVideoError(null);
    
    try {
      const res = await fetch('http://localhost:8000/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          input_text: input,
          model_type: modelType
        })
      });
      if (!res.ok) throw new Error('Generation failed');
      const data = await res.json();
      
      if (data.code) {
        setPendingCode(data.code);
      } else {
        try {
          const codeRes = await fetch('http://localhost:8000/code');
          if (codeRes.ok) {
            const codeData = await codeRes.json();
            setPendingCode(codeData.code);
          }
        } catch (codeErr) {
          console.error('Failed to fetch code:', codeErr);
        }
      }
      
      if (data.video_error) {
        setPendingVideoError("Error generating Video");
      } else {
        const videoRes = await fetch('http://localhost:8000/video');
        if (!videoRes.ok) throw new Error('No video found');
        const blob = await videoRes.blob();
        setPendingVideo(URL.createObjectURL(blob));
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100 flex flex-col">
      <header className="py-6 text-center text-3xl font-bold tracking-tight">AI Personal Tutor</header>
      <main className="flex flex-1 flex-col items-center px-4 max-w-6xl mx-auto w-full">
        <div className="flex flex-col md:flex-row w-full gap-8 mb-8">
          <div className="flex flex-col gap-4 w-full md:w-1/3">
            <textarea
              className="rounded border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 p-3 focus:outline-none focus:ring h-40"
              placeholder="Enter prompt for video generation..."
              value={input}
              onChange={e => setInput(e.target.value)}
            />
            
            <div className="mb-2">
              <label htmlFor="model-select" className="block text-sm font-medium mb-1">
                Select Model:
              </label>
              <select
                id="model-select"
                className="w-full rounded border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 p-2 focus:outline-none focus:ring"
                value={modelType}
                onChange={e => setModelType(e.target.value)}
              >
                <option value="base-model">Base Model</option>
                <option value="SFT">SFT Model</option>
                <option value="GRPO">GRPO Model</option>
              </select>
            </div>
            
            <button
              className="rounded bg-blue-600 text-white py-2 font-semibold hover:bg-blue-700 disabled:opacity-60"
              onClick={handleGenerate}
              disabled={loading || !input}
            >
              {loading ? 'Generating...' : 'Generate Video'}
            </button>
            {error && <div className="text-red-500">{error}</div>}
          </div>
          
          <div className="flex-1 flex items-center justify-center min-h-[300px] w-full md:w-2/3">
            {videoUrl ? (
              <video src={videoUrl} controls className="rounded shadow-lg max-w-full max-h-[400px]" />
            ) : (
              <div className="text-gray-400 border-2 border-dashed border-gray-300 dark:border-gray-700 rounded-lg p-12 flex items-center justify-center">
                {loading ? 'Generating video...' : 
                 showingVideo ? (
                   <div className="flex flex-col items-center">
                     <div className="animate-pulse text-blue-500 mb-2">Loading video...</div>
                     <div className="w-8 h-8 border-t-2 border-b-2 border-blue-500 rounded-full animate-spin"></div>
                   </div>
                 ) :
                 videoError ? <span className="text-red-500">{videoError}</span> : 
                 'No video yet'}
              </div>
            )}
          </div>
        </div>
        
        <div className="w-full mt-4">
          <h2 className="text-xl font-semibold mb-2">Generated Code</h2>
          <div className="bg-gray-800 text-gray-100 p-4 rounded-lg overflow-auto max-h-[400px] font-mono text-sm">
            {codeOutput ? (
              <pre>{codeOutput}</pre>
            ) : (
              <div className="text-gray-400">
                {loading ? 'Generating code...' : 
                 showingCode ? (
                   <div className="flex flex-col items-center">
                     <div className="animate-pulse text-blue-500 mb-2">Loading code...</div>
                     <div className="w-8 h-8 border-t-2 border-b-2 border-blue-500 rounded-full animate-spin"></div>
                   </div>
                 ) :
                 'No code generated yet'}
              </div>
            )}
          </div>
        </div>
      </main>
      <nav className="absolute top-4 right-8">
        <a href="/status" className="underline text-blue-500">Status</a>
      </nav>
    </div>
  );
}
