import { useState } from 'react';
import { Brain } from 'lucide-react';
import { RecordingScreen } from './components/RecordingScreen';
import { ProcessingScreen } from './components/ProcessingScreen';
import { ResultsScreen } from './components/ResultsScreen';
import { useAnalysisPolling } from './hooks/useAnalysisPolling';
import { apiService } from './services/api';
import type { AppScreen } from './types';

function App() {
  const [currentScreen, setCurrentScreen] = useState<AppScreen>('recording');
  const [error, setError] = useState<string | null>(null);
  const { status, results, error: pollingError, startPolling } = useAnalysisPolling();

  const handleRecordingComplete = async (blob: Blob) => {
    try {
      setError(null);
      setCurrentScreen('processing');

      const sessionResponse = await apiService.startSession();
      const sessionId = sessionResponse.session_id;

      await apiService.uploadVideo(sessionId, blob);
      await apiService.uploadAudio(sessionId, blob);

      startPolling(sessionId);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to upload recording';
      setError(errorMessage);
      setCurrentScreen('recording');
    }
  };

  const handleNewRecording = () => {
    setCurrentScreen('recording');
    setError(null);
  };

  if (results && currentScreen !== 'results') {
    setCurrentScreen('results');
  }

  return (
    <div className="min-h-screen bg-gray-100">
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="bg-primary p-2 rounded-lg">
              <Brain className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                Multimodal Emotion & Voice Analysis
              </h1>
              <p className="text-sm text-gray-600">
                AI-powered facial emotion detection and voice analysis
              </p>
            </div>
          </div>
        </div>
      </header>

      <main className="py-8">
        {error && (
          <div className="max-w-4xl mx-auto px-6 mb-6">
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-800">
              <h3 className="font-semibold mb-1">Error</h3>
              <p>{error}</p>
            </div>
          </div>
        )}

        {pollingError && (
          <div className="max-w-4xl mx-auto px-6 mb-6">
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-800">
              <h3 className="font-semibold mb-1">Analysis Error</h3>
              <p>{pollingError}</p>
              <button
                onClick={handleNewRecording}
                className="mt-3 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition"
              >
                Try Again
              </button>
            </div>
          </div>
        )}

        {currentScreen === 'recording' && (
          <RecordingScreen onRecordingComplete={handleRecordingComplete} />
        )}

        {currentScreen === 'processing' && (
          <ProcessingScreen status={status} />
        )}

        {currentScreen === 'results' && results && (
          <ResultsScreen results={results} onNewRecording={handleNewRecording} />
        )}
      </main>

      <footer className="bg-white border-t mt-12">
        <div className="max-w-7xl mx-auto px-6 py-6 text-center text-gray-600 text-sm">
          <p>Multimodal Analysis System - For Research and Clinical Applications</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
