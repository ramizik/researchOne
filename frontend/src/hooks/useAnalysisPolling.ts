import { useState, useEffect, useCallback, useRef } from 'react';
import { apiService } from '../services/api';
import type { StatusResponse, AnalysisResults } from '../types';
import { API_POLL_INTERVAL } from '../utils/constants';

interface UseAnalysisPollingReturn {
  status: StatusResponse | null;
  results: AnalysisResults | null;
  error: string | null;
  isPolling: boolean;
  startPolling: (sessionId: string) => void;
  stopPolling: () => void;
}

export const useAnalysisPolling = (): UseAnalysisPollingReturn => {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [results, setResults] = useState<AnalysisResults | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isPolling, setIsPolling] = useState(false);

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const sessionIdRef = useRef<string | null>(null);

  const stopPolling = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setIsPolling(false);
  }, []);

  const fetchResults = useCallback(async (sessionId: string) => {
    try {
      const analysisResults = await apiService.getResults(sessionId);
      setResults(analysisResults);
      stopPolling();
    } catch (err) {
      console.error('Error fetching results:', err);
      setError('Failed to fetch analysis results');
      stopPolling();
    }
  }, [stopPolling]);

  const pollStatus = useCallback(async () => {
    if (!sessionIdRef.current) return;

    try {
      const statusResponse = await apiService.getStatus(sessionIdRef.current);
      setStatus(statusResponse);

      if (statusResponse.status === 'completed') {
        await fetchResults(sessionIdRef.current);
      } else if (statusResponse.status === 'failed') {
        setError(statusResponse.error || 'Analysis failed');
        stopPolling();
      }
    } catch (err) {
      console.error('Error polling status:', err);
      setError('Failed to check analysis status');
      stopPolling();
    }
  }, [fetchResults, stopPolling]);

  const startPolling = useCallback((sessionId: string) => {
    sessionIdRef.current = sessionId;
    setIsPolling(true);
    setError(null);
    setResults(null);
    setStatus(null);

    pollStatus();

    intervalRef.current = setInterval(pollStatus, API_POLL_INTERVAL);
  }, [pollStatus]);

  useEffect(() => {
    return () => {
      stopPolling();
    };
  }, [stopPolling]);

  return {
    status,
    results,
    error,
    isPolling,
    startPolling,
    stopPolling,
  };
};
