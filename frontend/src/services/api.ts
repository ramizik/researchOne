import axios from 'axios';
import type { SessionResponse, StatusResponse, AnalysisResults, HealthCheckResponse } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
});

export const apiService = {
  async healthCheck(): Promise<HealthCheckResponse> {
    const response = await api.get<HealthCheckResponse>('/api/health');
    return response.data;
  },

  async startSession(): Promise<SessionResponse> {
    const response = await api.post<SessionResponse>('/api/analysis/start');
    return response.data;
  },

  async uploadVideo(sessionId: string, videoBlob: Blob): Promise<SessionResponse> {
    const formData = new FormData();
    formData.append('file', videoBlob, 'video.webm');

    const response = await api.post<SessionResponse>(
      `/api/analysis/upload-video?session_id=${sessionId}`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
    return response.data;
  },

  async uploadAudio(sessionId: string, audioBlob: Blob): Promise<SessionResponse> {
    const formData = new FormData();
    formData.append('file', audioBlob, 'audio.webm');

    const response = await api.post<SessionResponse>(
      `/api/analysis/upload-audio?session_id=${sessionId}`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
    return response.data;
  },

  async getStatus(sessionId: string): Promise<StatusResponse> {
    const response = await api.get<StatusResponse>(`/api/analysis/status/${sessionId}`);
    return response.data;
  },

  async getResults(sessionId: string): Promise<AnalysisResults> {
    const response = await api.get<AnalysisResults>(`/api/analysis/results/${sessionId}`);
    return response.data;
  },

  async deleteSession(sessionId: string): Promise<void> {
    await api.delete(`/api/analysis/session/${sessionId}`);
  },
};
