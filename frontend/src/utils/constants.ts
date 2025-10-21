import type { EmotionType } from '../types';

export const RECORDING_DURATION = 15000;

export const EMOTION_COLORS: Record<EmotionType, string> = {
  happiness: '#10b981',
  sadness: '#3b82f6',
  anger: '#ef4444',
  fear: '#8b5cf6',
  surprise: '#f59e0b',
  disgust: '#84cc16',
  neutral: '#6b7280',
};

export const EMOTION_EMOJIS: Record<EmotionType, string> = {
  happiness: '😊',
  sadness: '😢',
  anger: '😠',
  fear: '😨',
  surprise: '😲',
  disgust: '🤢',
  neutral: '😐',
};

export const VOICE_TYPE_COLORS = {
  bass: '#3b82f6',
  baritone: '#2563eb',
  tenor: '#8b5cf6',
  alto: '#ec4899',
  'mezzo-soprano': '#f472b6',
  soprano: '#fbbf24',
};

export const QUALITY_COLORS = {
  excellent: '#10b981',
  good: '#3b82f6',
  fair: '#f59e0b',
  poor: '#ef4444',
};

export const API_POLL_INTERVAL = 1000;
