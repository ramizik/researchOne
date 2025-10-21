import { Download, RotateCcw, Clock, Hash } from 'lucide-react';
import type { AnalysisResults } from '../types';
import { EmotionCard } from './EmotionCard';
import { VoiceCard } from './VoiceCard';
import { TranscriptionCard } from './TranscriptionCard';
import { InsightsCard } from './InsightsCard';
import { downloadJSON, formatTimestamp } from '../utils/helpers';

interface ResultsScreenProps {
  results: AnalysisResults;
  onNewRecording: () => void;
}

export const ResultsScreen = ({ results, onNewRecording }: ResultsScreenProps) => {
  const handleExport = () => {
    downloadJSON(results, `analysis-${results.metadata.session_id}.json`);
  };

  return (
    <div className="max-w-7xl mx-auto p-6">
      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 mb-2">Analysis Results</h1>
            <div className="flex flex-wrap items-center gap-4 text-sm text-gray-600">
              <div className="flex items-center gap-2">
                <Clock className="w-4 h-4" />
                <span>{formatTimestamp(results.metadata.timestamp)}</span>
              </div>
              <div className="flex items-center gap-2">
                <Hash className="w-4 h-4" />
                <span className="font-mono text-xs">{results.metadata.session_id}</span>
              </div>
            </div>
          </div>

          <div className="flex gap-3">
            <button
              onClick={handleExport}
              className="flex items-center gap-2 px-4 py-2 bg-primary text-white rounded-lg font-semibold hover:bg-blue-600 transition"
            >
              <Download className="w-4 h-4" />
              Export JSON
            </button>
            <button
              onClick={onNewRecording}
              className="flex items-center gap-2 px-4 py-2 bg-success text-white rounded-lg font-semibold hover:bg-green-600 transition"
            >
              <RotateCcw className="w-4 h-4" />
              New Recording
            </button>
          </div>
        </div>
      </div>

      <div className="grid gap-6">
        <EmotionCard data={results.emotion_analysis} />
        <VoiceCard data={results.voice_analysis} />
        <TranscriptionCard data={results.transcription_analysis} />
        <InsightsCard
          multimodal={results.multimodal_insights}
          geminiInsights={results.gemini_insights}
        />
      </div>

      <div className="mt-6 text-center">
        <button
          onClick={onNewRecording}
          className="inline-flex items-center gap-2 px-6 py-3 bg-primary text-white rounded-lg font-semibold hover:bg-blue-600 transition shadow-lg"
        >
          <RotateCcw className="w-5 h-5" />
          Start New Analysis
        </button>
      </div>
    </div>
  );
};
