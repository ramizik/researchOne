import { useState } from 'react';
import { Brain, Sparkles, ChevronDown, ChevronUp, Copy, CheckCircle } from 'lucide-react';
import type { MultimodalInsights } from '../types';
import { formatConfidence } from '../utils/helpers';

interface InsightsCardProps {
  multimodal: MultimodalInsights;
  geminiInsights: string;
}

export const InsightsCard = ({ multimodal, geminiInsights }: InsightsCardProps) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(geminiInsights);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const coherenceColor = multimodal.emotional_coherence === 'high' ? 'text-success' :
                         multimodal.emotional_coherence === 'moderate' ? 'text-warning' : 'text-danger';

  const alignmentColor = multimodal.voice_emotion_alignment === 'high' ? 'text-success' :
                        multimodal.voice_emotion_alignment === 'moderate' ? 'text-warning' : 'text-danger';

  const coherencePercentage = multimodal.emotional_coherence === 'high' ? 85 :
                              multimodal.emotional_coherence === 'moderate' ? 60 : 35;

  const alignmentPercentage = multimodal.voice_emotion_alignment === 'high' ? 85 :
                             multimodal.voice_emotion_alignment === 'moderate' ? 60 : 35;

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-2">
        <Brain className="w-6 h-6" />
        AI-Powered Insights
      </h2>

      <div className="mb-8 p-6 bg-gradient-to-r from-purple-100 via-blue-100 to-green-100 rounded-lg text-center">
        <Sparkles className="w-12 h-12 text-purple-600 mx-auto mb-3" />
        <h3 className="text-2xl font-bold text-gray-900 mb-2">
          {multimodal.overall_emotional_state.split('_').map(word =>
            word.charAt(0).toUpperCase() + word.slice(1)
          ).join(' ')}
        </h3>
        <p className="text-gray-600">Overall Emotional State</p>
        <div className="mt-4">
          <p className="text-sm text-gray-600 mb-2">Analysis Confidence</p>
          <div className="flex items-center gap-3 justify-center">
            <div className="flex-1 max-w-xs bg-white rounded-full h-3">
              <div
                className="h-full bg-gradient-to-r from-purple-500 to-blue-500 rounded-full"
                style={{ width: `${multimodal.confidence_score * 100}%` }}
              />
            </div>
            <span className="font-bold text-gray-900">{formatConfidence(multimodal.confidence_score)}</span>
          </div>
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-6 mb-8">
        <div className="p-6 bg-blue-50 rounded-lg">
          <h4 className="font-semibold text-gray-900 mb-3">Emotional Coherence</h4>
          <p className={`text-3xl font-bold ${coherenceColor} mb-2 capitalize`}>
            {multimodal.emotional_coherence}
          </p>
          <div className="w-full bg-white rounded-full h-3">
            <div
              className="h-full bg-blue-500 rounded-full transition-all"
              style={{ width: `${coherencePercentage}%` }}
            />
          </div>
          <p className="text-sm text-gray-600 mt-2">
            How well facial and vocal emotions align
          </p>
        </div>

        <div className="p-6 bg-purple-50 rounded-lg">
          <h4 className="font-semibold text-gray-900 mb-3">Voice-Emotion Alignment</h4>
          <p className={`text-3xl font-bold ${alignmentColor} mb-2 capitalize`}>
            {multimodal.voice_emotion_alignment}
          </p>
          <div className="w-full bg-white rounded-full h-3">
            <div
              className="h-full bg-purple-500 rounded-full transition-all"
              style={{ width: `${alignmentPercentage}%` }}
            />
          </div>
          <p className="text-sm text-gray-600 mt-2">
            Consistency between voice tone and emotions
          </p>
        </div>
      </div>

      <div className="mb-6">
        <h4 className="font-semibold text-gray-900 mb-3">Key Observations</h4>
        <ul className="space-y-2">
          {multimodal.key_observations.map((observation, index) => (
            <li key={index} className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg">
              <CheckCircle className="w-5 h-5 text-success flex-shrink-0 mt-0.5" />
              <span className="text-gray-700">{observation}</span>
            </li>
          ))}
        </ul>
      </div>

      <div className="border-t pt-6">
        <div className="flex items-center justify-between mb-4">
          <h4 className="font-semibold text-gray-900 flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-purple-600" />
            Gemini AI Comprehensive Analysis
          </h4>
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="flex items-center gap-1 text-primary hover:text-blue-600 transition"
          >
            {isExpanded ? (
              <>
                <span className="text-sm">Collapse</span>
                <ChevronUp className="w-4 h-4" />
              </>
            ) : (
              <>
                <span className="text-sm">Expand</span>
                <ChevronDown className="w-4 h-4" />
              </>
            )}
          </button>
        </div>

        <div className={`bg-gradient-to-br from-purple-50 to-blue-50 rounded-lg p-6 ${isExpanded ? '' : 'max-h-48 overflow-hidden relative'}`}>
          <p className="text-gray-700 whitespace-pre-line leading-relaxed">
            {geminiInsights}
          </p>
          {!isExpanded && (
            <div className="absolute bottom-0 left-0 right-0 h-24 bg-gradient-to-t from-blue-50 to-transparent" />
          )}
        </div>

        <button
          onClick={handleCopy}
          className="mt-3 flex items-center gap-2 text-sm text-gray-600 hover:text-primary transition"
        >
          {copied ? (
            <>
              <CheckCircle className="w-4 h-4 text-success" />
              <span className="text-success">Copied!</span>
            </>
          ) : (
            <>
              <Copy className="w-4 h-4" />
              <span>Copy AI Analysis</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
};
