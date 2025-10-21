import { MessageSquare, Award, Hash, Globe } from 'lucide-react';
import type { TranscriptionAnalysis } from '../types';
import { formatConfidence } from '../utils/helpers';

interface TranscriptionCardProps {
  data: TranscriptionAnalysis;
}

export const TranscriptionCard = ({ data }: TranscriptionCardProps) => {
  const { transcription, confidence, success, word_count, language_code } = data;

  const confidenceColor = confidence >= 0.8 ? 'text-success' : confidence >= 0.6 ? 'text-warning' : 'text-danger';

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-2">
        <MessageSquare className="w-6 h-6" />
        Speech Transcription
      </h2>

      {success ? (
        <>
          <div className="mb-6 p-6 bg-gray-50 rounded-lg border-l-4 border-primary">
            <p className="text-gray-800 text-lg leading-relaxed">
              {transcription || 'No speech detected'}
            </p>
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div className="p-4 bg-gradient-to-br from-blue-50 to-purple-50 rounded-lg text-center">
              <Award className="w-6 h-6 text-primary mx-auto mb-2" />
              <p className="text-sm text-gray-600 mb-1">Confidence</p>
              <p className={`text-2xl font-bold ${confidenceColor}`}>
                {formatConfidence(confidence)}
              </p>
              <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                <div
                  className="h-full bg-primary rounded-full"
                  style={{ width: `${confidence * 100}%` }}
                />
              </div>
            </div>

            <div className="p-4 bg-gradient-to-br from-green-50 to-blue-50 rounded-lg text-center">
              <Hash className="w-6 h-6 text-success mx-auto mb-2" />
              <p className="text-sm text-gray-600 mb-1">Word Count</p>
              <p className="text-2xl font-bold text-gray-900">{word_count}</p>
            </div>

            <div className="p-4 bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg text-center">
              <Globe className="w-6 h-6 text-purple-500 mx-auto mb-2" />
              <p className="text-sm text-gray-600 mb-1">Language</p>
              <p className="text-2xl font-bold text-gray-900">{language_code}</p>
            </div>
          </div>
        </>
      ) : (
        <div className="p-6 bg-yellow-50 border border-yellow-200 rounded-lg text-center">
          <p className="text-yellow-800">
            Speech transcription was not successful. Please ensure clear audio quality.
          </p>
        </div>
      )}
    </div>
  );
};
