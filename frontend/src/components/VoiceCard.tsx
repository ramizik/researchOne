import { Music, TrendingUp, Activity } from 'lucide-react';
import type { VoiceAnalysis } from '../types';
import { VOICE_TYPE_COLORS, QUALITY_COLORS } from '../utils/constants';
import { getJitterColor, getShimmerColor } from '../utils/helpers';

interface VoiceCardProps {
  data: VoiceAnalysis;
}

export const VoiceCard = ({ data }: VoiceCardProps) => {
  const { mean_pitch, voice_type, singing_characteristics, emotional_indicators, lowest_note, highest_note, jitter, shimmer, vibrato_rate } = data;

  const voiceTypeColor = VOICE_TYPE_COLORS[voice_type] || '#6b7280';
  const qualityColor = QUALITY_COLORS[singing_characteristics.overall_singing_quality];

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-2">
        <Music className="w-6 h-6" />
        Voice Analysis
      </h2>

      <div className="grid md:grid-cols-2 gap-6 mb-6">
        <div className="p-6 bg-gradient-to-br from-purple-50 to-blue-50 rounded-lg text-center">
          <p className="text-sm text-gray-600 mb-2">Mean Pitch</p>
          <p className="text-4xl font-bold text-gray-900">{mean_pitch.toFixed(1)}</p>
          <p className="text-sm text-gray-600 mt-1">Hz</p>
          <div className="mt-4">
            <span
              className="px-4 py-2 rounded-full text-white font-semibold text-sm capitalize"
              style={{ backgroundColor: voiceTypeColor }}
            >
              {voice_type.replace(/-/g, ' ')}
            </span>
          </div>
        </div>

        <div className="p-6 bg-gray-50 rounded-lg">
          <h4 className="font-semibold text-gray-900 mb-4">Pitch Range</h4>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Lowest Note</span>
              <span className="font-bold text-gray-900 text-lg">{lowest_note}</span>
            </div>
            <div className="w-full h-2 bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 rounded-full" />
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Highest Note</span>
              <span className="font-bold text-gray-900 text-lg">{highest_note}</span>
            </div>
          </div>
        </div>
      </div>

      <div className="mb-6">
        <h4 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <TrendingUp className="w-5 h-5" />
          Voice Quality Metrics
        </h4>
        <div className="grid grid-cols-3 gap-4">
          <MetricBox
            label="Jitter"
            value={(jitter * 100).toFixed(2) + '%'}
            colorClass={getJitterColor(jitter)}
          />
          <MetricBox
            label="Shimmer"
            value={(shimmer * 100).toFixed(2) + '%'}
            colorClass={getShimmerColor(shimmer)}
          />
          <MetricBox
            label="Vibrato Rate"
            value={vibrato_rate.toFixed(1) + ' Hz'}
            colorClass="text-primary"
          />
        </div>
      </div>

      <div className="mb-6 p-4 bg-gray-50 rounded-lg">
        <h4 className="font-semibold text-gray-900 mb-3">Singing Characteristics</h4>
        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <p className="text-sm text-gray-600 mb-1">Style</p>
            <p className="font-semibold text-gray-900 capitalize">
              {singing_characteristics.singing_style.replace(/_/g, ' ')}
            </p>
          </div>
          <div>
            <p className="text-sm text-gray-600 mb-1">Overall Quality</p>
            <span
              className="inline-block px-3 py-1 rounded-full text-white font-semibold text-sm capitalize"
              style={{ backgroundColor: qualityColor }}
            >
              {singing_characteristics.overall_singing_quality}
            </span>
          </div>
          <div>
            <p className="text-sm text-gray-600 mb-1">Pitch Stability</p>
            <div className="flex items-center gap-2">
              <div className="flex-1 bg-gray-200 rounded-full h-2">
                <div
                  className="h-full bg-success rounded-full"
                  style={{ width: `${Math.max(0, 100 - singing_characteristics.pitch_stability * 100)}%` }}
                />
              </div>
              <span className="text-xs text-gray-600">
                {(singing_characteristics.pitch_stability * 100).toFixed(0)}
              </span>
            </div>
          </div>
          <div>
            <p className="text-sm text-gray-600 mb-1">Vibrato</p>
            <p className="font-semibold text-gray-900 capitalize">
              {singing_characteristics.vibrato_quality}
              {singing_characteristics.vibrato_present ? ' (Present)' : ' (Absent)'}
            </p>
          </div>
        </div>
      </div>

      <div>
        <h4 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
          <Activity className="w-5 h-5" />
          Emotional Voice Indicators
        </h4>
        <div className="flex flex-wrap gap-2">
          <EmotionalBadge
            label="Energy"
            value={emotional_indicators.energy_level}
            color={emotional_indicators.energy_level === 'high' ? 'bg-orange-500' :
                   emotional_indicators.energy_level === 'medium' ? 'bg-yellow-500' : 'bg-blue-500'}
          />
          <EmotionalBadge
            label="Arousal"
            value={emotional_indicators.emotional_arousal}
            color={emotional_indicators.emotional_arousal === 'high' ? 'bg-red-500' :
                   emotional_indicators.emotional_arousal === 'medium' ? 'bg-orange-500' : 'bg-green-500'}
          />
          <EmotionalBadge
            label="Tension"
            value={emotional_indicators.voice_tension}
            color={emotional_indicators.voice_tension === 'very_tense' || emotional_indicators.voice_tension === 'tense' ? 'bg-red-500' : 'bg-green-500'}
          />
          <EmotionalBadge
            label="Quality"
            value={emotional_indicators.voice_quality}
            color={QUALITY_COLORS[emotional_indicators.voice_quality]}
          />
          <EmotionalBadge
            label="Pace"
            value={emotional_indicators.speaking_rate}
            color="bg-purple-500"
          />
          <EmotionalBadge
            label="Breath Control"
            value={emotional_indicators.breath_control}
            color={QUALITY_COLORS[emotional_indicators.breath_control]}
          />
        </div>
      </div>
    </div>
  );
};

interface MetricBoxProps {
  label: string;
  value: string;
  colorClass: string;
}

const MetricBox = ({ label, value, colorClass }: MetricBoxProps) => (
  <div className="p-4 bg-white border-2 border-gray-200 rounded-lg text-center">
    <p className="text-xs text-gray-600 mb-1">{label}</p>
    <p className={`text-xl font-bold ${colorClass}`}>{value}</p>
  </div>
);

interface EmotionalBadgeProps {
  label: string;
  value: string;
  color: string;
}

const EmotionalBadge = ({ label, value, color }: EmotionalBadgeProps) => (
  <div className={`${color} text-white px-4 py-2 rounded-full text-sm font-semibold`}>
    {label}: <span className="capitalize">{value.replace(/_/g, ' ')}</span>
  </div>
);
