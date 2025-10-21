import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import type { EmotionAnalysis } from '../types';
import { EMOTION_COLORS, EMOTION_EMOJIS } from '../utils/constants';
import { formatPercentage } from '../utils/helpers';

interface EmotionCardProps {
  data: EmotionAnalysis;
}

export const EmotionCard = ({ data }: EmotionCardProps) => {
  const { emotional_analysis, emotions_by_second, facial_expression_quality, emotional_stability_metrics } = data;

  const chartData = Object.entries(emotional_analysis.emotion_distribution).map(([emotion, value]) => ({
    name: emotion.charAt(0).toUpperCase() + emotion.slice(1),
    value: value,
    color: EMOTION_COLORS[emotion as keyof typeof EMOTION_COLORS] || '#6b7280',
  }));

  const timelineData = Object.entries(emotions_by_second).sort((a, b) => parseInt(a[0]) - parseInt(b[0]));

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-2">
        <span>Facial Emotion Analysis</span>
      </h2>

      <div className="mb-8 text-center p-6 bg-gradient-to-br from-blue-50 to-purple-50 rounded-lg">
        <div className="text-6xl mb-3">
          {EMOTION_EMOJIS[emotional_analysis.dominant_emotion]}
        </div>
        <h3 className="text-3xl font-bold text-gray-900 capitalize mb-2">
          {emotional_analysis.dominant_emotion}
        </h3>
        <p className="text-gray-600">Dominant Emotion</p>
      </div>

      <div className="grid md:grid-cols-2 gap-6 mb-8">
        <div>
          <h4 className="font-semibold text-gray-900 mb-4">Emotion Distribution</h4>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={chartData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={(props: any) => `${props.name}: ${Number(props.value).toFixed(1)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip formatter={(value: unknown) => `${Number(value).toFixed(1)}%`} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="space-y-4">
          <MetricRow
            label="Emotional Intensity"
            value={emotional_analysis.emotional_intensity}
            badge
          />
          <MetricRow
            label="Consistency"
            value={formatPercentage(emotional_analysis.emotional_consistency)}
            progress={emotional_analysis.emotional_consistency}
          />
          <MetricRow
            label="Complexity"
            value={emotional_analysis.emotional_complexity}
            badge
          />
          <MetricRow
            label="Detection Quality"
            value={facial_expression_quality.quality_level}
            badge
          />
        </div>
      </div>

      <div className="mb-6">
        <h4 className="font-semibold text-gray-900 mb-3">Emotion Timeline (by second)</h4>
        <div className="flex gap-1 flex-wrap">
          {timelineData.map(([second, emotions]) => {
            const primaryEmotion = emotions[0];
            const color = EMOTION_COLORS[primaryEmotion];
            return (
              <div
                key={second}
                className="flex-1 min-w-[50px] h-12 rounded flex items-center justify-center text-white font-semibold text-sm cursor-pointer hover:opacity-80 transition"
                style={{ backgroundColor: color }}
                title={`${parseInt(second) + 1}s: ${emotions.join(', ')}`}
              >
                {parseInt(second) + 1}
              </div>
            );
          })}
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-4 p-4 bg-gray-50 rounded-lg">
        <div>
          <p className="text-sm text-gray-600 mb-1">Stability</p>
          <p className="text-xl font-bold text-gray-900 capitalize">
            {emotional_stability_metrics.stability_level.replace(/_/g, ' ')}
          </p>
          <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
            <div
              className="h-full bg-success rounded-full"
              style={{ width: `${emotional_stability_metrics.stability_percentage}%` }}
            />
          </div>
        </div>
        <div>
          <p className="text-sm text-gray-600 mb-1">Volatility Score</p>
          <p className="text-xl font-bold text-gray-900">
            {emotional_stability_metrics.volatility_score.toFixed(2)}
          </p>
          <p className="text-xs text-gray-500 mt-1">
            {emotional_stability_metrics.volatility_score < 0.3 ? 'Very Stable' :
             emotional_stability_metrics.volatility_score < 0.6 ? 'Moderate' : 'Variable'}
          </p>
        </div>
      </div>
    </div>
  );
};

interface MetricRowProps {
  label: string;
  value: string | number;
  badge?: boolean;
  progress?: number;
}

const MetricRow = ({ label, value, badge, progress }: MetricRowProps) => (
  <div>
    <div className="flex justify-between items-center mb-1">
      <span className="text-sm text-gray-600">{label}</span>
      {badge ? (
        <span className="px-3 py-1 bg-primary bg-opacity-10 text-primary rounded-full text-sm font-semibold capitalize">
          {value}
        </span>
      ) : (
        <span className="font-semibold text-gray-900">{value}</span>
      )}
    </div>
    {progress !== undefined && (
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div
          className="h-full bg-primary rounded-full"
          style={{ width: `${progress}%` }}
        />
      </div>
    )}
  </div>
);
