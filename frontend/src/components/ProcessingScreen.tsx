import { Loader2, Brain, Video, Music, Sparkles } from 'lucide-react';
import type { StatusResponse } from '../types';

interface ProcessingScreenProps {
  status: StatusResponse | null;
}

export const ProcessingScreen = ({ status }: ProcessingScreenProps) => {
  const getStatusIcon = () => {
    if (!status) return <Loader2 className="w-8 h-8 animate-spin" />;

    switch (status.status) {
      case 'processing_video':
        return <Video className="w-8 h-8" />;
      case 'processing_audio':
        return <Music className="w-8 h-8" />;
      case 'processing_ai':
        return <Brain className="w-8 h-8" />;
      default:
        return <Loader2 className="w-8 h-8 animate-spin" />;
    }
  };

  const getStageStatus = (stage: string): 'completed' | 'current' | 'pending' => {
    if (!status) return 'pending';

    const stages = ['processing_video', 'processing_audio', 'processing_ai', 'completed'];
    const currentIndex = stages.indexOf(status.status);
    const stageIndex = stages.indexOf(stage);

    if (stageIndex < currentIndex) return 'completed';
    if (stageIndex === currentIndex) return 'current';
    return 'pending';
  };

  const progress = status?.progress || 0;

  return (
    <div className="max-w-3xl mx-auto p-6">
      <div className="bg-white rounded-lg shadow-lg p-8">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-20 h-20 bg-primary bg-opacity-10 rounded-full mb-4">
            <div className="text-primary">{getStatusIcon()}</div>
          </div>
          <h2 className="text-3xl font-bold text-gray-900 mb-2">Analyzing Your Recording</h2>
          <p className="text-gray-600">
            {status?.message || 'Processing your multimodal analysis...'}
          </p>
        </div>

        <div className="mb-8">
          <div className="flex justify-between text-sm font-medium text-gray-700 mb-2">
            <span>Progress</span>
            <span>{progress}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-primary to-success rounded-full transition-all duration-500 ease-out"
              style={{ width: `${progress}%` }}
            >
              <div className="h-full w-full bg-gradient-to-r from-transparent via-white to-transparent opacity-30 animate-pulse" />
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <ProcessingStage
            icon={<Video className="w-6 h-6" />}
            title="Facial Emotion Analysis"
            description="Detecting emotions frame by frame"
            status={getStageStatus('processing_video')}
          />

          <ProcessingStage
            icon={<Music className="w-6 h-6" />}
            title="Voice Characteristics Analysis"
            description="Analyzing pitch, tone, and vocal patterns"
            status={getStageStatus('processing_audio')}
          />

          <ProcessingStage
            icon={<Brain className="w-6 h-6" />}
            title="AI Insights Generation"
            description="Generating comprehensive psychological analysis"
            status={getStageStatus('processing_ai')}
          />

          <ProcessingStage
            icon={<Sparkles className="w-6 h-6" />}
            title="Finalizing Results"
            description="Compiling multimodal insights"
            status={getStageStatus('completed')}
          />
        </div>

        <div className="mt-8 text-center text-sm text-gray-500">
          <p>This may take 30-60 seconds depending on video quality</p>
        </div>
      </div>
    </div>
  );
};

interface ProcessingStageProps {
  icon: React.ReactNode;
  title: string;
  description: string;
  status: 'completed' | 'current' | 'pending';
}

const ProcessingStage = ({ icon, title, description, status }: ProcessingStageProps) => {
  const getStatusColor = () => {
    switch (status) {
      case 'completed':
        return 'text-success border-success bg-green-50';
      case 'current':
        return 'text-primary border-primary bg-blue-50 animate-pulse';
      case 'pending':
        return 'text-gray-400 border-gray-300 bg-gray-50';
    }
  };

  return (
    <div className={`flex items-start gap-4 p-4 rounded-lg border-2 ${getStatusColor()} transition-all duration-300`}>
      <div className="flex-shrink-0 mt-1">{icon}</div>
      <div className="flex-1 min-w-0">
        <h3 className="font-semibold text-gray-900">{title}</h3>
        <p className="text-sm text-gray-600">{description}</p>
      </div>
      {status === 'completed' && (
        <div className="flex-shrink-0">
          <div className="w-6 h-6 bg-success rounded-full flex items-center justify-center">
            <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          </div>
        </div>
      )}
    </div>
  );
};
