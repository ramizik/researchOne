import { useState, useEffect } from 'react';
import { Circle, StopCircle, RotateCcw, AlertCircle, CheckCircle } from 'lucide-react';
import { useMediaRecorder } from '../hooks/useMediaRecorder';
import { usePermissions } from '../hooks/usePermissions';
import { CameraPreview } from './CameraPreview';
import { WaveformVisualizer } from './WaveformVisualizer';
import { RECORDING_DURATION } from '../utils/constants';

interface RecordingScreenProps {
  onRecordingComplete: (blob: Blob) => void;
}

export const RecordingScreen = ({ onRecordingComplete }: RecordingScreenProps) => {
  const { hasPermissions, permissionError, checkPermissions } = usePermissions();
  const { isRecording, recordedBlob, error, startRecording, resetRecording } = useMediaRecorder(RECORDING_DURATION);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [countdown, setCountdown] = useState<number | null>(null);

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;

    if (isRecording) {
      setElapsedTime(0);
      interval = setInterval(() => {
        setElapsedTime((prev) => {
          if (prev >= 15) {
            clearInterval(interval);
            return 15;
          }
          return prev + 1;
        });
      }, 1000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isRecording]);

  const handleStartRecording = () => {
    setCountdown(3);
    const countdownInterval = setInterval(() => {
      setCountdown((prev) => {
        if (prev === null || prev <= 1) {
          clearInterval(countdownInterval);
          startRecording();
          return null;
        }
        return prev - 1;
      });
    }, 1000);
  };

  const handleSubmit = () => {
    if (recordedBlob) {
      onRecordingComplete(recordedBlob);
    }
  };

  const handleReset = () => {
    resetRecording();
    setElapsedTime(0);
  };

  if (permissionError) {
    return (
      <div className="max-w-2xl mx-auto p-6">
        <div className="bg-white rounded-lg shadow-lg p-8 text-center">
          <AlertCircle className="w-16 h-16 text-danger mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-gray-900 mb-4">Permission Required</h2>
          <p className="text-gray-600 mb-6">{permissionError}</p>
          <button
            onClick={checkPermissions}
            className="bg-primary text-white px-6 py-3 rounded-lg font-semibold hover:bg-blue-600 transition"
          >
            Grant Permissions
          </button>
        </div>
      </div>
    );
  }

  if (!hasPermissions) {
    return (
      <div className="max-w-2xl mx-auto p-6">
        <div className="bg-white rounded-lg shadow-lg p-8 text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-gray-600">Checking permissions...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      {countdown !== null && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
          <div className="text-white text-9xl font-bold animate-pulse">{countdown}</div>
        </div>
      )}

      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Multimodal Analysis Recording</h2>
        <p className="text-gray-600 mb-6">
          Record a 15-second video for comprehensive emotion and voice analysis
        </p>

        <CameraPreview isRecording={isRecording} elapsedTime={elapsedTime} />

        <div className="mt-6">
          <WaveformVisualizer isRecording={isRecording} />
        </div>

        {error && (
          <div className="mt-6 bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-danger flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="font-semibold text-danger">Recording Error</h3>
              <p className="text-red-700 text-sm mt-1">{error}</p>
            </div>
          </div>
        )}

        {recordedBlob && (
          <div className="mt-6 bg-green-50 border border-green-200 rounded-lg p-4 flex items-start gap-3">
            <CheckCircle className="w-5 h-5 text-success flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="font-semibold text-success">Recording Complete</h3>
              <p className="text-green-700 text-sm mt-1">
                Ready to analyze your 15-second recording
              </p>
            </div>
          </div>
        )}

        <div className="mt-6 flex justify-center gap-4">
          {!isRecording && !recordedBlob && (
            <button
              onClick={handleStartRecording}
              className="flex items-center gap-2 bg-primary text-white px-8 py-4 rounded-lg font-semibold text-lg hover:bg-blue-600 transition shadow-lg"
            >
              <Circle className="w-6 h-6" />
              Start Recording
            </button>
          )}

          {isRecording && (
            <button
              disabled
              className="flex items-center gap-2 bg-danger text-white px-8 py-4 rounded-lg font-semibold text-lg shadow-lg cursor-not-allowed opacity-75"
            >
              <StopCircle className="w-6 h-6 animate-pulse" />
              Recording... ({15 - elapsedTime}s remaining)
            </button>
          )}

          {recordedBlob && (
            <>
              <button
                onClick={handleReset}
                className="flex items-center gap-2 bg-gray-600 text-white px-6 py-4 rounded-lg font-semibold hover:bg-gray-700 transition"
              >
                <RotateCcw className="w-5 h-5" />
                Re-record
              </button>
              <button
                onClick={handleSubmit}
                className="flex items-center gap-2 bg-success text-white px-8 py-4 rounded-lg font-semibold hover:bg-green-600 transition shadow-lg"
              >
                <CheckCircle className="w-5 h-5" />
                Analyze Recording
              </button>
            </>
          )}
        </div>

        <div className="mt-6 text-center text-sm text-gray-500">
          <p>Recording will automatically stop after 15 seconds</p>
        </div>
      </div>
    </div>
  );
};
