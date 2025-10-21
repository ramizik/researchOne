import { useEffect, useRef } from 'react';
import { Video, VideoOff } from 'lucide-react';

interface CameraPreviewProps {
  isRecording: boolean;
  elapsedTime: number;
}

export const CameraPreview = ({ isRecording, elapsedTime }: CameraPreviewProps) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });

        streamRef.current = stream;

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error('Failed to start camera preview:', err);
      }
    };

    startCamera();

    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  return (
    <div className="relative w-full max-w-2xl mx-auto bg-gray-900 rounded-lg overflow-hidden shadow-xl">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="w-full aspect-video object-cover transform scale-x-[-1]"
      />

      {isRecording && (
        <div className="absolute top-4 left-4 flex items-center gap-2 bg-danger px-3 py-2 rounded-full shadow-lg animate-pulse">
          <div className="w-3 h-3 bg-white rounded-full" />
          <span className="text-white font-semibold">{elapsedTime}s</span>
        </div>
      )}

      <div className="absolute bottom-4 left-4 bg-black bg-opacity-50 px-3 py-2 rounded-full">
        {isRecording ? (
          <Video className="w-5 h-5 text-white" />
        ) : (
          <VideoOff className="w-5 h-5 text-gray-400" />
        )}
      </div>
    </div>
  );
};
