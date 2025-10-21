import { useEffect, useRef } from 'react';
import { Mic } from 'lucide-react';

interface WaveformVisualizerProps {
  isRecording: boolean;
}

export const WaveformVisualizer = ({ isRecording }: WaveformVisualizerProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    if (!isRecording) {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
      return;
    }

    const setupAudio = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        streamRef.current = stream;

        const audioContext = new AudioContext();
        const analyser = audioContext.createAnalyser();
        const microphone = audioContext.createMediaStreamSource(stream);

        analyser.fftSize = 256;
        microphone.connect(analyser);
        analyserRef.current = analyser;

        const draw = () => {
          if (!canvasRef.current || !analyserRef.current) return;

          const canvas = canvasRef.current;
          const ctx = canvas.getContext('2d');
          if (!ctx) return;

          const bufferLength = analyserRef.current.frequencyBinCount;
          const dataArray = new Uint8Array(bufferLength);
          analyserRef.current.getByteFrequencyData(dataArray);

          ctx.fillStyle = '#1f2937';
          ctx.fillRect(0, 0, canvas.width, canvas.height);

          const barWidth = (canvas.width / bufferLength) * 2.5;
          let x = 0;

          for (let i = 0; i < bufferLength; i++) {
            const barHeight = (dataArray[i] / 255) * canvas.height;

            const gradient = ctx.createLinearGradient(0, canvas.height - barHeight, 0, canvas.height);
            gradient.addColorStop(0, '#10b981');
            gradient.addColorStop(1, '#3b82f6');

            ctx.fillStyle = gradient;
            ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);

            x += barWidth + 1;
          }

          animationRef.current = requestAnimationFrame(draw);
        };

        draw();
      } catch (err) {
        console.error('Failed to setup audio visualizer:', err);
      }
    };

    setupAudio();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, [isRecording]);

  return (
    <div className="w-full max-w-2xl mx-auto bg-gray-800 rounded-lg overflow-hidden shadow-xl p-4">
      <div className="flex items-center gap-2 mb-2">
        <Mic className={`w-5 h-5 ${isRecording ? 'text-success' : 'text-gray-400'}`} />
        <span className="text-white font-semibold">Audio Input</span>
      </div>
      <canvas
        ref={canvasRef}
        width={800}
        height={100}
        className="w-full h-24 bg-gray-900 rounded"
      />
    </div>
  );
};
