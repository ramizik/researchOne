import { useState, useEffect } from 'react';

interface UsePermissionsReturn {
  hasPermissions: boolean;
  permissionError: string | null;
  checkPermissions: () => Promise<void>;
}

export const usePermissions = (): UsePermissionsReturn => {
  const [hasPermissions, setHasPermissions] = useState(false);
  const [permissionError, setPermissionError] = useState<string | null>(null);

  const checkPermissions = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      });

      stream.getTracks().forEach((track) => track.stop());

      setHasPermissions(true);
      setPermissionError(null);
    } catch (err) {
      const error = err as Error;
      setHasPermissions(false);

      if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
        setPermissionError('Camera and microphone access denied. Please grant permissions to continue.');
      } else if (error.name === 'NotFoundError') {
        setPermissionError('No camera or microphone found. Please connect a device and try again.');
      } else {
        setPermissionError('Failed to access camera and microphone. Please check your device settings.');
      }
    }
  };

  useEffect(() => {
    checkPermissions();
  }, []);

  return {
    hasPermissions,
    permissionError,
    checkPermissions,
  };
};
