export const formatDuration = (ms: number): string => {
  const seconds = Math.floor(ms / 1000);
  return `${seconds}s`;
};

export const formatTimestamp = (isoString: string): string => {
  const date = new Date(isoString);
  return date.toLocaleString();
};

export const downloadJSON = (data: unknown, filename: string): void => {
  const json = JSON.stringify(data, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

export const getJitterColor = (jitter: number): string => {
  if (jitter < 0.01) return 'text-success';
  if (jitter < 0.02) return 'text-warning';
  return 'text-danger';
};

export const getShimmerColor = (shimmer: number): string => {
  if (shimmer < 0.03) return 'text-success';
  if (shimmer < 0.05) return 'text-warning';
  return 'text-danger';
};

export const formatPercentage = (value: number): string => {
  return `${value.toFixed(1)}%`;
};

export const formatConfidence = (value: number): string => {
  return `${(value * 100).toFixed(0)}%`;
};
