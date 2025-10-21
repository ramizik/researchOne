export type EmotionType = 'happiness' | 'sadness' | 'anger' | 'fear' | 'surprise' | 'disgust' | 'neutral';

export type VoiceType = 'bass' | 'baritone' | 'tenor' | 'alto' | 'mezzo-soprano' | 'soprano';

export type QualityLevel = 'excellent' | 'good' | 'fair' | 'poor';

export type StabilityLevel = 'very_stable' | 'stable' | 'moderately_stable' | 'variable' | 'highly_variable';

export type EmotionalComplexity = 'simple' | 'moderate' | 'complex';

export type IntensityLevel = 'high' | 'medium' | 'low';

export type SingingStyle = 'monotone' | 'controlled' | 'expressive' | 'highly_variable';

export type VibratoQuality = 'slow' | 'normal' | 'fast' | 'none';

export type VoiceTension = 'relaxed' | 'normal' | 'tense' | 'very_tense';

export type SpeakingRate = 'slow' | 'normal' | 'fast';

export type CoherenceLevel = 'high' | 'moderate' | 'low';

export interface SessionResponse {
  session_id: string;
  status: string;
  message: string;
}

export interface StatusResponse {
  session_id: string;
  status: 'pending' | 'video_received' | 'audio_received' | 'processing_video' | 'processing_audio' | 'processing_ai' | 'completed' | 'failed';
  progress: number;
  message: string;
  error: string | null;
}

export interface EmotionsBySecond {
  [second: string]: EmotionType[];
}

export interface EmotionDistribution {
  [emotion: string]: number;
}

export interface EmotionalAnalysis {
  dominant_emotion: EmotionType;
  emotion_distribution: EmotionDistribution;
  emotional_intensity: IntensityLevel;
  emotional_consistency: number;
  emotional_complexity: EmotionalComplexity;
}

export interface FacialExpressionQuality {
  detection_rate: number;
  quality_level: QualityLevel;
}

export interface EmotionalStabilityMetrics {
  stability_percentage: number;
  volatility_score: number;
  stability_level: StabilityLevel;
}

export interface EmotionAnalysis {
  emotions_by_second: EmotionsBySecond;
  emotional_analysis: EmotionalAnalysis;
  facial_expression_quality: FacialExpressionQuality;
  emotional_stability_metrics: EmotionalStabilityMetrics;
}

export interface SingingCharacteristics {
  singing_style: SingingStyle;
  overall_singing_quality: QualityLevel;
  pitch_stability: number;
  vibrato_present: boolean;
  vibrato_quality: VibratoQuality;
}

export interface EmotionalIndicators {
  energy_level: IntensityLevel;
  emotional_arousal: IntensityLevel;
  voice_tension: VoiceTension;
  voice_quality: QualityLevel;
  speaking_rate: SpeakingRate;
  breath_control: QualityLevel;
}

export interface VoiceAnalysis {
  mean_pitch: number;
  voice_type: VoiceType;
  vibrato_rate: number;
  jitter: number;
  shimmer: number;
  lowest_note: string;
  highest_note: string;
  singing_characteristics: SingingCharacteristics;
  emotional_indicators: EmotionalIndicators;
}

export interface TranscriptionAnalysis {
  transcription: string;
  confidence: number;
  success: boolean;
  word_count: number;
  language_code: string;
}

export interface MultimodalInsights {
  emotional_coherence: CoherenceLevel;
  voice_emotion_alignment: CoherenceLevel;
  overall_emotional_state: string;
  confidence_score: number;
  key_observations: string[];
}

export interface AnalysisMetadata {
  session_id: string;
  timestamp: string;
  recording_duration: number;
  analysis_version?: string;
  status: string;
}

export interface AnalysisResults {
  metadata: AnalysisMetadata;
  emotion_analysis: EmotionAnalysis;
  voice_analysis: VoiceAnalysis;
  transcription_analysis: TranscriptionAnalysis;
  multimodal_insights: MultimodalInsights;
  gemini_insights: string;
}

export interface HealthCheckResponse {
  status: string;
  timestamp: string;
  gemini_available: boolean;
}

export type AppScreen = 'recording' | 'processing' | 'results';
