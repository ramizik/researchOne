# Multimodal Emotion & Voice Analysis - Frontend

A modern React application for real-time facial emotion detection, voice analysis, speech transcription, and AI-powered psychological insights.

## Features

- **15-Second Recording**: Simultaneous video and audio capture with visual feedback
- **Real-Time Processing**: Live camera preview and audio waveform visualization
- **Comprehensive Analysis**:
  - Facial emotion detection (happiness, sadness, anger, fear, surprise, disgust, neutral)
  - Voice characteristics analysis (pitch, tone, vibrato, jitter, shimmer)
  - Speech transcription with confidence scoring
  - AI-powered multimodal insights using Google Gemini
- **Professional UI**: Responsive design with interactive charts and visualizations
- **Export Capabilities**: Download analysis results as JSON

## Prerequisites

- Node.js 18+ and npm
- Backend API running on `http://localhost:8000` (see backend README)
- Modern browser with camera and microphone support

## Installation

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Configure environment**:
   Create a `.env.local` file:
   ```
   VITE_API_BASE_URL=http://localhost:8000
   ```

3. **Start development server**:
   ```bash
   npm run dev
   ```

   The app will be available at `http://localhost:5173`

## Usage

### Recording Flow

1. **Grant Permissions**: Allow camera and microphone access when prompted
2. **Start Recording**: Click "Start Recording" button
3. **Wait for Countdown**: 3-2-1 countdown before recording starts
4. **Automatic Stop**: Recording automatically stops after 15 seconds
5. **Review & Submit**: Preview your recording and click "Analyze Recording"

### Processing

- Real-time progress updates (0-100%)
- Stage-by-stage processing visualization:
  - Facial Emotion Analysis
  - Voice Characteristics Analysis
  - AI Insights Generation
  - Finalizing Results

### Results

The results dashboard displays:

#### Emotion Analysis
- Dominant emotion with emoji and distribution chart
- Timeline visualization showing emotions per second
- Emotional stability and consistency metrics
- Facial detection quality assessment

#### Voice Analysis
- Mean pitch and voice type classification
- Pitch range with musical notes
- Voice quality metrics (jitter, shimmer, vibrato)
- Singing characteristics and style
- Emotional voice indicators

#### Speech Transcription
- Full transcription text
- Confidence score
- Word count and language detection

#### AI Insights
- Overall emotional state
- Emotional coherence and voice-emotion alignment
- Key observations
- Comprehensive Gemini AI psychological analysis

### Actions

- **Export JSON**: Download complete analysis results
- **New Recording**: Start a fresh analysis session

## Technology Stack

- **React 18** with TypeScript
- **Vite** for fast development and building
- **Tailwind CSS** for styling
- **Recharts** for data visualization
- **Lucide React** for icons
- **Axios** for API communication
- **MediaRecorder API** for recording

## Project Structure

```
frontend/
├── src/
│   ├── components/       # React components
│   │   ├── CameraPreview.tsx
│   │   ├── WaveformVisualizer.tsx
│   │   ├── RecordingScreen.tsx
│   │   ├── ProcessingScreen.tsx
│   │   ├── ResultsScreen.tsx
│   │   ├── EmotionCard.tsx
│   │   ├── VoiceCard.tsx
│   │   ├── TranscriptionCard.tsx
│   │   ├── InsightsCard.tsx
│   │   └── ErrorBoundary.tsx
│   ├── hooks/           # Custom React hooks
│   │   ├── useMediaRecorder.ts
│   │   ├── useAnalysisPolling.ts
│   │   └── usePermissions.ts
│   ├── services/        # API service layer
│   │   └── api.ts
│   ├── types/           # TypeScript type definitions
│   │   └── index.ts
│   ├── utils/           # Helper functions
│   │   ├── constants.ts
│   │   └── helpers.ts
│   ├── App.tsx          # Main application component
│   ├── main.tsx         # Application entry point
│   └── index.css        # Global styles
├── .env.local           # Environment configuration
├── package.json
├── tsconfig.json
├── vite.config.ts
└── tailwind.config.js
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_BASE_URL` | Backend API base URL | `http://localhost:8000` |

## Browser Requirements

- Chrome 90+ (recommended)
- Firefox 88+
- Safari 14+
- Edge 90+

**Required Features**:
- MediaRecorder API support
- getUserMedia API support
- WebRTC support

## Troubleshooting

### Camera/Microphone Not Working

1. **Check Permissions**: Ensure browser has camera/microphone permissions
2. **HTTPS Required**: Some browsers require HTTPS for media access (except localhost)
3. **Device Availability**: Verify devices are connected and not in use by other apps

### API Connection Errors

1. **Backend Running**: Ensure backend API is running on `http://localhost:8000`
2. **CORS Issues**: Backend should allow requests from frontend origin
3. **Network**: Check firewall settings

### Recording Issues

1. **Browser Support**: Verify browser supports required MediaRecorder codecs
2. **Storage Space**: Ensure sufficient browser storage available
3. **Memory**: Close other tabs if experiencing performance issues

### Build Issues

```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Clear Vite cache
rm -rf node_modules/.vite
npm run dev
```

## Building for Production

```bash
# Build optimized production bundle
npm run build

# Preview production build locally
npm run preview
```

The built files will be in the `dist/` directory.

## API Integration

The frontend communicates with the backend REST API:

- `POST /api/analysis/start` - Start new session
- `POST /api/analysis/upload-video` - Upload video file
- `POST /api/analysis/upload-audio` - Upload audio file
- `GET /api/analysis/status/:id` - Poll analysis status
- `GET /api/analysis/results/:id` - Get complete results

See backend API documentation for full details.

## Performance Optimization

- Lazy loading for heavy components
- Debounced resize handlers
- Optimized chart rendering
- Proper cleanup of media streams
- Efficient state management

## Accessibility

- Keyboard navigation support
- ARIA labels for screen readers
- Color contrast compliance
- Focus indicators
- Responsive design for all screen sizes

## License

For research and educational purposes.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify backend API is running and accessible
3. Check browser console for error messages
4. Review network requests in browser DevTools
