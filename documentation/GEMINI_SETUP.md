# Gemini AI Setup Guide

This guide will help you set up Google's Gemini AI for comprehensive multimodal analysis.

## Prerequisites

1. **Google AI Studio Account**: You need a Google account to access Google AI Studio
2. **API Key**: Generate an API key from Google AI Studio

## Setup Steps

### 1. Install Required Package

```bash
pip install google-generativeai
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

### 2. Get Your API Key

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Click on "Get API Key" in the left sidebar
4. Create a new API key
5. Copy the API key

### 3. Configure API Key

You have two options to configure your API key:

#### Option A: Environment Variable (Recommended)
```bash
# Windows (Command Prompt)
set GOOGLE_API_KEY=your_api_key_here

# Windows (PowerShell)
$env:GOOGLE_API_KEY="your_api_key_here"

# Linux/Mac
export GOOGLE_API_KEY=your_api_key_here
```

#### Option B: Add to credentials.json
Add your API key to the existing `credentials.json` file:

```json
{
  "api_key": "your_api_key_here",
  "type": "service_account",
  "project_id": "your-project-id",
  ...
}
```

### 4. Test the Setup

Run the multimodal analysis script:

```bash
python multimodal_analysis.py
```

The script will automatically detect your API key and use Gemini AI for comprehensive analysis.

## Features

With Gemini AI integration, you'll get:

- **Comprehensive Emotional Analysis**: AI-powered insights into emotional states
- **Voice-Emotion Alignment**: Analysis of how facial expressions align with voice characteristics
- **Psychological Insights**: Professional psychological assessment based on multimodal data
- **Vocal Characteristics Analysis**: Detailed analysis of voice parameters and their emotional significance
- **Multimodal Coherence**: Assessment of how well different modalities work together
- **Professional Recommendations**: AI-generated insights and recommendations

## Troubleshooting

### Common Issues

1. **"Gemini AI not available"**: Install the package with `pip install google-generativeai`
2. **"API key not found"**: Make sure your API key is properly set in environment variables or credentials.json
3. **"Gemini configuration failed"**: Check that your API key is valid and has proper permissions

### API Key Security

- Never commit your API key to version control
- Use environment variables for production deployments
- Keep your API key secure and don't share it publicly

## Cost Information

- Google AI Studio offers free tier with generous limits
- Check current pricing at [Google AI Studio Pricing](https://aistudio.google.com/pricing)
- Monitor your usage in the Google AI Studio dashboard

## Support

If you encounter issues:

1. Check the Google AI Studio documentation
2. Verify your API key is correct
3. Ensure you have internet connectivity
4. Check the console output for specific error messages
5. See [Error Handling Guide](ERROR_HANDLING.md) for detailed troubleshooting