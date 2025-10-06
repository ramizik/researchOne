# Google Cloud Speech-to-Text Setup Guide

This guide will help you set up Google Cloud Speech-to-Text API for the multimodal analysis project.

## Prerequisites

- Google account
- Credit card (for billing, though there's a free tier)
- Python environment with the project dependencies installed

## Step 1: Create Google Cloud Project

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Select a project" → "New Project"
3. Enter project name: `multimodal-analysis` (or your preferred name)
4. Click "Create"

## Step 2: Enable Billing

1. In the Google Cloud Console, go to "Billing"
2. Link a billing account to your project
3. Note: Google provides $300 in free credits for new users

## Step 3: Enable Speech-to-Text API

1. Go to [API Library](https://console.cloud.google.com/apis/library)
2. Search for "Speech-to-Text API"
3. Click on it and press "Enable"

## Step 4: Create Service Account

1. Go to [IAM & Admin](https://console.cloud.google.com/iam-admin/serviceaccounts)
2. Click "Create Service Account"
3. Enter details:
   - Name: `multimodal-speech-service`
   - Description: `Service account for multimodal analysis speech transcription`
4. Click "Create and Continue"
5. Assign the "Editor" role (or "Speech-to-Text Client" for more restricted access)
6. Click "Done"

## Step 5: Generate JSON Key File

1. In the "Service Accounts" page, find your newly created service account
2. Click on the service account name
3. Go to the "Keys" tab
4. Click "Add Key" → "Create new key"
5. Choose "JSON" format
6. Click "Create" to download the key file
7. Save the file as `credentials.json` in your project directory

## Step 6: Place Credentials File

Simply place your downloaded `credentials.json` file in the project directory (same folder as `multimodal_analysis.py`).

The system will automatically detect and use this file for authentication. No environment variables needed!

## Step 7: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 8: Test the Setup

Run the multimodal analysis to test the integration:

```bash
python multimodal_analysis.py
```

## Troubleshooting

### Common Issues:

1. **"No credentials found" error:**
   - Ensure `credentials.json` is in the project directory
   - Verify the JSON file is valid and not corrupted
   - Check that the file is named exactly `credentials.json`

2. **"Permission denied" error:**
   - Ensure the service account has the correct permissions
   - Check that the Speech-to-Text API is enabled

3. **"Billing not enabled" error:**
   - Enable billing for your Google Cloud project
   - The free tier should be sufficient for testing

4. **"API not enabled" error:**
   - Go to the API Library and ensure Speech-to-Text API is enabled

### Testing Credentials:

You can test your credentials with this simple script:

```python
from google.cloud import speech

try:
    client = speech.SpeechClient()
    print("✅ Google Cloud Speech-to-Text credentials are working!")
except Exception as e:
    print(f"❌ Error: {e}")
```

## Cost Information

- **Free Tier:** 60 minutes of audio per month
- **Pricing:** $0.006 per 15-second increment after free tier
- **15-second recording:** Approximately $0.006 per analysis

## Security Notes

- Keep your JSON credentials file secure
- Never commit credentials to version control
- Consider using more restrictive IAM roles in production
- The `multimodal-speech-credentials.json` file should be added to `.gitignore`

## Next Steps

Once setup is complete, your multimodal analysis will include:
- Facial emotion detection
- Voice analysis (pitch, vibrato, jitter, shimmer)
- Speech transcription with confidence scores
- Word-level timing information
- Comprehensive multimodal insights

The system will gracefully handle cases where Google Cloud is not available, falling back to emotion and voice analysis only.
