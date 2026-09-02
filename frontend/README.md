# VoiceCheck Frontend v2

A polished React/Vite frontend for the AI voice detector.

## Run

```bash
npm install
npm run dev
```

By default the frontend calls `/api/analyze`, which is proxied to the FastAPI
backend at `http://localhost:5000` during development.

To use another API URL:

```bash
# PowerShell
$env:VITE_API_URL="http://localhost:5000"
npm run dev
```

## Expected backend response

The UI accepts fields such as:

```json
{
  "prediction": "fake",
  "confidence": 0.87
}
```

It also tolerates `REAL`, `FAKE`, `HUMAN`, and `AI_GENERATED` prediction labels.

## Design

The UI intentionally uses a restrained dark/graphite visual system rather than neon AI styling. The post-analysis screen becomes a compact dashboard with:

- verdict and confidence
- human vs AI probability
- audio details
- model/processing metadata
- analysis pipeline status
- waveform visualization
- microphone recording saved as WebM before analysis

The pipeline animation is presentation-layer feedback while the single `/api/analyze` request is processing; it does not claim that the backend exposes streaming progress.
