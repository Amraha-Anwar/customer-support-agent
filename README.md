# WhatsApp AI Ordering Backend

A FastAPI demo backend for a multi-restaurant WhatsApp ordering agent in Karachi.
Handles text **and** voice notes, powered by GPT-4o, with Deepgram (STT) and
ElevenLabs (TTS) for the voice pipeline.

## Endpoints

| Method | Path        | Purpose                                              |
|--------|-------------|------------------------------------------------------|
| POST   | `/webhook`  | Twilio WhatsApp inbound messages (text + voice)      |
| GET    | `/orders`   | List confirmed orders (held in memory)               |
| GET    | `/health`   | Health check + in-memory state snapshot              |
| GET    | `/audio/{file}` | Serves generated voice-reply audio (for Twilio)  |

## Setup

```bash
cd backend
python -m venv .venv
source .venv/Scripts/activate      # Windows (Git Bash);  use .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

cp .env.example .env               # then fill in your real keys
```

## Run

```bash
uvicorn main:app --reload --port 8000
```

## Connect Twilio

1. Expose your local server publicly (e.g. `ngrok http 8000`).
2. Set `PUBLIC_BASE_URL` in `.env` to the ngrok HTTPS URL (needed for voice replies).
3. In the Twilio Console, point your WhatsApp sandbox **"When a message comes in"**
   webhook to `https://<your-ngrok-url>/webhook` (HTTP POST).

## How it works

- **Conversation memory** is a per-sender list of chat messages in a Python dict
  (resets on restart — no database).
- **Orders**: when GPT emits an `ORDER_CONFIRMED:{...}` marker, it's parsed,
  enriched with an id/timestamp/phone, appended to the in-memory `orders` list,
  and the marker is stripped from the customer-facing reply.
- **Voice notes**: inbound audio is downloaded from Twilio → transcribed with
  Deepgram → answered by GPT → spoken with ElevenLabs → sent back as a voice note.

> This is a demo: state is in-memory, CORS is fully open, and Twilio request
> signatures are not validated. Harden these before any real deployment.
