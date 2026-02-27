# game-watcher-ai
This AI is tailored to watching Minecraft gameplay and making live commentary with OpenAI for both text generation and TTS.

Required env vars:
- `OPENAI_API_KEY`

Optional tuning env vars:
- `OPENAI_CHAT_MODEL` (default: `gpt-4o-mini`)
- `OPENAI_TTS_MODEL` (default: `gpt-4o-mini-tts`)
- `OPENAI_TTS_VOICE` (default: `alloy`)
- `OPENAI_TTS_FORMAT` (recommended: `wav`)
- `OPENAI_TTS_SPEED` (default: `1.0`)
- `GW_ENABLE_SCREENSHOTS` (`1`/`0`, default off)
- `GW_SCREENSHOT_EVENTS` (comma-separated notable events)
- `GW_MAX_EVENT_LOG` (default: `2000`)

Code layout:
- `server.py`: FastAPI entrypoint and routing only.
- `gw_state.py`: runtime config, shared mutable state, and item helpers.
- `gw_memory.py`: memory compaction, embedding, and memory summaries.
- `gw_media.py`: screenshot capture and S3 cleanup.
- `gw_ai.py`: OpenAI text generation + TTS worker pipeline.
- `gw_events.py`: event handlers, batching, scoring, enqueue logic, and background loops.
