import asyncio
import json
import time

from fastapi import FastAPI, Request

import gw_ai
import gw_events
import gw_memory as mem
import gw_state as st

app = FastAPI()


@app.post("/event")
async def handle_event(req: Request):
    ev = await req.json()
    if ev.get("type") == "player_position":
        return {"response": None}
    if ev.get("type") == "player_state":
        st.LAST_PLAYER_STATE = mem.compact_for_context(ev)
        return {"response": None}

    ts = ev.get("timestamp", time.time())
    idx = len(st.EVENT_LOG)
    compact_ev = mem.compact_for_context(ev)
    print(
        f"[Event] Received: {mem.limit_text(json.dumps(compact_ev, ensure_ascii=False), st.MAX_LOG_CHARS)}"
    )
    st.EVENT_LOG.append((ts, compact_ev))
    await gw_events.process_event(idx, ts, ev)
    return {"response": None}


@app.on_event("startup")
async def start_services():
    print("[Startup] Launching analyzer and TTS worker")
    asyncio.create_task(gw_events.session_flusher())
    asyncio.create_task(gw_ai.worker_loop())
    asyncio.create_task(gw_events.heartbeat_easter_eggs())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000, access_log=False, log_level="warning")
