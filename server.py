import os
import re
import json
import time
import random
import asyncio
from collections import deque, Counter, defaultdict
from dataclasses import dataclass
from fastapi import FastAPI, Request
import openai
from ore_session import OreSession
from hit_session import HitSession
from elevenlabs import ElevenLabs, play
from dotenv import load_dotenv
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------------------
# Configuration & Constants
# ------------------------------------------------------------------------------
load_dotenv()

# OpenAI & TTS
openai.api_key = os.getenv("OPENAI_API_KEY")
tts = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")

print("Configuration loaded. Voice ID =", VOICE_ID)

# Memory settings
MC_DAY_SECONDS = 20 * 60
MEMORY_WINDOW  = 10 * MC_DAY_SECONDS

#Player state tracking
LAST_PLAYER_STATE = None

#MINING CONFIG
MIN_VEIN_SEQ = 2
STATE = defaultdict(lambda: {"breaks": deque(), "matched": 0})
SESSIONS = defaultdict(OreSession)

#Pikcup config
SPECIAL_PICKUP_STATE = defaultdict(int)

#Hit session config
HIT_SESSIONS = defaultdict(HitSession)
ATTACKER_STATS = defaultdict(lambda: {"total_hits": 0, "sessions": 0})
INITIAL_HIT_ALERT = 10
THRESHOLD_FACTOR = 1.5
HIT_TOTALS = defaultdict(int)      # src -> total hits ever
NEXT_ALERT = defaultdict(lambda: INITIAL_HIT_ALERT)  # src -> next threshold

#Friendly fire config
FRIENDLY_ANIMALS = {
    "minecraft:cow", "minecraft:sheep", "minecraft:pig", "minecraft:chicken",
    "minecraft:rabbit", "minecraft:horse", "minecraft:donkey", "minecraft:mule",
    "minecraft:llama", "minecraft:parrot", "minecraft:turtle", "minecraft:bee",
    "minecraft:fox", "minecraft:cat", "minecraft:panda", "minecraft:axolotl"
}

FRIENDLY_HIT_STATE = defaultdict(lambda: {
    "threshold": random.randint(30, 50),
    "count": 0
})

# Vector DB setup (persistent client)
chroma = PersistentClient(path=".chromadb")
collection = chroma.get_or_create_collection("mc_memory")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load Item Categories from JSON
with open("items.json") as f:
    _cfg = json.load(f)["items"]
ITEMS_CONFIG = _cfg
print("Loaded", len(ITEMS_CONFIG), "pickup configs from items.json")

# Data Structures
@dataclass
class MemoryEvent:
    timestamp: float
    type: str
    details: dict

MEMORY = deque()
EVENT_LOG = deque(maxlen=2000)
CONSUMED_INDICES = set()
comment_queue = asyncio.Queue()
death_counters = defaultdict(int)

app = FastAPI()

# ------------------------------------------------------------------------------
# Memory Management
# ------------------------------------------------------------------------------
def record_memory(ev_type: str, **details):
    """
    Record an event in memory and vector DB.
    Args:
        ev_type (str): Type of event (e.g., "death", "pickup").
        details (dict): Additional details about the event.
    """
    mem_type = details.get("event", ev_type)
    now = time.time()
    MEMORY.append(MemoryEvent(now, mem_type, details))
    cutoff = now - MEMORY_WINDOW
    while MEMORY and MEMORY[0].timestamp < cutoff:
        MEMORY.popleft()
    print(f"[Memory] Recorded {mem_type} {details}. Memory size = {len(MEMORY)}")
    text = f"{mem_type}: {details}"
    embedding = embedder.encode(text).tolist()
    collection.add(
        documents=[text],
        ids=[str(now)],
        embeddings=[embedding],
        metadatas=[{"timestamp": now, **details}]
    )

def summarize_memory(limit: int = 5) -> str:
    """
    Build a brief summary of recent events:
      - Deaths (counts per cause)
      - Damage taken (counts per attacker)
      - Item pickups (counts per item)
      - Friendly-fire incidents (counts per entity)
    """
    lines = []

    death_events = [ev for ev in MEMORY if ev.type == "death"]
    if death_events:
        total_deaths = len(death_events)
        causes = Counter(ev.details.get("cause", "unknown") for ev in death_events)
        parts = [f"{cnt} to {cause}" for cause, cnt in causes.items()]
        lines.append(f"You died {total_deaths} times ({', '.join(parts)}).")

    hurt_events = [ev for ev in MEMORY if ev.type == "player_hurt"]
    if hurt_events:
        total_hurts = len(hurt_events)
        attackers = Counter(ev.details.get("attacker", "unknown") for ev in hurt_events)
        parts = [f"{cnt} by {att}" for att, cnt in attackers.items()]
        lines.append(f"You took damage {total_hurts} times ({', '.join(parts)}).")

    friendly_events = [ev for ev in MEMORY if ev.type == "friendly_fire"]
    if friendly_events:
        entity_counts = Counter(ev.details.get("entity", "unknown") for ev in friendly_events)
        for entity, cnt in entity_counts.items():
            lines.append(f"You triggered friendly fire on {cnt}x {entity}.")

    if len(lines) > limit:
        lines = lines[-limit:]

    summary = "\n".join(f"- {l}" for l in lines) if lines else "- Nothing memorable yet."
    print(f"[Memory] Summary:\n{summary}")
    return summary

# ------------------------------------------------------------------------------
# AI & TTS Worker
# ------------------------------------------------------------------------------

async def enqueue_meta(meta: dict):
    """
    Enqueue metadata for processing by the TTS worker.
    Args:
        meta (dict): Metadata to enqueue.
    """
    print(f"[Queue] Enqueuing metadata: {meta}")
    await comment_queue.put(meta)

async def _worker():
    """
    Worker that processes metadata from the queue and generates TTS responses.
    This runs in a loop, waiting for new metadata to process.
    """
    while True:
        meta = await comment_queue.get()
        print(f"[TTS Worker] Got meta: {meta}")
        try:
            await _call_ai_and_tts(meta)
        except Exception as ex:
            print(f"[TTS Worker] Error: {ex}")
        comment_queue.task_done()

async def _call_ai_and_tts(meta: dict):
    """
    Call the AI model to generate a response based on the metadata.
    Args:
        meta (dict): Metadata containing the event details.
    """
    context_event = meta.get("context_event", {})
    print(f"[AI] Processing context_event: {context_event}")
    player_state = json.dumps(meta.get("player_state", {}), indent=2)
    current = json.dumps({k:v for k,v in meta.items() if k!="player_state"}, indent=2)
    memory_summary = summarize_memory()
    query_emb = embedder.encode(current).tolist()
    results = collection.query(query_embeddings=[query_emb], n_results=5)["metadatas"][0]
    system_prompt = (
        "You are a sarcastic, very annoyed Minecraft coach who adjusts tone based on context: annoyed/fake-supportive for good or routine events, furious/angry for repeated mistakes or harm.  \n"
        "Always say “you” (never the username).  \n"
        "Profanity allowed (FUCK, SHIT, DAMN).  \n"
        "Use ALL-CAPS, !!!, …, and — for emphasis; may wrap key words in <emphasis level='strong'>…</emphasis>.  \n"
        "Occasionally rant 1-2 sentences if it fits.  \n"
    )
    user_prompt = (        
        f"MEMORY SUMMARY:\n{memory_summary}\n\n"
        f"NEW EVENT: {current}\n\n"
        f"CURRENT PLAYER CONTEXT:\n{player_state}\n\n"
    )
    print(f"[AI] Prompt:\n{user_prompt}")
    resp = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
            ],
        max_tokens=120,
        temperature=1.0
    )
    text = resp.choices[0].message.content.strip()
    print(f"[AI] Response: {text}")
    audio = tts.text_to_speech.convert(text=text, voice_id=VOICE_ID)
    play(audio)
    print("[TTS] Played audio.")

# ------------------------------------------------------------------------------
# Event Handlers
# ------------------------------------------------------------------------------
def on_player_hurt(idx: int, ev: dict):
    """
    Every time you get hurt (mob, cactus, fall, etc.), increment a
    never‐forgotten counter and alert when you cross each dynamically
    increasing milestone.
    """
    if ev.get("type") != "player_hurt":
        return None
    src = ev.get("damage_type", "unknown")
    HIT_TOTALS[src] += 1
    total = HIT_TOTALS[src]
    if total >= NEXT_ALERT[src]:
        meta = {
            "event":    "struggling",
            "attacker": src,
            "hits":     total
        }
        NEXT_ALERT[src] *= 2
        return meta
    return None

def on_player_death(idx: int, ev: dict):
    """
    Called on player_death events. Tracks deaths per cause.
    If the cause is already being tracked, it finalizes the session.
    """
    if ev.get("type") != "player_death":
        return None
    cause = ev.get("cause", "unknown")
    ts = ev.get("timestamp", time.time())
    sess = HIT_SESSIONS.get(cause)
    stats = ATTACKER_STATS[cause]
    sessions = stats["sessions"]
    total_hits= stats["total_hits"]
    if sessions > 0:
        avg = total_hits / sessions
        threshold = max(INITIAL_HIT_ALERT, int(avg * THRESHOLD_FACTOR))
    else:
        threshold = INITIAL_HIT_ALERT
    if sess:
        stats["sessions"]  += 1
        stats["total_hits"] += sess.count
        del HIT_SESSIONS[cause]
        if sess.count >= threshold:
            return {
                "event":    "hit_and_die",
                "attacker": cause,
                "hits":     sess.count,
                "cause":    cause
            }
    death_counters[cause] += 1
    return {
        "event": "death",
        "cause": cause,
        "count": death_counters[cause]
    }

def on_block_break(idx: int, ev: dict):
    """
    Called on every block_break event. Tracks ore breaks and starts a session.
    If the block is an ore, it records the break and updates the session.
    """
    if ev.get("type") != "block_break": return None
    blk = ev.get("block","")
    if not blk.endswith("_ore"): return None
    ore = blk[:-4]
    ts  = time.time()
    SESSIONS[ore].record_break(idx, ts)
    return None

def on_item_pickup(idx: int, ev: dict):
    """
    Called on item_pickup events. Updates ore sessions based on pickups.
    If the ore is being tracked, it records the pickup and updates the session.
    """
    if ev.get("type") != "item_pickup": return None
    ore = ev["item"]
    cnt = ev.get("count",1)
    sess = SESSIONS.get(ore)
    if not sess or not sess.pending:
        return None
    ts = time.time()
    sess.record_pickup(cnt, ts)
    return None

def on_special_item_pickup(idx: int, ev: dict):
    """
    Fires when you cross the threshold for any item in items.json.
    Resets that counter so you only alert once per threshold.
    """
    if ev.get("type") != "item_pickup":
        return None
    item = ev["item"]
    cfg  = ITEMS_CONFIG.get(item)
    if not cfg:
        return None
    cnt = ev.get("count", 1)
    SPECIAL_PICKUP_STATE[item] += cnt
    if SPECIAL_PICKUP_STATE[item] < cfg["threshold"]:
        return None
    SPECIAL_PICKUP_STATE[item] = 0
    return {
        "event": "item_collected",
        "item": item,
        "category": cfg["category"],
        "threshold": cfg["threshold"]
    }

def on_friendly_hit(idx: int, ev: dict):
    """
    Called on every event. If it's an entity_hit on a friendly animal,
    we increment its counter. Once it reaches its threshold, we emit
    a single 'friendly_fire' event and reset its state.
    """
    if ev.get("type") != "entity_hit":
        return None
    ent = ev.get("entity_type", "")
    if ent not in FRIENDLY_ANIMALS:
        return None
    state = FRIENDLY_HIT_STATE[ent]
    state["count"] += 1
    print(f"[Detect] {ent} hit count: {state['count']}/{state['threshold']}")
    if state["count"] < state["threshold"]:
        return None
    FRIENDLY_HIT_STATE[ent] = {
        "threshold": random.randint(30, 50),
        "count": 0
    }
    return { 
        "event":  "friendly_fire",
        "entity": ent
    }

# ------------------------------------------------------------------------------
# Event Processing and Session Management
# ------------------------------------------------------------------------------
async def process_event(idx: int, ts: float, ev: dict):
    """
    Process a single event and trigger appropriate handlers.
    This function is called for every event received.
    """
    for fn in (on_block_break, on_item_pickup, on_friendly_hit, on_player_hurt, on_player_death):
        meta = fn(idx, ev)
        if meta:
            print(f"[Analyzer] {fn.__name__} emitted meta: {meta}")
            meta["context_event"] = LAST_PLAYER_STATE
            await enqueue_meta(meta)
            record_memory(fn.__name__, **meta)

async def session_flusher():
    """
    Periodically checks all sessions and flushes expired ones.
    If a session has expired, it finalizes it and emits the metadata.
    """
    while True:
        now = time.time()
        for ore, sess in list(SESSIONS.items()):
            if sess.is_expired(now):
                matched = sess.matched
                sess.pending.clear()
                sess.matched = 0
                sess.due_ts  = 0.0
                if matched >= MIN_VEIN_SEQ:
                    meta = {"event": "vein_mine", "ore": ore, "blocks": matched}
                elif matched > 0:
                    meta = {"event": "pickup", "item": ore, "count": matched}
                else:
                    del SESSIONS[ore]
                    continue
                print(f"[Analyzer] session end → {meta}")
                await enqueue_meta(meta)
                record_memory(meta["event"], **meta)
                del SESSIONS[ore]
        for attacker, sess in list(HIT_SESSIONS.items()):
            if sess.is_expired(now):
                stats = ATTACKER_STATS[attacker]
                stats["sessions"]  += 1
                stats["total_hits"] += sess.count
                del HIT_SESSIONS[attacker]
                sessions = stats["sessions"]
                total_hits = stats["total_hits"]
                avg = total_hits / sessions if sessions>0 else 0
                threshold = max(INITIAL_HIT_ALERT, int(avg * THRESHOLD_FACTOR))
                if sess.count >= threshold:
                    meta = {
                        "event": "struggling",
                        "attacker": attacker,
                        "hits": sess.count
                    }
                    print(f"[Analyzer] hit‐session end → {meta}")
                    await enqueue_meta(meta)
                    record_memory("struggling", **meta)
        await asyncio.sleep(0.5)

@app.post("/event")
async def handle_event(req: Request):
    """
    Handle incoming game events.
    Expects JSON with a 'type' field and other relevant data.
    """
    ev = await req.json()
    if ev.get("type") == "player_position":
        return {"response": None}
    if ev.get("type") == "player_state":
        global LAST_PLAYER_STATE
        LAST_PLAYER_STATE = ev
        return {"response": None}
    if LAST_PLAYER_STATE is not None:
        ev["player_state"] = LAST_PLAYER_STATE
    ts = ev["timestamp"]
    idx = len(EVENT_LOG)
    print(f"[Event] Received: {ev}")
    EVENT_LOG.append((ts, ev))
    await process_event(idx, ts, ev)
    return {"response": None}

@app.on_event("startup")
async def start_services():
    """
    Startup tasks to initialize the server.
    """
    print("[Startup] Launching analyzer and TTS worker")
    asyncio.create_task(session_flusher())
    asyncio.create_task(_worker())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000, access_log=False, log_level="warning")
