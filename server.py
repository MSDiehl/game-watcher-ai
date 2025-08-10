import json
import random
import asyncio
import os, io, time
import boto3
from botocore.config import Config
from mss import mss
from PIL import Image
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

# AWS
S3_BUCKET = os.getenv("GW_S3_BUCKET")
S3_PREFIX = os.getenv("GW_S3_PREFIX")
AWS_REGION = os.getenv("AWS_REGION")
s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    config=Config(
        signature_version="s3v4",
        s3={"addressing_style": "virtual"}  # ensures bucketname.s3.<region>.amazonaws.com
    ),
)

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

# Tilt state tracking
TILT_STATE = defaultdict(int)  # player_id -> tilt value
MAX_TILT = 10
TILT_DECAY = 1

# Chat settings
CHAT_COOLDOWN_SECONDS = 8
LAST_CHAT_REPLY = defaultdict(float)  # player_id -> last reply ts


def adjust_tilt(player_id, delta):
    TILT_STATE[player_id] = max(0, min(MAX_TILT, TILT_STATE[player_id] + delta))
    print(f"[Tilt] Player {player_id} tilt now {TILT_STATE[player_id]}")
    return TILT_STATE[player_id]

# Heartbeat settings
# Heartbeat control
heartbeat_reset = asyncio.Event()
HEARTBEAT_MIN = 60   # 1 minute
HEARTBEAT_MAX = 180  # 3 minutes


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

# 2) Harden metadata builders

def _to_meta_primitive(val, max_len=2000):
    """Return a Chroma-safe primitive; stringify non-primitives; never return None."""
    if isinstance(val, (str, int, float, bool)):
        return val
    if val is None:
        return "null"
    try:
        s = json.dumps(val, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        s = str(val)
    if len(s) > max_len:
        s = s[:max_len - 3] + "..."
    return s

def _build_safe_metadata(details: dict, now: float) -> dict:
    """Flatten/convert details so all values are primitives; drop large/volatile keys and any Nones."""
    skip_keys = {
        "screenshot_b64",
        "screenshot_url",
        "screenshot_s3_key",
        "player_state_raw",
    }
    md = {"timestamp": float(now)}
    for k, v in details.items():
        if k in skip_keys:
            continue
        if v is None:
            continue
        md[k] = _to_meta_primitive(v)
    return md

def record_memory(ev_type: str, **details):
    mem_type = details.get("event", ev_type)
    now = time.time()
    text = f"{mem_type}: " + json.dumps(details, separators=(",", ":"), ensure_ascii=False)
    MEMORY.append(MemoryEvent(now, mem_type, details))
    cutoff = now - MEMORY_WINDOW
    while MEMORY and MEMORY[0].timestamp < cutoff:
        MEMORY.popleft()
    print(f"[Memory] Recorded {mem_type} {details}. Memory size = {len(MEMORY)}")
    embedding = embedder.encode(text).tolist()
    safe_meta = _build_safe_metadata(details, now)

    collection.add(
        documents=[text],
        ids=[str(now)],
        embeddings=[embedding],
        metadatas=[safe_meta],
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

def capture_screenshot_to_s3_url(
    max_side_px: int = 1280,
    jpeg_quality: int = 70,
    expires: int = 180
) -> tuple[str | None, str | None]:
    """
    Capture primary monitor, downscale+JPEG, upload private to S3, return (presigned_url, s3_key).
    URL is regional and short-lived. Delete the object after the AI call.
    """
    try:
        with mss() as sct:
            raw = sct.grab(sct.monitors[1])
            img = Image.frombytes("RGB", raw.size, raw.rgb)

        w, h = img.size
        scale = max(w, h) / float(max_side_px)
        if scale > 1:
            img = img.resize((int(w/scale), int(h/scale)), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        buf.seek(0)

        ts = int(time.time() * 1000)
        key = f"{S3_PREFIX}shot_{ts}.jpg"

        s3.upload_fileobj(
            buf,
            S3_BUCKET,
            key,
            ExtraArgs={"ContentType": "image/jpeg", "ACL": "private"},
        )

        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": key},
            ExpiresIn=expires,
        )
        global_host = ".s3.amazonaws.com"
        regional_host = f".s3.{AWS_REGION}.amazonaws.com"
        if global_host in url and regional_host not in url:
            url = url.replace(global_host, regional_host)

        return url, key
    except Exception as e:
        print(f"[Screenshot/S3] Failed: {e}")
        return None, None

def delete_s3_object(key: str):
    try:
        s3.delete_object(Bucket=S3_BUCKET, Key=key)
    except Exception as e:
        print(f"[Screenshot/S3] Delete failed: {e}")

async def enqueue_meta(meta: dict):
    """
    Enqueue metadata for processing by the TTS worker.
    Args:
        meta (dict): Metadata to enqueue.
    """
    print(f"[Queue] Enqueuing metadata: {meta}")

    if meta.get("event") != "heartbeat":
        reset_heartbeat_timer()
    
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
    player_id = meta.get("player")
    tilt_value = TILT_STATE.get(player_id, 0)
    player_state = json.dumps(meta.get("player_state", {}), indent=2)
    current = json.dumps({k:v for k,v in meta.items() if k!="player_state"}, indent=2)
    memory_summary = summarize_memory()
    system_prompt = (
        "You are a sarcastic, very annoyed Minecraft coach who adjusts tone based on context: annoyed/fake-supportive for good or routine events, furious/angry for repeated mistakes or harm.  \n"
        "Always say “you” (never the username).  \n"
        "Profanity allowed (FUCK, SHIT, DAMN).  \n"
        "Use ALL-CAPS, !!!, …, and — for emphasis;  \n"
        "Occasionally rant 1-2 sentences if it fits.  \n"
    )
    system_prompt += f"Current rage level (0 calm - {MAX_TILT} max rage): {tilt_value}. Adjust your tone accordingly.\n"
    user_prompt = (        
        f"MEMORY SUMMARY:\n{memory_summary}\n\n"
        f"NEW EVENT: {current}\n\n"
        f"CURRENT PLAYER CONTEXT:\n{player_state}\n\n"
    )
    content_parts = [{"type": "text", "text": user_prompt}]
    if meta.get("screenshot_url"):
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": meta["screenshot_url"]},
            #"detail": "low"
        })
    resp = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_parts}
            ],
        max_tokens=160,
        temperature=1.0
    )
    text = resp.choices[0].message.content.strip()
    print(f"[AI] Response: {text}")
    audio = tts.text_to_speech.convert(text=text, voice_id=VOICE_ID)
    play(audio)
    print("[TTS] Played audio.")
    key = meta.get("screenshot_s3_key")
    if key:
        delete_s3_object(key)

# ------------------------------------------------------------------------------
# Event Handlers
# ------------------------------------------------------------------------------

def on_chat_message(idx: int, ev: dict):
    """
    Respond to player chat. Basic cooldown per player to avoid spam.
    Emits a 'player_chat' meta that flows into the AI/TTS worker.
    """
    if ev.get("type") != "chat_message":
        return None
    player = ev.get("player", "unknown")
    msg = (ev.get("message") or "").strip()
    now = time.time()
    if now - LAST_CHAT_REPLY[player] < CHAT_COOLDOWN_SECONDS:
        return None
    LAST_CHAT_REPLY[player] = now
    return {
        "event": "player_chat",
        "player": player,
        "message": msg
    }

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
    player_id = ev.get("player")
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
                "event": "hit_and_die",
                "attacker": cause,
                "hits": sess.count,
                "cause": cause
            }
    death_counters[cause] += 1
    adjust_tilt(player_id, +2)
    return {
        "event": "death",
        "cause": cause,
        "count": death_counters[cause],
        "player": player_id,
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
    for fn in (on_block_break, 
               on_item_pickup, 
               on_friendly_hit, 
               on_player_hurt, 
               on_player_death, 
               on_chat_message, 
               on_special_item_pickup):
        meta = fn(idx, ev)
        if meta:
            print(f"[Analyzer] {fn.__name__} emitted meta: {meta}")
            url, key = capture_screenshot_to_s3_url()
            if url and key:
                meta["screenshot_url"] = url
                meta["screenshot_s3_key"] = key
            if LAST_PLAYER_STATE is not None:
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
                url, key = capture_screenshot_to_s3_url()
                if url and key:
                    meta["screenshot_url"] = url
                    meta["screenshot_s3_key"] = key
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
                    url, key = capture_screenshot_to_s3_url()
                    if url and key:
                        meta["screenshot_url"] = url
                        meta["screenshot_s3_key"] = key
                    await enqueue_meta(meta)
                    record_memory("struggling", **meta)
        await asyncio.sleep(0.5)

# ------------------------------------------------------------------------------
# FastAPI Endpoints & heartbeat
# ------------------------------------------------------------------------------

async def heartbeat_easter_eggs():
    """
    Fires a snarky 'heartbeat' meta every 1-3 minutes, but the countdown
    resets whenever any other AI-bound event is enqueued.
    """
    while True:
        wait_time = random.randint(HEARTBEAT_MIN, HEARTBEAT_MAX)
        heartbeat_reset.clear()
        sleep_task = asyncio.create_task(asyncio.sleep(wait_time))
        reset_task = asyncio.create_task(heartbeat_reset.wait())
        done, pending = await asyncio.wait(
            {sleep_task, reset_task},
            return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        if reset_task in done and heartbeat_reset.is_set():
            print("[Heartbeat] Timer reset due to recent AI event.")
            continue
        meta = {"event": "heartbeat"}
        choice = random.choice(["memory", "performance", "random_tip"])
        if choice == "memory" and len(MEMORY) > 0:
            ev = random.choice(list(MEMORY))
            meta["comment"] = f"Remember when {ev.type} happened? {ev.details}"
        elif choice == "performance":
            meta["comment"] = "Performance review: " + summarize_memory(limit=3)
        else:
            meta["comment"] = random.choice([
                "Hey, ever thought about NOT dying?",
                "Pro tip: Lava is hot.",
                "Do you even know how to swing that pickaxe?",
                "This is SOOOO boring to watch!",
                "You know you can place blocks, right?",
                "I could do this better with my eyes closed.",
            ])
        meta["context_event"] = LAST_PLAYER_STATE
        await enqueue_meta(meta)

def reset_heartbeat_timer():
    """Signal the heartbeat loop to restart its wait window."""
    if not heartbeat_reset.is_set():
        heartbeat_reset.set()

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
    asyncio.create_task(heartbeat_easter_eggs())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000, access_log=False, log_level="warning")
