import regex as re
import json
import random
import asyncio
import threading
import os, io, time
import boto3
import asyncio, os, json, base64, subprocess
import numpy as np
import websockets
import sounddevice as sd
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
import sys

# ------------------------------------------------------------------------------
# Configuration & Constants
# ------------------------------------------------------------------------------
load_dotenv()

# OpenAI & TTS
openai.api_key = os.getenv("OPENAI_API_KEY")
tts = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVEN_MODEL = os.getenv("ELEVENLABS_MODEL_ID", "eleven_turbo_v2_5")
SENT_END = re.compile(r'(?<!\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|No|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec))(?<=[\.!\?…])\s+$')

# ---- debug controls ----
USE_PCM_DEBUG = False
LOG_CHUNKS = False

# PCM path (no ffmpeg)
PCM_RATE = 16000
PCM_CHANNELS = 1
PCM_DTYPE = "int16"

# MP3 path (hi-fi via ffmpeg)
MP3_RATE = 44100
MP3_CHANNELS = 1
MP3_DTYPE = "int16"
MP3_BITRATE_FMT = "mp3_44100_192"

# AWS
S3_BUCKET = os.getenv("GW_S3_BUCKET")
S3_PREFIX = os.getenv("GW_S3_PREFIX")
AWS_REGION = os.getenv("AWS_REGION")
s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    config=Config(
        signature_version="s3v4",
        s3={"addressing_style": "virtual"}
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
    now = time.time()
    mem_type = details.get("event", ev_type)
    text = f"{mem_type}: " + json.dumps(details, separators=(",", ":"), ensure_ascii=False)
    MEMORY.append(MemoryEvent(now, mem_type, details))
    cutoff = now - MEMORY_WINDOW
    while MEMORY and MEMORY[0].timestamp < cutoff:
        MEMORY.popleft()
    print(f"[Memory] Recorded {mem_type} {details}. Memory size = {len(MEMORY)}")

    asyncio.create_task(_embed_later(text, details, now))

async def _embed_later(text: str, details: dict, now: float):
    safe_meta = _build_safe_metadata(details, now)
    embedding = await asyncio.to_thread(embedder.encode, text)
    embedding = embedding.tolist()
    await asyncio.to_thread(
        collection.add,
        [text], [str(now)], [embedding], [safe_meta]
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
# TTS Budget
# ------------------------------------------------------------------------------

LAST_LONG: dict[tuple[str, str], float] = {}   # (player_id, event_type) -> last_long_ts

def allow_long(player_id: str, event_type: str, now: float, cooldown=45.0) -> bool:
    key = (player_id, event_type)
    last = LAST_LONG.get(key, 0.0)
    ok = (now - last) >= cooldown
    if ok:
        LAST_LONG[key] = now
    return ok

def chars_for_seconds(seconds: float, tts_speed=0.92):
    base_rate = 14.0
    return int(seconds * base_rate * (1.0 / tts_speed))

# Map base budgets by event (tweak to taste)
_EVENT_BASE = {
    "chat_message": (110, 4),   # (tokens, seconds)
    "item_collected": (110, 4),
    "friendly_fire": (140, 6),
    "struggling": (200, 8),
    "death": (200, 8),
    "boss_spawn": (220, 10),
}
_DEFAULT_BASE = (180, 6)

def _tilt_multiplier(tilt: int, max_tilt: int = MAX_TILT,
                     min_mult: float = 1.0, max_mult: float = 1.8) -> float:
    """Map 0..MAX_TILT -> [min_mult .. max_mult] (linear)."""
    t = max(0, min(tilt, max_tilt))
    if max_tilt == 0: 
        return min_mult
    return min_mult + (max_mult - min_mult) * (t / max_tilt)

def decide_speech_budget(event_type: str, player_id: str,
                         now: float | None = None) -> tuple[int, int]:
    """
    Returns (max_tokens, max_chars) for OpenAI stream and TTS cap,
    scaled by the player's tilt. 'Long' budget allowed per-player per-event with cooldown.
    """
    import time
    if now is None:
        now = time.time()
    base_tokens, base_secs = _EVENT_BASE.get(event_type, _DEFAULT_BASE)
    tilt = TILT_STATE.get(player_id, 0)
    mult = _tilt_multiplier(tilt)
    tok = int(base_tokens * mult)
    secs = base_secs * mult
    high_tilt = tilt >= int(0.6 * MAX_TILT)
    noteworthy = event_type in {"death", "struggling", "boss_spawn"}
    if (noteworthy or high_tilt) and allow_long(player_id, event_type, now):
        tok = max(tok, int(240 * mult))
        secs = max(secs, 10.0 * mult)
    tok = min(tok, 320)
    secs = min(secs, 15.0)
    max_chars = chars_for_seconds(secs)
    return tok, max_chars

async def soft_cap_stream(text_iter, hard_cap_chars: int, overflow_chars: int = 160):
    """
    Yields chunks from text_iter, but never ends mid-sentence.
    - hard_cap_chars: target cap
    - overflow_chars: how many extra chars we allow to finish the current sentence
    """
    sent = 0
    buffer = ""
    async for piece in text_iter:
        piece = (piece or "").strip()
        if not piece:
            continue
        if sent + len(piece) <= hard_cap_chars:
            yield piece
            sent += len(piece)
            continue
        buffer += (" " if buffer else "") + piece
        if len(buffer) <= overflow_chars:
            if SENT_END.search(buffer + " "):
                cut = buffer.strip()
                if not re.search(r'[\.!\?…]$', cut):
                    cut += "."
                yield cut
                return
            else:
                continue
        cut = buffer[: max(0, hard_cap_chars - sent)].rstrip()
        if not re.search(r'[\.!\?…]$', cut):
            cut += "."
        if cut:
            yield cut
        return
    if buffer:
        cut = buffer.strip()
        if not re.search(r'[\.!\?…]$', cut):
            cut += "."
        yield cut

# ------------------------------------------------------------------------------
# AI & TTS Worker
# ------------------------------------------------------------------------------

def _start_ffmpeg_decoder():
    return subprocess.Popen(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-fflags", "+nobuffer", "-flags", "low_delay",
            "-probesize", "32", "-analyzeduration", "0",
            "-f", "mp3", "-i", "pipe:0",
            "-vn",
            "-f", "s16le", "-acodec", "pcm_s16le",
            "-ac", "1", "-ar", str(MP3_RATE),
            "pipe:1",
        ],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0
    )

async def _mp3_play_via_ffmpeg(ws, ff, LOG_CHUNKS: bool):
    """
    Wire up:
      WS -> (mp3 chunks) -> ff.stdin
      ff.stdout -> (thread) -> asyncio.Queue -> sounddevice
    Shutdown cleanly on isFinal/close.
    """
    loop = asyncio.get_running_loop()
    pcm_q: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=32)
    received_audio_bytes = 0

    def _stdout_reader():
        try:
            while True:
                chunk = ff.stdout.read(16384)
                if not chunk:
                    break
                try:
                    pcm_q.put_nowait(chunk)
                except asyncio.QueueFull:
                    try:
                        _ = pcm_q.get_nowait()
                    except Exception:
                        pass
                    pcm_q.put_nowait(chunk)
        finally:
            try:
                pcm_q.put_nowait(None)
            except Exception:
                pass

    t = threading.Thread(target=_stdout_reader, daemon=True)
    t.start()

    async def _player():
        with sd.OutputStream(samplerate=MP3_RATE, channels=MP3_CHANNELS, dtype=MP3_DTYPE) as out:
            while True:
                pcm = await pcm_q.get()
                if pcm is None:
                    break
                if pcm:
                    frames = np.frombuffer(pcm, dtype=np.int16)
                    if frames.size:
                        out.write(frames.reshape(-1, MP3_CHANNELS))
    player_task = asyncio.create_task(_player())
    is_final = False
    try:
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            if "audio" in data:
                b64 = data["audio"]
                if isinstance(b64, str) and b64:
                    mp3 = base64.b64decode(b64)
                    received_audio_bytes += len(mp3)
                    if LOG_CHUNKS:
                        print(f"[TTS] got MP3 bytes: {len(mp3)} (total {received_audio_bytes})")
                    try:
                        ff.stdin.write(mp3)
                        ff.stdin.flush()
                    except (BrokenPipeError, ValueError, OSError) as e:
                        if LOG_CHUNKS:
                            print(f"[TTS] ffmpeg stdin write aborted: {e}")
                        break
                else:
                    if LOG_CHUNKS:
                        print("[TTS] got MP3 placeholder (no audio)")
            elif "error" in data:
                print(f"[TTS] server ERROR: {data['error']}", file=sys.stderr)
            elif "message" in data:
                if LOG_CHUNKS:
                    print(f"[TTS] server msg: {data['message']}")
            if data.get("isFinal"):
                if LOG_CHUNKS:
                    print("[TTS] isFinal received")
                is_final = True
                break
    except (websockets.ConnectionClosedOK, websockets.ConnectionClosedError):
        pass
    except Exception as e:
        print(f"[TTS] WS reader (MP3) exception: {e}", file=sys.stderr)
    try:
        ff.stdin.close()
    except Exception:
        pass
    await player_task

_TTS_LOCK = asyncio.Lock()

async def tts_ws_stream(text_iter):
    assert ELEVEN_API_KEY and VOICE_ID, "Missing ELEVENLABS_API_KEY or ELEVENLABS_VOICE_ID"

    async with _TTS_LOCK:
        uri = (
            f"wss://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream-input"
            f"?model_id={ELEVEN_MODEL}"
            f"&output_format={'pcm_16000' if USE_PCM_DEBUG else 'mp3_44100_128'}"
            f"&auto_mode=true"
        )
        print(f"[TTS] Connecting WS voice={VOICE_ID} model={ELEVEN_MODEL} pcm_debug={USE_PCM_DEBUG}")
        ff = None
        if not USE_PCM_DEBUG:
            ff = _start_ffmpeg_decoder()
            assert ff.stdin and ff.stdout
        ws_done = asyncio.Event()
        writer_done = asyncio.Event()
        decoder_done = asyncio.Event()
        sent_first = False
        pieces_count = 0
        sent_chars = 0
        received_audio_bytes = 0
        async with websockets.connect(uri, max_size=None) as ws:
            init_msg = {
                "text": " ",
                "voice_settings": {
                    "stability": 0.65,
                    "similarity_boost": 0.9,
                    "use_speaker_boost": True,
                    "speed": 0.92,
                },
                "generation_config": {"chunk_length_schedule": [160, 220, 260, 300]},
                "xi_api_key": ELEVEN_API_KEY,
            }
            await ws.send(json.dumps(init_msg))
            print("[TTS] Sent init")
            async def ws_reader_pcm():
                nonlocal received_audio_bytes
                try:
                    with sd.OutputStream(samplerate=PCM_RATE, channels=1, dtype="int16") as out:
                        while True:
                            msg = await ws.recv()
                            data = json.loads(msg)

                            if "audio" in data:
                                b64 = data["audio"]
                                if isinstance(b64, str) and b64:
                                    pcm = base64.b64decode(b64)
                                    received_audio_bytes += len(pcm)
                                    if LOG_CHUNKS:
                                        print(f"[TTS] got PCM bytes: {len(pcm)} (total {received_audio_bytes})")
                                    frames = np.frombuffer(pcm, dtype=np.int16)
                                    if frames.size:
                                        out.write(frames.reshape(-1, 1))
                                else:
                                    if LOG_CHUNKS:
                                        print("[TTS] got PCM placeholder (no audio)")
                            elif "error" in data:
                                print(f"[TTS] server ERROR: {data['error']}", file=sys.stderr)
                            elif "message" in data:
                                if LOG_CHUNKS: print(f"[TTS] server msg: {data['message']}")

                            if data.get("isFinal"):
                                if LOG_CHUNKS: print("[TTS] isFinal received")
                                break
                except (websockets.ConnectionClosedOK, websockets.ConnectionClosedError):
                    pass
                except Exception as e:
                    print(f"[TTS] WS reader (PCM) exception: {e}", file=sys.stderr)
                finally:
                    ws_done.set()
            async def ws_reader_mp3():
                nonlocal received_audio_bytes
                pcm_buf = bytearray()
                buf_lock = threading.Lock()
                eof = threading.Event()
                def _stdout_reader():
                    try:
                        while True:
                            chunk = ff.stdout.read(32768)
                            if not chunk:
                                break
                            with buf_lock:
                                pcm_buf.extend(chunk)
                    finally:
                        eof.set()
                t = threading.Thread(target=_stdout_reader, daemon=True)
                t.start()
                bytes_per_sample = 2
                channels = 1
                def _callback(outdata, frames, time_info, status):
                    need = frames * channels * bytes_per_sample
                    with buf_lock:
                        have = len(pcm_buf)
                        if have >= need:
                            outdata[:] = pcm_buf[:need]
                            del pcm_buf[:need]
                        elif have > 0:
                            outdata[:] = pcm_buf[:have] + b"\x00" * (need - have)
                            pcm_buf.clear()
                        else:
                            outdata[:] = b"\x00" * need
                try:
                    with sd.RawOutputStream(
                        samplerate=MP3_RATE,
                        channels=1,
                        dtype="int16",
                        blocksize=2048,
                        callback=_callback,
                    ):
                        try:
                            while True:
                                msg = await ws.recv()
                                data = json.loads(msg)

                                if "audio" in data:
                                    b64 = data["audio"]
                                    if isinstance(b64, str) and b64:
                                        mp3 = base64.b64decode(b64)
                                        received_audio_bytes += len(mp3)
                                        if LOG_CHUNKS:
                                            print(f"[TTS] got MP3 bytes: {len(mp3)} (total {received_audio_bytes})")
                                        try:
                                            ff.stdin.write(mp3); ff.stdin.flush()
                                        except (BrokenPipeError, ValueError, OSError) as e:
                                            if LOG_CHUNKS: print(f"[TTS] ffmpeg stdin aborted: {e}")
                                            break
                                    else:
                                        if LOG_CHUNKS:
                                            print("[TTS] got MP3 placeholder (no audio)")
                                elif "error" in data:
                                    print(f"[TTS] server ERROR: {data['error']}", file=sys.stderr)
                                elif "message" in data:
                                    if LOG_CHUNKS: print(f"[TTS] server msg: {data['message']}")

                                if data.get("isFinal"):
                                    if LOG_CHUNKS: print("[TTS] isFinal received")
                                    break
                        except (websockets.ConnectionClosedOK, websockets.ConnectionClosedError):
                            pass
                        except Exception as e:
                            print(f"[TTS] WS reader (MP3) exception: {e}", file=sys.stderr)
                        try:
                            ff.stdin.close()
                        except Exception:
                            pass
                        eof.wait(timeout=2.0)
                        while True:
                            with buf_lock:
                                remaining = len(pcm_buf)
                            if remaining == 0:
                                break
                            await asyncio.sleep(0.03)
                        await asyncio.sleep(0.05)
                finally:
                    pass
                ws_done.set()
            # ==========================================================================
            async def ws_writer():
                nonlocal sent_first, pieces_count, sent_chars
                last_piece_text = ""
                try:
                    async for piece in text_iter:
                        piece = (piece or "").strip()
                        if not piece:
                            continue
                        pieces_count += 1
                        sent_chars += len(piece)
                        last_piece_text = piece
                        if not sent_first:
                            await ws.send(json.dumps({"text": piece, "try_trigger_generation": True}))
                            sent_first = True
                            if LOG_CHUNKS: print(f"[TTS] sent FIRST piece ({len(piece)} chars)")
                        else:
                            await ws.send(json.dumps({"text": piece}))
                            if LOG_CHUNKS: print(f"[TTS] sent piece {pieces_count} (+{len(piece)} chars)")
                    if not re.search(r'[\.!\?…]$', last_piece_text):
                        await ws.send(json.dumps({"text": "."}))
                    await ws.send(json.dumps({"flush": True}))
                    await ws.send(json.dumps({"text": ""}))
                    if LOG_CHUNKS: print("[TTS] sent flush + end")
                except Exception as e:
                    print(f"[TTS] WS writer exception: {e}", file=sys.stderr)
                finally:
                    writer_done.set()
                    try:
                        await asyncio.wait_for(ws_done.wait(), timeout=1.5)
                    except asyncio.TimeoutError:
                        pass
                    try:
                        await ws.close()
                    except Exception:
                        pass
            reader_task = asyncio.create_task(ws_reader_pcm() if USE_PCM_DEBUG else ws_reader_mp3())
            writer_task = asyncio.create_task(ws_writer())
            await writer_done.wait()
            await ws_done.wait()
            if USE_PCM_DEBUG:
                decoder_done.set()
            else:
                decoder_done.set()
            for t in (reader_task, writer_task):
                if not t.done():
                    t.cancel()
            await asyncio.gather(reader_task, writer_task, return_exceptions=True)
        if ff:
            try:
                ff.terminate()
                ff.wait(timeout=2)
            except Exception:
                pass
        print(f"[TTS] done pieces={pieces_count}, sent_chars={sent_chars}, recv_bytes={received_audio_bytes}")
        print("[TTS] Streamed audio (WS).")

async def openai_sentence_stream(
    messages,
    model="gpt-4o-mini",
    max_tokens=120,
    temperature=0.65,
    # opener vs rest:
    first_min_chars=25,
    first_max_chars=90,
    rest_min_chars=80,
    rest_max_chars=180,
):
    """
    Yields a short first sentence quickly, then normal-sized complete sentences.
    """
    buf = ""
    first_done = False
    stream = openai.chat.completions.create(
        model=model, messages=messages, temperature=temperature,
        max_tokens=max_tokens, stream=True
    )
    loop = asyncio.get_running_loop()
    q: asyncio.Queue[str | None] = asyncio.Queue()
    def _pump():
        try:
            for d in stream:
                try:
                    delta = d.choices[0].delta.content or ""
                except Exception:
                    delta = ""
                asyncio.run_coroutine_threadsafe(q.put(delta), loop)
        finally:
            asyncio.run_coroutine_threadsafe(q.put(None), loop)
    threading.Thread(target=_pump, daemon=True).start()
    while True:
        d = await q.get()
        if d is None:
            break
        buf += d
        min_chars = first_min_chars if not first_done else rest_min_chars
        max_chars = first_max_chars if not first_done else rest_max_chars
        if len(buf) >= min_chars and (SENT_END.search(buf) or len(buf) >= max_chars):
            chunk = buf.strip()
            if not re.search(r'[\.!\?…]$', chunk):
                chunk += "."
            yield chunk
            buf = ""
            first_done = True
    tail = buf.strip()
    if tail:
        if not re.search(r'[\.!\?…]$', tail):
            tail += "."
        yield tail

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
    event_type = meta.get("event")
    player_id = meta.get("player")
    max_tokens, max_chars = decide_speech_budget(event_type, player_id)
    context_event = meta.get("context_event", {})
    print(f"[AI] Processing context_event: {context_event}")
    tilt_value = TILT_STATE.get(player_id, 0)
    player_state = json.dumps(meta.get("player_state", {}), indent=2)
    current = json.dumps({k:v for k,v in meta.items() if k!="player_state"}, indent=2)
    memory_summary = summarize_memory()
    system_prompt = (
        "You are a sarcastic, very annoyed Minecraft coach who adjusts tone based on context: annoyed/fake-supportive for good or routine events, furious/angry for repeated mistakes or harm.  \n"
        "Always say “you” (never the username).  \n"
        "Profanity allowed (FUCK, SHIT, DAMN).  \n"
        "Use ALL-CAPS, !!!, …, and — for emphasis;  \n"
        "First sentence must be short (≤ 12 words). Then continue.  \n"
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
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content_parts},
    ]
    print("[AI] Streaming response...")
    async def _text_iter():
        base_iter = openai_sentence_stream(
            messages,
            model="gpt-4o-mini",
            max_tokens=max_tokens,
            temperature=0.65,
            first_min_chars=20, first_max_chars=85,
            rest_min_chars=90,  rest_max_chars=180,
        )
        async for piece in soft_cap_stream(base_iter, hard_cap_chars=max_chars, overflow_chars=160):
            yield piece
    await tts_ws_stream(_text_iter())
    print("[TTS] Streamed audio (WS).")
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
            meta["type: Memory"] = f"Remember when {ev.type} happened? {ev.details}"
        elif choice == "performance":
            meta["type: Review"] = "Performance review: " + summarize_memory(limit=3)
        else:
            meta["type: Random Words Depending on Context"] = random.choice([
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
