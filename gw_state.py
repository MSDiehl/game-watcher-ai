import asyncio
import json
import os
import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass

import boto3
import openai
import regex as re
from botocore.config import Config
from chromadb import PersistentClient
from dotenv import load_dotenv
from hit_session import HitSession
from ore_session import OreSession
from sentence_transformers import SentenceTransformer

load_dotenv()

BASE_DIR = os.path.dirname(__file__)

# OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")
TTS_FORMAT = os.getenv("OPENAI_TTS_FORMAT", "wav")
TTS_SPEED = float(os.getenv("OPENAI_TTS_SPEED", "1.0"))
MAX_MEMORY_TEXT_CHARS = int(os.getenv("GW_MAX_MEMORY_TEXT_CHARS", "2000"))
MAX_CONTEXT_CHARS = int(os.getenv("GW_MAX_CONTEXT_CHARS", "6000"))
MAX_LOG_CHARS = int(os.getenv("GW_MAX_LOG_CHARS", "1200"))
MAX_LOG_KEYS = int(os.getenv("GW_MAX_LOG_KEYS", "40"))
MAX_LOG_LIST_ITEMS = int(os.getenv("GW_MAX_LOG_LIST_ITEMS", "20"))
MAX_EVENT_LOG = int(os.getenv("GW_MAX_EVENT_LOG", "2000"))

SCREENSHOT_ENABLED = os.getenv("GW_ENABLE_SCREENSHOTS", "0").lower() in {
    "1",
    "true",
    "yes",
}
SCREENSHOT_EVENTS = {
    e.strip()
    for e in os.getenv(
        "GW_SCREENSHOT_EVENTS",
        "death,hit_and_die,struggling,health_low,advancement,boss_spawn",
    ).split(",")
    if e.strip()
}

SENT_END = re.compile(
    r"(?<!\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|No|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec))(?<=[\.!\?…])\s+$"
)

# AWS
S3_BUCKET = os.getenv("GW_S3_BUCKET")
S3_PREFIX = os.getenv("GW_S3_PREFIX", "")
AWS_REGION = os.getenv("AWS_REGION")
s3 = None
if S3_BUCKET and AWS_REGION:
    s3 = boto3.client(
        "s3",
        region_name=AWS_REGION,
        config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
    )

print(
    f"Configuration loaded. chat_model={CHAT_MODEL}, tts_model={TTS_MODEL}, tts_voice={TTS_VOICE}, screenshots={SCREENSHOT_ENABLED}"
)

# Memory settings
MC_DAY_SECONDS = 20 * 60
MEMORY_WINDOW = 10 * MC_DAY_SECONDS

# Player state tracking
LAST_PLAYER_STATE = None

# MINING CONFIG
MIN_VEIN_SEQ = 2
SESSIONS = defaultdict(OreSession)

# Pickup config
SPECIAL_PICKUP_STATE = defaultdict(int)

# Hit session config
HIT_SESSIONS = defaultdict(HitSession)
ATTACKER_STATS = defaultdict(lambda: {"total_hits": 0, "sessions": 0})
INITIAL_HIT_ALERT = 10
THRESHOLD_FACTOR = 1.5
HIT_TOTALS = defaultdict(int)  # src -> total hits ever
NEXT_ALERT = defaultdict(lambda: INITIAL_HIT_ALERT)  # src -> next threshold

# Batching / debounce state
BATCH_IDLE_SECONDS = 2.0
SHORT_IDLE_SECONDS = 1.0
SMELT_BATCH = {}
BREW_BATCH = {}
TRADE_BATCH = {}
DROP_BATCH = {}
CRAFT_ROLLUP = defaultdict(
    lambda: {
        "items": Counter(),
        "value": 0,
        "task": None,
        "idle": SHORT_IDLE_SECONDS,
    }
)
DEATH_FALLOUT = defaultdict(
    lambda: {
        "items": Counter(),
        "event": "death",
        "cause": None,
        "count": 0,
        "task": None,
        "idle": 1.5,
    }
)

HEALTH_LOW_PENDING = {}
EFFECT_PENDING = {}

# Friendly fire config
FRIENDLY_ANIMALS = {
    "minecraft:cow",
    "minecraft:sheep",
    "minecraft:pig",
    "minecraft:chicken",
    "minecraft:rabbit",
    "minecraft:horse",
    "minecraft:donkey",
    "minecraft:mule",
    "minecraft:llama",
    "minecraft:parrot",
    "minecraft:turtle",
    "minecraft:bee",
    "minecraft:fox",
    "minecraft:cat",
    "minecraft:panda",
    "minecraft:axolotl",
}
FRIENDLY_HIT_STATE = defaultdict(
    lambda: {"threshold": random.randint(30, 50), "count": 0}
)

# Vector DB setup
chroma = PersistentClient(path=os.path.join(BASE_DIR, ".chromadb"))
collection = chroma.get_or_create_collection("mc_memory")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Items config
with open(os.path.join(BASE_DIR, "items.json"), encoding="utf-8") as f:
    raw_cfg = json.load(f)
DEFAULTS = raw_cfg.get("defaults", {})
ITEMS_CONFIG = raw_cfg.get("items", {})
print("Loaded", len(ITEMS_CONFIG), "items from items.json")


def _cfg_for(item_id: str) -> dict:
    base = dict(DEFAULTS)
    base.update(ITEMS_CONFIG.get(item_id, {}))
    return base


def item_value(item_id: str) -> int:
    return int(_cfg_for(item_id).get("value", DEFAULTS.get("value", 0)))


def item_nice(item_id: str) -> str:
    speak = _cfg_for(item_id).get("speak", {})
    return speak.get("nice_name") or item_id


def event_idle(item_id: str, ev_kind: str, fallback: float) -> float:
    evs = _cfg_for(item_id).get("events", {})
    return float(
        evs.get(ev_kind, {}).get(
            "batch_idle",
            DEFAULTS.get("events", {}).get(ev_kind, {}).get("batch_idle", fallback),
        )
    )


# Tilt state tracking
TILT_STATE = defaultdict(int)
MAX_TILT = 10
TILT_DECAY = 1

# Chat settings
CHAT_COOLDOWN_SECONDS = 8
LAST_CHAT_REPLY = defaultdict(float)


def adjust_tilt(player_id, delta):
    TILT_STATE[player_id] = max(0, min(MAX_TILT, TILT_STATE[player_id] + delta))
    print(f"[Tilt] Player {player_id} tilt now {TILT_STATE[player_id]}")
    return TILT_STATE[player_id]


# Heartbeat settings
heartbeat_reset = asyncio.Event()
HEARTBEAT_MIN = 60
HEARTBEAT_MAX = 180


@dataclass
class MemoryEvent:
    timestamp: float
    type: str
    details: dict


MEMORY = deque()
EVENT_LOG = deque(maxlen=MAX_EVENT_LOG)
comment_queue = asyncio.Queue(maxsize=200)
death_counters = defaultdict(int)

