import os
import time
import threading
import requests
import re
import io
import base64
from queue import Queue
from dotenv import load_dotenv
from PIL import ImageGrab
from elevenlabs.client import ElevenLabs
from elevenlabs import play

# Load environment variables
load_dotenv()

# Configuration
API_URL = os.getenv('AI_SERVER_URL', 'http://localhost:5000/event')
ELEVEN_API_KEY = os.getenv('ELEVENLABS_API_KEY')
if not ELEVEN_API_KEY:
    raise ValueError("Please set ELEVENLABS_API_KEY in your .env file.")
client = ElevenLabs(api_key=ELEVEN_API_KEY)
VOICE_ID = os.getenv('ELEVEN_VOICE_ID', '54Cze5LrTSyLgbO6Fhlc')
MODEL_ID = os.getenv('ELEVEN_MODEL_ID', 'eleven_multilingual_v2')

# Paths and timings
LOG_PATH = os.path.expandvars(r'%AppData%\.minecraft\logs\latest.log')
LOG_SEND_INTERVAL = float(os.getenv('LOG_SEND_INTERVAL', '60'))  # seconds
DEATH_COOLDOWN = float(os.getenv('DEATH_COOLDOWN', '10'))        # seconds

# Patterns and state
DEATH_PATTERN = re.compile(r"(?:was slain by|fell out of the world|drowned|burned to death|suffocated|fell from a high place|was shot by|was impaled by)")
event_queue = Queue()
last_event_time = 0
last_death_time = 0


def process_event(event):
    """
    Send event to AI server and play the TTS response.
    Supports optional screenshot in event payload.
    """
    try:
        resp = requests.post(API_URL, json=event)
        resp.raise_for_status()
        ai_text = resp.json().get('response')
        if ai_text:
            print(f"[AI] {ai_text}")
            audio = client.text_to_speech.convert(
                text=ai_text,
                voice_id=VOICE_ID,
                model_id=MODEL_ID
            )
            play(audio)
    except Exception as e:
        print(f"[Sender Error] {e}")


def event_sender():
    """
    Thread to process queued events as they arrive.
    """
    while True:
        event = event_queue.get()
        process_event(event)
        event_queue.task_done()


def on_new_log_line(line):
    """Callback for each new log line."""
    global last_event_time, last_death_time
    now = time.time()
    payload = {
        'type': 'game_log',
        'window': 'Minecraft',
        'details': {'game': 'Minecraft'},
        'log': line,
        'timestamp': now,
    }
    # Immediate processing for death logs with cooldown and reset timer
    if DEATH_PATTERN.search(line):
        if now - last_death_time >= DEATH_COOLDOWN:
            last_death_time = now
            last_event_time = now
            process_event(payload)
    else:
        # Queue non-death events if interval has passed
        if now - last_event_time >= LOG_SEND_INTERVAL:
            last_event_time = now
            event_queue.put(payload)


def heartbeat():
    """Periodic heartbeat events include a screenshot."""
    global last_event_time
    while True:
        time.sleep(LOG_SEND_INTERVAL)
        now = time.time()
        if now - last_event_time >= LOG_SEND_INTERVAL:
            last_event_time = now
            payload = {
                'type': 'heartbeat',
                'window': 'Minecraft',
                'details': {},
                'timestamp': now,
            }
            # Capture and encode a screenshot
            img = ImageGrab.grab()
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            b64 = base64.b64encode(buf.getvalue()).decode()
            payload['screenshot'] = f"data:image/png;base64,{b64}"
            event_queue.put(payload)


def main():
    # Start the event sender thread
    threading.Thread(target=event_sender, daemon=True).start()
    print("[input.py] Event sender thread started.")

    # Start log tailer
    class LogTailer:
        def __init__(self, path, callback):
            self.path = path
            self.callback = callback
            self._stop = threading.Event()
        def start(self):
            threading.Thread(target=self._run, daemon=True).start()
        def _run(self):
            if not os.path.isfile(self.path):
                print(f"[LogTailer] File not found: {self.path}")
                return
            with open(self.path, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(0, os.SEEK_END)
                while not self._stop.is_set():
                    line = f.readline()
                    if line:
                        self.callback(line.strip())
                    else:
                        time.sleep(0.1)
        def stop(self):
            self._stop.set()

    tailer = LogTailer(LOG_PATH, on_new_log_line)
    tailer.start()
    print(f"[input.py] Tailing log: {LOG_PATH}")

    # Start heartbeat thread
    threading.Thread(target=heartbeat, daemon=True).start()
    print(f"[input.py] Heartbeat every {LOG_SEND_INTERVAL} seconds with screenshot.")

    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[input.py] Shutting down...")
        tailer.stop()


if __name__ == '__main__':
    main()
