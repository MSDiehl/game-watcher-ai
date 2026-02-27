import asyncio
import io
import json
import threading
import wave

import numpy as np
import openai
import regex as re
import sounddevice as sd

import gw_memory as mem
import gw_state as st
from gw_media import delete_s3_object

LAST_LONG: dict[tuple[str, str], float] = {}
_TTS_LOCK = asyncio.Lock()

_EVENT_BASE = {
    "player_chat": (200, 6),
    "item_collected": (200, 6),
    "friendly_fire": (300, 10),
    "struggling": (500, 15),
    "death": (500, 15),
    "boss_spawn": (600, 18),
}
_DEFAULT_BASE = (180, 6)


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


def _tilt_multiplier(
    tilt: int, max_tilt: int = st.MAX_TILT, min_mult: float = 1.0, max_mult: float = 1.8
) -> float:
    t = max(0, min(tilt, max_tilt))
    if max_tilt == 0:
        return min_mult
    return min_mult + (max_mult - min_mult) * (t / max_tilt)


def decide_speech_budget(
    event_type: str, player_id: str, now: float | None = None
) -> tuple[int, int]:
    import time

    if now is None:
        now = time.time()
    base_tokens, base_secs = _EVENT_BASE.get(event_type, _DEFAULT_BASE)
    tilt = st.TILT_STATE.get(player_id, 0)
    mult = _tilt_multiplier(tilt)
    tok = int(base_tokens * mult)
    secs = base_secs * mult
    high_tilt = tilt >= int(0.6 * st.MAX_TILT)
    noteworthy = event_type in {"death", "struggling", "boss_spawn"}
    if (noteworthy or high_tilt) and allow_long(player_id, event_type, now):
        tok = max(tok, int(240 * mult))
        secs = max(secs, 10.0 * mult)
    secs = min(secs, 15.0)
    max_chars = chars_for_seconds(secs)
    return tok, max_chars


async def soft_cap_stream(text_iter, hard_cap_chars: int, overflow_chars: int = 160):
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
            if st.SENT_END.search(buffer + " "):
                cut = buffer.strip()
                if not re.search(r"[\.!\?…]$", cut):
                    cut += "."
                yield cut
                return
            continue
        cut = buffer[: max(0, hard_cap_chars - sent)].rstrip()
        if not re.search(r"[\.!\?…]$", cut):
            cut += "."
        if cut:
            yield cut
        return
    if buffer:
        cut = buffer.strip()
        if not re.search(r"[\.!\?…]$", cut):
            cut += "."
        yield cut


def _speech_create_kwargs(text: str) -> dict:
    kwargs = {
        "model": st.TTS_MODEL,
        "voice": st.TTS_VOICE,
        "input": text,
    }
    if st.TTS_SPEED and abs(st.TTS_SPEED - 1.0) > 0.001:
        kwargs["speed"] = st.TTS_SPEED
    return kwargs


def _speech_response_bytes(resp) -> bytes:
    if resp is None:
        return b""
    if isinstance(resp, (bytes, bytearray)):
        return bytes(resp)
    if hasattr(resp, "read"):
        return resp.read()
    if hasattr(resp, "iter_bytes"):
        return b"".join(resp.iter_bytes())
    if hasattr(resp, "content"):
        content = resp.content
        if isinstance(content, str):
            return content.encode("utf-8")
        return bytes(content)
    return bytes(resp)


def _create_speech_bytes(text: str) -> bytes:
    kwargs = _speech_create_kwargs(text)
    attempt_kwargs = [
        dict(kwargs, format=st.TTS_FORMAT),
        dict(kwargs, response_format=st.TTS_FORMAT),
        kwargs,
    ]
    last_error = None
    for kw in attempt_kwargs:
        try:
            resp = openai.audio.speech.create(**kw)
            audio_bytes = _speech_response_bytes(resp)
            if audio_bytes:
                return audio_bytes
        except TypeError as ex:
            last_error = ex
            continue
        except Exception as ex:
            last_error = ex
            break
    if last_error:
        raise RuntimeError(f"TTS synthesis failed: {last_error}")
    raise RuntimeError("TTS synthesis failed: empty audio response")


def _play_wav_bytes(audio_bytes: bytes):
    if not audio_bytes.startswith(b"RIFF"):
        raise ValueError(
            "OpenAI TTS did not return WAV bytes. Set OPENAI_TTS_FORMAT=wav."
        )
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        pcm_bytes = wav_file.readframes(frame_count)
    if sample_width != 2:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")
    frames = np.frombuffer(pcm_bytes, dtype=np.int16)
    frames = frames.reshape(-1, channels if channels > 1 else 1)
    sd.play(frames, samplerate=sample_rate, blocking=True)


async def tts_openai_stream(text_iter):
    parts = []
    async for piece in text_iter:
        piece = (piece or "").strip()
        if piece:
            parts.append(piece)
    text = " ".join(parts).strip()
    if not text:
        print("[TTS] Empty text; skipping synthesis.")
        return
    print(f"[TTS] Synthesizing {len(text)} chars with OpenAI TTS...")
    async with _TTS_LOCK:
        audio_bytes = await asyncio.to_thread(_create_speech_bytes, text)
        await asyncio.to_thread(_play_wav_bytes, audio_bytes)
    print(f"[TTS] Played {len(audio_bytes)} bytes.")


async def openai_sentence_stream(
    messages,
    model=st.CHAT_MODEL,
    max_tokens=120,
    temperature=0.65,
    first_min_chars=25,
    first_max_chars=90,
    rest_min_chars=80,
    rest_max_chars=180,
):
    buf = ""
    first_done = False
    stream = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
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
        if len(buf) >= min_chars and (st.SENT_END.search(buf) or len(buf) >= max_chars):
            chunk = buf.strip()
            if not re.search(r"[\.!\?…]$", chunk):
                chunk += "."
            yield chunk
            buf = ""
            first_done = True
    tail = buf.strip()
    if tail:
        if not re.search(r"[\.!\?…]$", tail):
            tail += "."
        yield tail


async def call_ai_and_tts(meta: dict):
    event_type = meta.get("event")
    player_id = meta.get("player")
    max_tokens, max_chars = decide_speech_budget(event_type, player_id)
    target_tokens = int((max_chars / 4.0) * 1.35)
    model_max_tokens = max(96, min(max_tokens, target_tokens))

    context_event = mem.compact_for_context(meta.get("context_event", {}))
    print(f"[AI] Processing context_event: {context_event}")
    tilt_value = st.TILT_STATE.get(player_id, 0)

    player_context = json.dumps(context_event, indent=2, ensure_ascii=False)
    player_context = mem.limit_text(player_context, st.MAX_CONTEXT_CHARS)
    compact_meta = mem.compact_for_context(meta)
    current = json.dumps(compact_meta, indent=2, ensure_ascii=False)
    current = mem.limit_text(current, st.MAX_CONTEXT_CHARS)

    memory_summary = mem.summarize_memory()
    system_prompt = (
        "You are a sarcastic, very annoyed Minecraft coach who adjusts tone based on context: annoyed/fake-supportive for good or routine events, furious/angry for repeated mistakes or harm.\n"
        "Always say 'you' (never the username of the player).\n"
        "Profanity allowed (FUCK, SHIT, DAMN).\n"
        "Use ALL-CAPS and !!! for emphasis.\n"
        "First sentence must be short (<= 12 words). Then continue.\n"
    )
    system_prompt += (
        f"Current rage level (0 calm - {st.MAX_TILT} max rage): {tilt_value}. "
        "Adjust your tone accordingly.\n"
    )
    system_prompt += (
        f"Keep it concise: about {max_chars} characters total, ending cleanly.\n"
    )

    user_prompt = (
        f"MEMORY SUMMARY:\n{memory_summary}\n\n"
        f"NEW EVENT: {current}\n\n"
        f"CURRENT PLAYER CONTEXT:\n{player_context}\n\n"
    )
    content_parts = [{"type": "text", "text": user_prompt}]
    if meta.get("screenshot_url"):
        content_parts.append(
            {"type": "image_url", "image_url": {"url": meta["screenshot_url"]}}
        )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content_parts},
    ]

    print("[AI] Streaming response...")

    async def _text_iter():
        base_iter = openai_sentence_stream(
            messages,
            model=st.CHAT_MODEL,
            max_tokens=model_max_tokens,
            temperature=0.65,
            first_min_chars=20,
            first_max_chars=85,
            rest_min_chars=90,
            rest_max_chars=180,
        )
        async for piece in soft_cap_stream(
            base_iter, hard_cap_chars=max_chars, overflow_chars=160
        ):
            print(f"[AI] piece (+{len(piece)} chars): {piece}")
            yield piece

    await tts_openai_stream(_text_iter())
    print("[TTS] Streamed audio (OpenAI).")
    key = meta.get("screenshot_s3_key")
    if key:
        delete_s3_object(key)


async def worker_loop():
    while True:
        meta = await st.comment_queue.get()
        print(f"[TTS Worker] Got meta: {meta}")
        try:
            await call_ai_and_tts(meta)
        except Exception as ex:
            print(f"[TTS Worker] Error: {ex}")
        st.comment_queue.task_done()

