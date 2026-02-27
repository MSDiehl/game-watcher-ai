import asyncio
import json
import time
import uuid
from collections import Counter

import gw_state as st


def limit_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def compact_for_context(value, *, depth: int = 0, max_depth: int = 3):
    """
    Reduce nested payload size for prompt/log safety while preserving useful structure.
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return limit_text(value, 240)
    if depth >= max_depth:
        return "<truncated>"
    if isinstance(value, list):
        clipped = value[: st.MAX_LOG_LIST_ITEMS]
        out = [
            compact_for_context(v, depth=depth + 1, max_depth=max_depth)
            for v in clipped
        ]
        if len(value) > st.MAX_LOG_LIST_ITEMS:
            out.append(f"...(+{len(value) - st.MAX_LOG_LIST_ITEMS} more)")
        return out
    if isinstance(value, dict):
        out = {}
        items = list(value.items())[: st.MAX_LOG_KEYS]
        for k, v in items:
            out[str(k)] = compact_for_context(v, depth=depth + 1, max_depth=max_depth)
        if len(value) > st.MAX_LOG_KEYS:
            out["..."] = f"+{len(value) - st.MAX_LOG_KEYS} more keys"
        return out
    return limit_text(str(value), 240)


def _sanitize_memory_details(details: dict) -> dict:
    skip = {
        "screenshot_b64",
        "screenshot_url",
        "screenshot_s3_key",
        "player_state_raw",
        "player_state",
        "context_event",
    }
    compacted = {}
    for k, v in details.items():
        if k in skip or v is None:
            continue
        compacted[k] = compact_for_context(v)
    return compacted


def _to_meta_primitive(val, max_len=2000):
    if isinstance(val, (str, int, float, bool)):
        return val
    if val is None:
        return "null"
    try:
        s = json.dumps(val, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        s = str(val)
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def _build_safe_metadata(details: dict, now: float) -> dict:
    skip_keys = {
        "screenshot_b64",
        "screenshot_url",
        "screenshot_s3_key",
        "player_state_raw",
        "player_state",
        "context_event",
    }
    md = {"timestamp": float(now)}
    for k, v in details.items():
        if k in skip_keys or v is None:
            continue
        md[k] = _to_meta_primitive(v)
    return md


def record_memory(ev_type: str, **details):
    now = time.time()
    mem_type = details.get("event", ev_type)
    compact_details = _sanitize_memory_details(details)
    text = f"{mem_type}: " + json.dumps(
        compact_details, separators=(",", ":"), ensure_ascii=False
    )
    text = limit_text(text, st.MAX_MEMORY_TEXT_CHARS)

    st.MEMORY.append(st.MemoryEvent(now, mem_type, details))
    cutoff = now - st.MEMORY_WINDOW
    while st.MEMORY and st.MEMORY[0].timestamp < cutoff:
        st.MEMORY.popleft()
    print(f"[Memory] Recorded {mem_type} {details}. Memory size = {len(st.MEMORY)}")

    asyncio.create_task(_embed_later(text, compact_details, now))


async def _embed_later(text: str, details: dict, now: float):
    try:
        safe_meta = _build_safe_metadata(details, now)
        embedding = await asyncio.to_thread(st.embedder.encode, text)
        embedding = embedding.tolist()
        await asyncio.to_thread(
            st.collection.add,
            ids=[f"{now:.3f}-{uuid.uuid4().hex[:8]}"],
            documents=[text],
            embeddings=[embedding],
            metadatas=[safe_meta],
        )
    except Exception as ex:
        print(f"[Memory] Embed failed: {ex}")


def summarize_memory(limit: int = 5) -> str:
    lines = []

    death_events = [ev for ev in st.MEMORY if ev.type in {"death", "hit_and_die"}]
    if death_events:
        total_deaths = len(death_events)
        causes = Counter(ev.details.get("cause", "unknown") for ev in death_events)
        parts = [f"{cnt} to {cause}" for cause, cnt in causes.items()]
        lines.append(f"You died {total_deaths} times ({', '.join(parts)}).")

    hurt_events = [ev for ev in st.MEMORY if ev.type == "struggling"]
    if hurt_events:
        total_hurts = len(hurt_events)
        attackers = Counter(ev.details.get("attacker", "unknown") for ev in hurt_events)
        parts = [f"{cnt} by {att}" for att, cnt in attackers.items()]
        lines.append(f"You took damage {total_hurts} times ({', '.join(parts)}).")

    friendly_events = [ev for ev in st.MEMORY if ev.type == "friendly_fire"]
    if friendly_events:
        entity_counts = Counter(
            ev.details.get("entity", "unknown") for ev in friendly_events
        )
        for entity, cnt in entity_counts.items():
            lines.append(f"You triggered friendly fire on {cnt}x {entity}.")

    if len(lines) > limit:
        lines = lines[-limit:]

    summary = (
        "\n".join(f"- {line}" for line in lines) if lines else "- Nothing memorable yet."
    )
    print(f"[Memory] Summary:\n{summary}")
    return summary

