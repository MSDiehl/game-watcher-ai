import asyncio
import random
import time
from collections import Counter, defaultdict, deque

import gw_memory as mem
import gw_state as st
from gw_media import capture_screenshot_to_s3_url

SPEAK_STATE = defaultdict(lambda: {"last_spoke_global": 0.0})
ITEM_STATE = defaultdict(
    lambda: {"last_spoke": 0.0, "rolling": deque(), "total_since_spoke": 0}
)

ALWAYS_NOTABLE = {
    "death",
    "hit_and_die",
    "struggling",
    "advancement",
    "health_low",
    "friendly_fire",
    "boss_spawn",
    "player_chat",
}

SPEAK_MIN_GAP = 12.0
ITEM_MIN_GAP = 30.0
ROLLING_WINDOW = 120.0
BIG_BATCH = 32
VALUE_POP = 8


def reset_heartbeat_timer():
    if not st.heartbeat_reset.is_set():
        st.heartbeat_reset.set()


async def enqueue_meta(meta: dict):
    print(f"[Queue] Enqueuing metadata: {meta}")

    if meta.get("event") != "heartbeat":
        reset_heartbeat_timer()

    try:
        st.comment_queue.put_nowait(meta)
    except asyncio.QueueFull:
        if meta.get("event") in ALWAYS_NOTABLE:
            await st.comment_queue.put(meta)
        else:
            print(
                f"[Queue] Dropped non-critical meta because queue is full: {meta.get('event')}"
            )


async def _emit_meta(meta: dict):
    await maybe_enqueue(meta)
    mem.record_memory(meta.get("event", "meta"), **meta)


def on_health_low(idx: int, ev: dict):
    if ev.get("type") != "health_low":
        return None
    player = ev.get("player")
    hp = ev.get("current")
    if not player:
        return None
    task = st.HEALTH_LOW_PENDING.pop(player, None)
    if task and not task.done():
        task.cancel()

    async def _fire():
        try:
            await asyncio.sleep(1.0)
            meta = {"event": "health_low", "health": hp}
            await _emit_meta(meta)
        except asyncio.CancelledError:
            pass

    st.HEALTH_LOW_PENDING[player] = asyncio.create_task(_fire())
    return None


def cancel_health_low_for(player: str):
    task = st.HEALTH_LOW_PENDING.pop(player, None)
    if task and not task.done():
        task.cancel()


def on_entity_kill(idx: int, ev: dict):
    if ev.get("type") != "entity_kill":
        return None
    return {"event": "entity_kill", "entity_type": ev.get("entity_type")}


def on_smelt_complete(idx: int, ev: dict):
    if ev.get("type") != "smelt_complete":
        return None
    player = ev.get("player")
    item = ev.get("item")
    count = int(ev.get("count", 1))
    if not player or not item:
        return None
    key = (player, item)
    batch_state = st.SMELT_BATCH.get(
        key,
        {
            "count": 0,
            "task": None,
            "idle": st.event_idle(item, "smelt", st.BATCH_IDLE_SECONDS),
        },
    )
    batch_state["count"] += count
    idle = batch_state["idle"]
    if batch_state.get("task") and not batch_state["task"].done():
        batch_state["task"].cancel()

    async def _flush():
        try:
            await asyncio.sleep(idle)
            total = batch_state["count"]
            value = total * st.item_value(item)
            meta = {
                "event": "smelt_batch",
                "item": item,
                "item_name": st.item_nice(item),
                "count": total,
                "value": value,
            }
            st.SMELT_BATCH.pop(key, None)
            await _emit_meta(meta)
        except asyncio.CancelledError:
            pass

    batch_state["task"] = asyncio.create_task(_flush())
    st.SMELT_BATCH[key] = batch_state
    return None


def on_potion_brew(idx: int, ev: dict):
    if ev.get("type") != "potion_brew":
        return None
    dimension = ev.get("dimension")
    location = ev.get("location")
    if not dimension or not location:
        return None
    key = (dimension, location)
    batch_state = st.BREW_BATCH.get(key)
    if not batch_state:
        batch_state = {"by_output": {}, "ingredient": None, "task": None}

    ingredient = ev.get("ingredient_item")
    if ingredient:
        batch_state["ingredient"] = ingredient

    for bottle in ev.get("result_bottles", []):
        if bottle.get("empty"):
            continue
        item = bottle.get("item")
        if not item:
            continue
        potion_id = bottle.get("potion_id")
        color = bottle.get("custom_color")
        effects = bottle.get("custom_effects")
        if effects:
            signature = tuple(
                sorted(
                    (e.get("id"), int(e.get("amplifier", 0)), int(e.get("duration", 0)))
                    for e in effects
                )
            )
        else:
            signature = None
        output_key = (item, potion_id, color, signature)
        batch_state["by_output"][output_key] = (
            batch_state["by_output"].get(output_key, 0) + 1
        )

    if batch_state.get("task") and not batch_state["task"].done():
        batch_state["task"].cancel()

    async def _flush():
        try:
            await asyncio.sleep(st.BATCH_IDLE_SECONDS)
            by_output = batch_state["by_output"]
            results = []
            total = 0
            for (item, potion_id, color, signature), cnt in by_output.items():
                entry = {"item": item, "count": cnt, "item_name": st.item_nice(item)}
                if potion_id:
                    entry["potion_id"] = potion_id
                if color is not None:
                    entry["custom_color"] = color
                if signature:
                    entry["custom_effects"] = [
                        {"id": eid, "amplifier": amp, "duration": dur}
                        for (eid, amp, dur) in signature
                    ]
                results.append(entry)
                total += cnt
            meta = {
                "event": "brew_batch",
                "dimension": key[0],
                "location": key[1],
                "ingredient_item": batch_state.get("ingredient"),
                "results": results,
                "total_bottles": total,
            }
            st.BREW_BATCH.pop(key, None)
            await _emit_meta(meta)
        except asyncio.CancelledError:
            pass

    batch_state["task"] = asyncio.create_task(_flush())
    st.BREW_BATCH[key] = batch_state
    return None


def on_villager_trade(idx: int, ev: dict):
    if ev.get("type") != "villager_trade":
        return None
    player = ev.get("player")
    dimension = ev.get("dimension")
    villager_pos = ev.get("villager_pos")
    if not player or not dimension or not villager_pos:
        return None

    profession = ev.get("villager_profession")
    level = ev.get("villager_level")
    key = (player, dimension, villager_pos)
    batch_state = st.TRADE_BATCH.get(
        key,
        {"trades": [], "task": None, "profession": profession, "level": level},
    )
    trade = {
        "sell": {"item": ev.get("sell_item"), "count": int(ev.get("sell_count", 1))},
        "buy": [],
        "uses": ev.get("uses"),
        "max_uses": ev.get("max_uses"),
        "merchant_experience": ev.get("merchant_experience"),
        "special_price": ev.get("special_price"),
        "demand_bonus": ev.get("demand_bonus"),
        "price_multiplier": ev.get("price_multiplier"),
        "out_of_stock": bool(ev.get("out_of_stock")),
    }
    buy_a_item = ev.get("buy_a_item")
    if buy_a_item:
        trade["buy"].append({"item": buy_a_item, "count": int(ev.get("buy_a_count", 0))})
    buy_b_item = ev.get("buy_b_item")
    if buy_b_item:
        trade["buy"].append({"item": buy_b_item, "count": int(ev.get("buy_b_count", 0))})

    batch_state["trades"].append(trade)
    if batch_state.get("task") and not batch_state["task"].done():
        batch_state["task"].cancel()
    trade["sell"]["item_name"] = st.item_nice(trade["sell"]["item"])
    trade["sell"]["value"] = st.item_value(trade["sell"]["item"]) * int(
        trade["sell"]["count"]
    )

    async def _flush():
        try:
            await asyncio.sleep(st.SHORT_IDLE_SECONDS)
            trades = batch_state["trades"]
            total_trades = len(trades)
            summary = {}
            emeralds_spent = 0
            for trade_entry in trades:
                sell_item = trade_entry["sell"]["item"]
                sell_count = int(trade_entry["sell"]["count"])
                if sell_item:
                    summary[sell_item] = summary.get(sell_item, 0) + sell_count
                for buy in trade_entry["buy"]:
                    buy["item_name"] = st.item_nice(buy["item"])
                    if buy.get("item") == "minecraft:emerald":
                        emeralds_spent += int(buy.get("count", 0))
            meta = {
                "event": "trade_batch",
                "dimension": dimension,
                "villager_pos": villager_pos,
                "villager_profession": batch_state.get("profession"),
                "villager_level": batch_state.get("level"),
                "total_trades": total_trades,
                "emeralds_spent": emeralds_spent,
                "summary": summary,
                "trades": trades,
            }
            st.TRADE_BATCH.pop(key, None)
            await _emit_meta(meta)
        except asyncio.CancelledError:
            pass

    batch_state["task"] = asyncio.create_task(_flush())
    st.TRADE_BATCH[key] = batch_state
    return None


def on_item_drop(idx: int, ev: dict):
    if ev.get("type") != "item_drop":
        return None
    player = ev.get("player")
    item = ev.get("item")
    count = int(ev.get("count", 1))
    if not player or not item:
        return None

    death_fallout = st.DEATH_FALLOUT.get(player)
    if death_fallout and death_fallout.get("task") and not death_fallout["task"].done():
        death_fallout["items"][item] += count
        return None

    key = (player, item)
    batch_state = st.DROP_BATCH.get(
        key,
        {
            "count": 0,
            "task": None,
            "idle": st.event_idle(item, "drop", st.SHORT_IDLE_SECONDS),
        },
    )
    batch_state["count"] += count
    idle = batch_state["idle"]
    if batch_state.get("task") and not batch_state["task"].done():
        batch_state["task"].cancel()

    async def _flush():
        try:
            await asyncio.sleep(idle)
            total = batch_state["count"]
            st.DROP_BATCH.pop(key, None)
            meta = {
                "event": "drop_batch",
                "player": player,
                "item": item,
                "item_name": st.item_nice(item),
                "count": total,
                "value": total * st.item_value(item),
            }
            await _emit_meta(meta)
        except asyncio.CancelledError:
            pass

    batch_state["task"] = asyncio.create_task(_flush())
    st.DROP_BATCH[key] = batch_state
    return None


def on_item_craft(idx: int, ev: dict):
    if ev.get("type") != "item_craft":
        return None
    player = ev.get("player")
    item = ev.get("item")
    count = int(ev.get("count", 1))
    if not player or not item or count <= 0:
        return None

    rollup = st.CRAFT_ROLLUP[player]
    rollup["items"][item] += count
    if item in st.ITEMS_CONFIG:
        rollup["value"] += st.item_value(item) * count
    idle = rollup.get("idle", st.SHORT_IDLE_SECONDS)
    if rollup.get("task") and not rollup["task"].done():
        rollup["task"].cancel()

    async def _flush():
        try:
            await asyncio.sleep(idle)
            items = rollup["items"]
            total_items = sum(items.values())
            total_value = rollup["value"]
            summary = []
            for item_id, cnt in items.items():
                if item_id in st.ITEMS_CONFIG:
                    summary.append(
                        {
                            "item": item_id,
                            "item_name": st.item_nice(item_id),
                            "count": int(cnt),
                            "value": int(st.item_value(item_id) * cnt),
                            "category": st.ITEMS_CONFIG[item_id].get("category"),
                        }
                    )
            st.CRAFT_ROLLUP.pop(player, None)
            if not summary:
                return
            meta = {
                "event": "craft_rollup",
                "items": sorted(summary, key=lambda x: (-(x["value"]), x["item"])),
                "total_items": int(total_items),
                "total_value": int(total_value),
            }
            await _emit_meta(meta)
        except asyncio.CancelledError:
            pass

    rollup["task"] = asyncio.create_task(_flush())
    st.CRAFT_ROLLUP[player] = rollup
    return None


def on_effect_apply(idx: int, ev: dict):
    if ev.get("type") != "effect_apply":
        return None
    player = ev.get("player")
    effect_id = ev.get("effect_id")
    amplifier = ev.get("amplifier")
    duration = ev.get("duration")
    if not player or not effect_id:
        return None
    key = (player, effect_id)
    task = st.EFFECT_PENDING.pop(key, None)
    if task and not task.done():
        task.cancel()

    async def _fire():
        try:
            await asyncio.sleep(0.75)
            meta = {
                "event": "effect_apply",
                "effect_id": effect_id,
                "amplifier": amplifier,
                "duration": duration,
            }
            await _emit_meta(meta)
        except asyncio.CancelledError:
            pass

    st.EFFECT_PENDING[key] = asyncio.create_task(_fire())
    return None


def on_advancement(idx: int, ev: dict):
    if ev.get("type") != "advancement":
        return None
    return {
        "event": "advancement",
        "player": ev.get("player"),
        "id": ev.get("id"),
        "title": ev.get("advancement"),
        "desc": ev.get("description"),
    }


def on_chat_message(idx: int, ev: dict):
    if ev.get("type") != "chat_message":
        return None
    player = ev.get("player", "unknown")
    message = (ev.get("message") or "").strip()
    now = time.time()
    if now - st.LAST_CHAT_REPLY[player] < st.CHAT_COOLDOWN_SECONDS:
        return None
    st.LAST_CHAT_REPLY[player] = now
    return {"event": "player_chat", "message": message}


def on_player_hurt(idx: int, ev: dict):
    if ev.get("type") != "player_hurt":
        return None
    src = ev.get("damage_type", "unknown")
    st.HIT_SESSIONS[src].record_hit(time.time())
    st.HIT_TOTALS[src] += 1
    total = st.HIT_TOTALS[src]
    if total >= st.NEXT_ALERT[src]:
        meta = {
            "event": "struggling",
            "attacker": src,
            "hits": total,
            "session_hits": st.HIT_SESSIONS[src].count,
        }
        st.NEXT_ALERT[src] *= 2
        return meta
    return None


def on_player_death(idx: int, ev: dict):
    if ev.get("type") != "player_death":
        return None
    cause = ev.get("cause", "unknown")
    player_id = ev.get("player")
    if player_id:
        cancel_health_low_for(player_id)
    session = st.HIT_SESSIONS.get(cause)
    attacker_stats = st.ATTACKER_STATS[cause]
    sessions = attacker_stats["sessions"]
    total_hits = attacker_stats["total_hits"]
    if sessions > 0:
        avg = total_hits / sessions
        threshold = max(st.INITIAL_HIT_ALERT, int(avg * st.THRESHOLD_FACTOR))
    else:
        threshold = st.INITIAL_HIT_ALERT

    if session:
        attacker_stats["sessions"] += 1
        attacker_stats["total_hits"] += session.count
        del st.HIT_SESSIONS[cause]
        hit_and_die = session.count >= threshold
    else:
        hit_and_die = False

    st.death_counters[cause] += 1
    if player_id:
        st.adjust_tilt(player_id, +2)
    death_key = player_id or "unknown"
    fallout = st.DEATH_FALLOUT[death_key]
    fallout["items"] = Counter()
    fallout["event"] = "hit_and_die" if hit_and_die else "death"
    fallout["cause"] = cause
    fallout["count"] = st.death_counters[cause]
    if fallout.get("task") and not fallout["task"].done():
        fallout["task"].cancel()
    for (player, item), drop_state in list(st.DROP_BATCH.items()):
        if player == player_id:
            fallout["items"][item] += int(drop_state.get("count", 0))
            if drop_state.get("task") and not drop_state["task"].done():
                drop_state["task"].cancel()
            st.DROP_BATCH.pop((player, item), None)

    async def _flush_death():
        try:
            await asyncio.sleep(fallout["idle"])
            drops = []
            total_value = 0
            for item_id, cnt in fallout["items"].items():
                val = st.item_value(item_id) * cnt
                total_value += val
                drops.append(
                    {
                        "item": item_id,
                        "item_name": st.item_nice(item_id),
                        "count": int(cnt),
                        "value": int(val),
                    }
                )
            meta = {
                "event": fallout["event"],
                "cause": fallout["cause"],
                "count": fallout["count"],
            }
            if drops:
                meta["dropped_items"] = sorted(
                    drops, key=lambda x: (-x["value"], x["item"])
                )
                meta["dropped_total_value"] = int(total_value)
            await _emit_meta(meta)
        except asyncio.CancelledError:
            pass
        finally:
            st.DEATH_FALLOUT.pop(death_key, None)

    fallout["task"] = asyncio.create_task(_flush_death())
    return None


def on_block_break(idx: int, ev: dict):
    if ev.get("type") != "block_break":
        return None
    block_id = ev.get("block", "")
    if not block_id.endswith("_ore"):
        return None
    ore = block_id[:-4]
    st.SESSIONS[ore].record_break(idx, time.time())
    return None


def on_item_pickup(idx: int, ev: dict):
    if ev.get("type") != "item_pickup":
        return None
    ore = ev["item"]
    cnt = ev.get("count", 1)
    session = st.SESSIONS.get(ore)
    if not session or not session.pending:
        return None
    session.record_pickup(cnt, time.time())
    return None


def on_special_item_pickup(idx: int, ev: dict):
    if ev.get("type") != "item_pickup":
        return None
    item = ev["item"]
    cfg = st.ITEMS_CONFIG.get(item)
    if not cfg:
        return None
    cnt = ev.get("count", 1)
    st.SPECIAL_PICKUP_STATE[item] += cnt
    threshold = int(cfg.get("threshold", 0))
    if threshold <= 0 or st.SPECIAL_PICKUP_STATE[item] < threshold:
        return None
    st.SPECIAL_PICKUP_STATE[item] -= threshold
    return {
        "event": "item_collected",
        "item": item,
        "item_name": st.item_nice(item),
        "category": cfg.get("category"),
        "threshold": threshold,
    }


def on_friendly_hit(idx: int, ev: dict):
    if ev.get("type") != "entity_hit":
        return None
    entity = ev.get("entity_type", "")
    if entity not in st.FRIENDLY_ANIMALS:
        return None
    state = st.FRIENDLY_HIT_STATE[entity]
    state["count"] += 1
    print(f"[Detect] {entity} hit count: {state['count']}/{state['threshold']}")
    if state["count"] < state["threshold"]:
        return None
    st.FRIENDLY_HIT_STATE[entity] = {"threshold": random.randint(30, 50), "count": 0}
    return {"event": "friendly_fire", "entity": entity}


def _now():
    return time.time()


def _purge_old(dq: deque, now: float, window: float = ROLLING_WINDOW):
    while dq and (now - dq[0][0]) > window:
        dq.popleft()


def _item_key(meta: dict) -> str:
    event = meta.get("event", "")
    item = (
        meta.get("item")
        or meta.get("ore")
        or meta.get("effect_id")
        or meta.get("attacker")
        or ""
    )
    return f"{event}:{item}"


def _category_boost(item_id: str) -> int:
    cfg = st.ITEMS_CONFIG.get(item_id, {})
    cat = cfg.get("category", "")
    return {
        "rare": 8,
        "quest": 6,
        "combat": 5,
        "valuable": 4,
        "mobility": 3,
        "utility": 2,
        "ammo": 2,
        "food": 1,
        "trash": 0,
    }.get(cat, 1)


def _score(meta: dict, now: float) -> float:
    event = meta.get("event", "")
    if event in ALWAYS_NOTABLE:
        return 999.0
    last_global = SPEAK_STATE["*"]["last_spoke_global"]
    global_penalty = 0.0 if (now - last_global) >= SPEAK_MIN_GAP else -1.0
    score = 0.0 + global_penalty
    if event in ("smelt_batch", "drop_batch"):
        item = meta.get("item")
        if not item:
            return -999.0
        count = int(
            meta.get("count", meta.get("total_bottles", meta.get("total_trades", 1)))
        )
        value = int(meta.get("value", count * st.item_value(item) if item else 0))
        key = _item_key(meta)
        item_state = ITEM_STATE[key]
        _purge_old(item_state["rolling"], now)
        item_state["rolling"].append((now, count))
        item_state["total_since_spoke"] += count
        score += _category_boost(item)
        if value >= VALUE_POP:
            score += 1.0
        threshold = st.ITEMS_CONFIG.get(item, {}).get("threshold", BIG_BATCH)
        if item_state["total_since_spoke"] >= threshold:
            score += 2.0
        if (now - item_state["last_spoke"]) >= ITEM_MIN_GAP:
            score += 0.6
        if item_state["rolling"]:
            recent_total = sum(c for _, c in item_state["rolling"])
            recent_avg = recent_total / max(1, len(item_state["rolling"]))
            if count >= max(8, 2 * recent_avg):
                score += 1.2
        if (now - item_state["last_spoke"]) < 10.0:
            score -= 1.5
        if st.ITEMS_CONFIG.get(item, {}).get("category") in ("trash",):
            if item_state["total_since_spoke"] < (2 * threshold):
                score -= 0.8
    elif event == "trade_batch":
        total_trades = int(meta.get("total_trades", 0))
        emeralds_spent = int(meta.get("emeralds_spent", 0))
        if total_trades <= 0:
            return -999.0
        score += 0.8
        if total_trades >= 3:
            score += 0.7
        if emeralds_spent >= 24:
            score += 0.8
    elif event == "brew_batch":
        total_bottles = int(meta.get("total_bottles", 0))
        if total_bottles <= 0:
            return -999.0
        score += 0.6
        if total_bottles >= 3:
            score += 0.6
        if total_bottles >= 6:
            score += 0.6
    elif event == "craft_rollup":
        items = meta.get("items", [])
        if not items:
            return -999.0
        total_value = int(meta.get("total_value", 0))
        max_cat_boost = 0
        threshold_hit = False
        notable_rare = False
        for entry in items:
            item_id = entry.get("item")
            cnt = int(entry.get("count", 0))
            cfg = st.ITEMS_CONFIG.get(item_id)
            if not cfg:
                continue
            max_cat_boost = max(max_cat_boost, _category_boost(item_id))
            threshold = int(cfg.get("threshold", BIG_BATCH))
            if cnt >= threshold:
                threshold_hit = True
            if cfg.get("category") in ("rare", "quest"):
                notable_rare = True
        if max_cat_boost >= 5:
            score += 0.8
        if total_value >= (VALUE_POP * 2):
            score += 0.8
        if threshold_hit:
            score += 1.4
        if notable_rare:
            score += 1.0
        key = "craft_rollup"
        item_state = ITEM_STATE[key]
        if (now - item_state["last_spoke"]) >= ITEM_MIN_GAP:
            score += 0.6
        else:
            score -= 1.2
        if (now - SPEAK_STATE["*"]["last_spoke_global"]) < SPEAK_MIN_GAP:
            score -= 1.0
    elif event == "effect_apply":
        amp = int(meta.get("amplifier", 0))
        dur = int(meta.get("duration", 0))
        if amp >= 1 or dur >= 45 * 20:
            score += 1.2
    elif event == "entity_kill":
        score += 0.5
    return score


def _should_capture_screenshot(meta: dict) -> bool:
    return meta.get("event") in st.SCREENSHOT_EVENTS


def _enrich_meta_for_ai(meta: dict) -> dict:
    enriched = dict(meta)
    if "context_event" not in enriched and st.LAST_PLAYER_STATE is not None:
        enriched["context_event"] = mem.compact_for_context(st.LAST_PLAYER_STATE)
    elif "context_event" in enriched:
        enriched["context_event"] = mem.compact_for_context(enriched["context_event"])
    if _should_capture_screenshot(enriched):
        url, key = capture_screenshot_to_s3_url()
        if url and key:
            enriched["screenshot_url"] = url
            enriched["screenshot_s3_key"] = key
    return enriched


async def maybe_enqueue(meta: dict) -> bool:
    now = _now()
    key = _item_key(meta)
    score = _score(meta, now)
    if score >= 1.0:
        SPEAK_STATE["*"]["last_spoke_global"] = now
        if key:
            ITEM_STATE[key]["last_spoke"] = now
            ITEM_STATE[key]["total_since_spoke"] = 0
        await enqueue_meta(_enrich_meta_for_ai(meta))
        return True
    return False


EVENT_HANDLERS = (
    on_player_death,
    on_entity_kill,
    on_health_low,
    on_villager_trade,
    on_smelt_complete,
    on_item_drop,
    on_effect_apply,
    on_item_craft,
    on_potion_brew,
    on_advancement,
    on_block_break,
    on_item_pickup,
    on_friendly_hit,
    on_player_hurt,
    on_chat_message,
    on_special_item_pickup,
)


async def process_event(idx: int, ts: float, ev: dict):
    for handler in EVENT_HANDLERS:
        meta = handler(idx, ev)
        if meta:
            print(f"[Analyzer] {handler.__name__} emitted meta: {meta}")
            await maybe_enqueue(meta)
            mem.record_memory(handler.__name__, **meta)


async def session_flusher():
    while True:
        now = time.time()
        for ore, session in list(st.SESSIONS.items()):
            if session.is_expired(now):
                matched = session.matched
                session.pending.clear()
                session.matched = 0
                session.due_ts = 0.0
                if matched >= st.MIN_VEIN_SEQ:
                    meta = {"event": "vein_mine", "ore": ore, "blocks": matched}
                elif matched > 0:
                    meta = {"event": "pickup", "item": ore, "count": matched}
                else:
                    del st.SESSIONS[ore]
                    continue
                print(f"[Analyzer] session end -> {meta}")
                await maybe_enqueue(meta)
                mem.record_memory(meta["event"], **meta)
                del st.SESSIONS[ore]

        for attacker, session in list(st.HIT_SESSIONS.items()):
            if session.is_expired(now):
                stats = st.ATTACKER_STATS[attacker]
                stats["sessions"] += 1
                stats["total_hits"] += session.count
                del st.HIT_SESSIONS[attacker]
                sessions = stats["sessions"]
                total_hits = stats["total_hits"]
                avg = total_hits / sessions if sessions > 0 else 0
                threshold = max(st.INITIAL_HIT_ALERT, int(avg * st.THRESHOLD_FACTOR))
                if session.count >= threshold:
                    meta = {
                        "event": "struggling",
                        "attacker": attacker,
                        "hits": session.count,
                    }
                    print(f"[Analyzer] hit-session end -> {meta}")
                    await maybe_enqueue(meta)
                    mem.record_memory("struggling", **meta)
        await asyncio.sleep(0.5)


async def heartbeat_easter_eggs():
    while True:
        wait_time = random.randint(st.HEARTBEAT_MIN, st.HEARTBEAT_MAX)
        st.heartbeat_reset.clear()
        sleep_task = asyncio.create_task(asyncio.sleep(wait_time))
        reset_task = asyncio.create_task(st.heartbeat_reset.wait())
        done, pending = await asyncio.wait(
            {sleep_task, reset_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        if reset_task in done and st.heartbeat_reset.is_set():
            print("[Heartbeat] Timer reset due to recent AI event.")
            continue

        meta = {"event": "heartbeat"}
        choice = random.choice(["memory", "performance", "random_tip"])
        if choice == "memory" and len(st.MEMORY) > 0:
            event = random.choice(list(st.MEMORY))
            meta["type: Memory"] = f"Remember when {event.type} happened? {event.details}"
        elif choice == "performance":
            meta["type: Review"] = "Performance review: " + mem.summarize_memory(limit=3)
        else:
            meta["type: Random Words Depending on Context"] = random.choice(
                [
                    "Hey, ever thought about NOT dying?",
                    "Pro tip: Lava is hot.",
                    "Do you even know how to swing that pickaxe?",
                    "This is SOOOO boring to watch!",
                    "You know you can place blocks, right?",
                    "I could do this better with my eyes closed.",
                ]
            )
        meta["context_event"] = mem.compact_for_context(st.LAST_PLAYER_STATE)
        await enqueue_meta(meta)

