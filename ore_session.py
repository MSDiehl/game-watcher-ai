from collections import deque

SESSION_IDLE = 1.0  

class OreSession:
    __slots__ = ("pending","matched","due_ts")
    def __init__(self):
        self.pending = deque()   # break indices not yet matched
        self.matched = 0         # total matched so far
        self.due_ts  = 0.0       # time.time() when this session expires

    def record_break(self, idx, ts):
        self.pending.append(idx)
        # DO NOT reset self.matched here!
        self.due_ts = ts + SESSION_IDLE

    def record_pickup(self, cnt, ts):
        for _ in range(min(cnt, len(self.pending))):
            self.pending.popleft()
            self.matched += 1
        self.due_ts = ts + SESSION_IDLE

    def is_expired(self, now):
        return now >= self.due_ts

    def finalize(self, ore):
        """
        Called once on expiry. Returns vein_mine meta
        and resets this session for reuse.
        """
        blocks = self.matched
        # reset for next session
        self.pending.clear()
        self.matched = 0
        self.due_ts  = 0.0
        return {"event": "vein_mine", "ore": ore, "blocks": blocks}