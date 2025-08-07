# game-watcher-ai/hit_session.py
HIT_IDLE = 2.0  

# Track per‐attacker hit sessions
class HitSession:
    __slots__ = ("count","due_ts")
    def __init__(self):
        self.count  = 0
        self.due_ts = 0.0

    def record_hit(self, ts):
        self.count  += 1
        self.due_ts = ts + HIT_IDLE

    def is_expired(self, now):
        return now >= self.due_ts