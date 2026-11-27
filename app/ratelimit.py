"""Sliding-window rate limiter, in-memory per process.

Good enough for one server: counters reset on restart and aren't shared
across workers. If the deployment ever scales past one process, this is
the seam to swap for Redis.
"""
import time
from collections import defaultdict, deque
from threading import Lock


class RateLimiter:
    def __init__(self, limit: int, window_seconds: int = 3600):
        self._limit = limit
        self._window = window_seconds
        self._hits = defaultdict(deque)
        self._lock = Lock()
        self._swept = time.time()

    def allow(self, key: str) -> bool:
        """Check and record in one step — the normal request counter."""
        with self._lock:
            self._sweep()
            if self._over_limit(key):
                return False
            self._hits[key].append(time.time())
            return True

    def record(self, key: str):
        """Record without checking — for events noticed after the fact,
        like a scope-gate refusal."""
        with self._lock:
            self._sweep()
            self._hits[key].append(time.time())

    def release(self, key: str):
        """Give back the latest hit, for a request that failed on our side."""
        with self._lock:
            hits = self._hits.get(key)
            if hits:
                hits.pop()

    def is_blocked(self, key: str) -> bool:
        with self._lock:
            return self._over_limit(key)

    def _over_limit(self, key) -> bool:
        # .get, not [key]: a check alone used to leave an empty entry behind
        now = time.time()
        hits = self._hits.get(key, ())
        while hits and hits[0] <= now - self._window:
            hits.popleft()
        if not hits:
            self._hits.pop(key, None)
        return len(hits) >= self._limit

    def _sweep(self):
        """Once a window, forget every key whose hits have all expired.
        Every visitor used to stay in memory until a restart, the ones who
        never came back included."""
        now = time.time()
        if now - self._swept < self._window:
            return
        self._swept = now
        for key in [key for key, hits in self._hits.items()
                    if not hits or hits[-1] <= now - self._window]:
            del self._hits[key]
