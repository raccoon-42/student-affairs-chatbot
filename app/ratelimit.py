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

    def allow(self, key: str) -> bool:
        """Check and record in one step — the normal request counter."""
        with self._lock:
            if self._over_limit(key):
                return False
            self._hits[key].append(time.time())
            return True

    def record(self, key: str):
        """Record without checking — for events noticed after the fact,
        like a scope-gate refusal."""
        with self._lock:
            self._hits[key].append(time.time())

    def is_blocked(self, key: str) -> bool:
        with self._lock:
            return self._over_limit(key)

    def _over_limit(self, key) -> bool:
        now = time.time()
        hits = self._hits[key]
        while hits and hits[0] <= now - self._window:
            hits.popleft()
        return len(hits) >= self._limit
