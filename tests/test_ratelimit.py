import time

from app.ratelimit import RateLimiter


def test_blocks_after_limit_within_window():
    limiter = RateLimiter(limit=2, window_seconds=3600)
    assert limiter.allow("1.2.3.4")
    assert limiter.allow("1.2.3.4")
    assert not limiter.allow("1.2.3.4")
    assert limiter.allow("5.6.7.8")  # other keys unaffected


def test_window_expiry_frees_slots(monkeypatch):
    limiter = RateLimiter(limit=1, window_seconds=60)
    assert limiter.allow("ip")
    assert not limiter.allow("ip")

    monkeypatch.setattr("app.ratelimit.time.time", lambda: 10**10)  # far future
    assert limiter.allow("ip")


def test_expired_keys_are_forgotten(monkeypatch):
    # regression (#18): every key stayed in memory until a restart, and a
    # check alone added an empty entry
    clock = [1000.0]
    monkeypatch.setattr("app.ratelimit.time.time", lambda: clock[0])
    limiter = RateLimiter(limit=5, window_seconds=60)
    assert not limiter.is_blocked("never-sent")
    for n in range(100):
        limiter.allow(f"visitor-{n}")
    assert len(limiter._hits) == 100

    clock[0] += 61  # a window later; none of them came back
    limiter.allow("new-visitor")
    assert list(limiter._hits) == ["new-visitor"]
