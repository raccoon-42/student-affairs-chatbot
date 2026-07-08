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
