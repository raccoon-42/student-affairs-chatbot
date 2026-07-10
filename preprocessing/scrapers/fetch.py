"""Polite fetching for iyte.edu.tr hosts: shared session, pacing, retry.

The university servers drop connections under rapid sequential requests —
seen on ceng.iyte.edu.tr (roster + profile pages) and on
ogrenciisleri.iyte.edu.tr (25 mevzuat PDFs), where a bare requests.get
times out mid-download partway through the list.
"""
import time

import requests

_session = requests.Session()


def fetch(url, timeout=30):
    for attempt in range(3):
        try:
            resp = _session.get(url, timeout=timeout)
            resp.raise_for_status()
            time.sleep(0.5)
            return resp
        except (requests.ConnectionError, requests.Timeout) as error:
            if attempt == 2:
                raise
            wait = 5 * (attempt + 1)
            print(f"  {error.__class__.__name__} on {url}, retrying in {wait}s...")
            time.sleep(wait)
