import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Every test gets a fresh SQLite file; storage reads the path per call."""
    monkeypatch.setattr(settings, "DB_PATH", str(tmp_path / "test.db"))
