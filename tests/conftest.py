"""Shared test setup: tests never write to the real cache (OAG cache, conversation memory, audit log)."""

import shutil

import pytest

from airport_agent import settings

REAL_CACHE = settings.get_settings().cache_dir


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    (cache / "oag").mkdir(parents=True)
    for f in (REAL_CACHE / "oag").glob("*.json"):     # read-only copy, so cached OAG days still need no calls
        shutil.copy(f, cache / "oag" / f.name)
    monkeypatch.setattr(settings.Settings, "cache_dir", property(lambda self: cache))
    yield cache
