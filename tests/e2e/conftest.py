"""End-to-end setup: start the real Streamlit app on a free port, in its own temporary cache.

Offline tier (default): AIRPORT_AGENT_CLAUDE=fake — the scripted stand-in for Claude, free and repeatable.
Live tier: set RUN_LIVE_E2E=1 to also run the tests marked `live` against the real Claude (costs a few cents).
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
STREAMLIT = ROOT / ".venv" / "bin" / "streamlit"
REAL_OAG_CACHE = ROOT / "data" / "cache" / "oag"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start(tmp: Path, fake: bool) -> tuple[subprocess.Popen, str]:
    cache = tmp / "cache"
    (cache / "oag").mkdir(parents=True, exist_ok=True)
    for f in REAL_OAG_CACHE.glob("*.json"):            # cached OAG days: no OAG calls during tests
        shutil.copy(f, cache / "oag" / f.name)
    port = _free_port()
    env = {**os.environ, "AIRPORT_AGENT_CACHE_DIR": str(cache)}
    env.pop("ANTHROPIC_BASE_URL", None)
    if fake:
        env["AIRPORT_AGENT_CLAUDE"] = "fake"
    else:
        env.pop("AIRPORT_AGENT_CLAUDE", None)
    proc = subprocess.Popen(
        [str(STREAMLIT), "run", "src/airport_agent/ui/app.py", "--server.port", str(port),
         "--server.headless", "true", "--browser.gatherUsageStats", "false"],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    for _ in range(60):
        try:
            if httpx.get(f"{url}/_stcore/health", timeout=1).status_code == 200:
                return proc, url
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    proc.kill()
    raise RuntimeError("the Streamlit app did not start")


@pytest.fixture(scope="session")
def app_url(tmp_path_factory):
    if not STREAMLIT.exists() or not (ROOT / "data/reference/signal_reference.parquet").exists():
        pytest.skip("app environment or processed data missing")
    proc, url = _start(tmp_path_factory.mktemp("e2e_fake"), fake=True)
    yield url
    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture(scope="session")
def live_app_url(tmp_path_factory):
    if os.environ.get("RUN_LIVE_E2E") != "1":
        pytest.skip("live e2e tests run only with RUN_LIVE_E2E=1 (they call the real Claude)")
    sys.path.insert(0, str(ROOT / "src"))
    from airport_agent.settings import Settings
    if Settings().anthropic_api_key is None:
        pytest.skip("ANTHROPIC_API_KEY is not set")
    proc, url = _start(tmp_path_factory.mktemp("e2e_live"), fake=False)
    yield url
    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture(autouse=True)
def no_browser_console_errors(request):
    """Fail an e2e test if the page logs a JavaScript error while it runs."""
    if "page" not in request.fixturenames:
        yield
        return
    page = request.getfixturevalue("page")
    errors: list[str] = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))
    yield
    assert not errors, "browser console errors: " + " | ".join(errors[:5])
