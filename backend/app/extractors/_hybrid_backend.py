"""Lifecycle helper for the opendataloader-pdf-hybrid Docling-Fast server.

The hybrid engine talks to a separate process (`opendataloader-pdf-hybrid`)
that has to be listening on `HYBRID_URL` before any extraction request can
succeed. Forcing every user to start that subprocess by hand was the long-
standing reason the engine "didn't work" out of the box.

This module spawns it lazily on the first hybrid extraction, polls until it
is healthy, holds the handle for the rest of the process lifetime, and
tears it down at interpreter exit. Idempotent — if the backend is already
listening (e.g. the user started it manually) we just return.
"""
from __future__ import annotations

import asyncio
import atexit
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen

logger = logging.getLogger(__name__)

_lock = asyncio.Lock()
_process: Optional[subprocess.Popen] = None


def _is_healthy(url: str, timeout: float = 1.0) -> bool:
    """Cheap check that something is listening at `url`. The server returns
    404 on `/`, which still means the HTTP listener is up — that's enough.
    """
    try:
        with urlopen(url.rstrip("/") + "/", timeout=timeout) as resp:
            # Any 2xx/3xx response means the server is up and routing.
            return resp.status < 600
    except HTTPError:
        # 4xx/5xx — the server responded, it's listening. That's healthy enough
        # for our purposes (the Java client uses its own endpoint paths).
        return True
    except URLError:
        return False
    except (OSError, ConnectionError):
        return False


def _hybrid_cli_path() -> Optional[str]:
    """Locate the `opendataloader-pdf-hybrid` CLI.

    Checks PATH first, then falls back to the bin directory next to the
    running interpreter — handles the common case of running uvicorn via
    `.venv/bin/uvicorn` without activating the venv (so PATH doesn't
    include `.venv/bin`).
    """
    found = shutil.which("opendataloader-pdf-hybrid")
    if found:
        return found
    sibling = os.path.join(os.path.dirname(sys.executable), "opendataloader-pdf-hybrid")
    if os.path.isfile(sibling) and os.access(sibling, os.X_OK):
        return sibling
    return None


async def ensure_running(
    url: str,
    autostart: bool = True,
    startup_timeout: int = 120,
    poll_interval: float = 2.0,
) -> None:
    """Make sure something is listening at `url`.

    Fast path: if a backend is already healthy, return immediately. Otherwise,
    if `autostart` is True and the CLI is installed, spawn it as a subprocess
    and wait up to `startup_timeout` seconds for it to come up. The subprocess
    is registered with atexit so it dies with the parent process.
    """
    if _is_healthy(url):
        return

    if not autostart:
        raise RuntimeError(
            f"Hybrid backend not reachable at {url} and autostart is disabled. "
            "Start it with: opendataloader-pdf-hybrid --port "
            f"{_port_from_url(url)} --log-level warning"
        )

    cli = _hybrid_cli_path()
    if not cli:
        raise RuntimeError(
            "opendataloader-pdf-hybrid is not on PATH. Install with:\n"
            "  pip install 'opendataloader-pdf[hybrid]'"
        )

    async with _lock:
        # Re-check after acquiring the lock — another coroutine may have raced
        # us and already started the backend.
        if _is_healthy(url):
            return

        global _process
        if _process is not None and _process.poll() is None:
            # We already spawned it; just wait for it to become healthy.
            pass
        else:
            parsed = urlparse(url)
            host = parsed.hostname or "127.0.0.1"
            port = str(parsed.port or 5002)
            logger.info("Spawning opendataloader-pdf-hybrid on %s:%s", host, port)
            _process = subprocess.Popen(
                [cli, "--host", host, "--port", port, "--log-level", "warning"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                # New session so we can take down the whole subprocess group on exit.
                start_new_session=True,
            )

        # Poll for health; abort early if the process dies.
        deadline = time.time() + startup_timeout
        last_log = 0.0
        while time.time() < deadline:
            if _process.poll() is not None:
                code = _process.returncode
                _process = None
                raise RuntimeError(
                    f"opendataloader-pdf-hybrid exited prematurely with code {code}. "
                    "Check that the package is correctly installed: "
                    "pip install 'opendataloader-pdf[hybrid]'"
                )
            if _is_healthy(url):
                logger.info("opendataloader-pdf-hybrid is healthy at %s", url)
                return
            now = time.time()
            if now - last_log > 15:
                logger.info(
                    "Waiting for opendataloader-pdf-hybrid to come up "
                    "(loading docling models can take ~30s)..."
                )
                last_log = now
            await asyncio.sleep(poll_interval)

        # Timed out — leave the process alone (next request can retry) but
        # surface a clear error.
        raise RuntimeError(
            f"opendataloader-pdf-hybrid did not become healthy on {url} "
            f"within {startup_timeout}s. Try starting it manually to see "
            f"its log output: opendataloader-pdf-hybrid --port "
            f"{_port_from_url(url)}"
        )


def _port_from_url(url: str) -> str:
    parsed = urlparse(url)
    return str(parsed.port or 5002)


@atexit.register
def _cleanup() -> None:
    """Best-effort: terminate the spawned backend so it doesn't outlive us."""
    global _process
    proc = _process
    _process = None
    if proc is None or proc.poll() is not None:
        return
    try:
        # Take down the whole process group spawned by start_new_session=True.
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass
