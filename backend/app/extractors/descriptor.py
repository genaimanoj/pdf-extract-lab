"""Static metadata for each registered extractor.

Every engine declares one of these so /engines can tell the frontend what's
installed, what license it ships under, and — importantly — whether the
current process can actually run it right now (keys present, binaries on
PATH, etc.).
"""
from __future__ import annotations

import os
import shutil
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def _resolve_bin(name: str) -> Optional[str]:
    """Look up an executable by name, with a venv-aware fallback.

    `shutil.which()` only checks PATH, but uvicorn launched via
    `.venv/bin/uvicorn` doesn't extend PATH with the venv's bin
    directory. Console scripts installed by the venv (e.g.
    `opendataloader-pdf-hybrid`) live next to the running interpreter,
    so a sibling-of-`sys.executable` lookup catches them.
    """
    found = shutil.which(name)
    if found:
        return found
    sibling = os.path.join(os.path.dirname(sys.executable), name)
    if os.path.isfile(sibling) and os.access(sibling, os.X_OK):
        return sibling
    return None


@dataclass(frozen=True)
class ExtractorDescriptor:
    # Stable machine name — what callers pass to /extract.
    name: str
    # Short human label for dropdowns.
    label: str
    # One-line description of approach / strengths.
    description: str
    license: str
    # Environment variables that must be set for the engine to run.
    requires_env: List[str] = field(default_factory=list)
    # Binaries that must be on PATH (e.g. "java", "tesseract").
    requires_bin: List[str] = field(default_factory=list)
    # Upstream project URL, if any.
    homepage: Optional[str] = None

    def check_available(self, env_lookup: Dict[str, str] | None = None) -> Tuple[bool, Optional[str]]:
        """Return (available, reason-if-not). env_lookup overrides os.environ, useful for tests."""
        env_lookup = env_lookup if env_lookup is not None else dict(os.environ)
        for var in self.requires_env:
            if not env_lookup.get(var):
                return False, f"Missing environment variable: {var}"
        for binary in self.requires_bin:
            if _resolve_bin(binary) is None:
                return False, f"Missing binary on PATH: {binary}"
        return True, None

    def to_public_dict(self, env_lookup: Dict[str, str] | None = None) -> Dict[str, Any]:
        """Serialize to the shape /engines returns to the frontend."""
        available, reason = self.check_available(env_lookup)
        d = asdict(self)
        d["available"] = available
        d["unavailable_reason"] = reason
        return d
