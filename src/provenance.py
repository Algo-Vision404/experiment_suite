"""Run provenance and environment capture utilities."""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from importlib.metadata import distributions
from typing import Any, Dict, Optional

def _git_revision() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None

def capture_environment() -> Dict[str, Any]:
    packages = {}
    for dist in distributions():
        name = dist.metadata.get("Name")
        if name:
            packages[name.lower()] = dist.version
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "python_implementation": platform.python_implementation(),
        "git_commit": _git_revision(),
        "cwd": os.getcwd(),
        "packages": dict(sorted(packages.items())),
    }
