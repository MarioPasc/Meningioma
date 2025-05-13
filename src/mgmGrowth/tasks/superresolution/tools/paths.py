# file: src/mgmGrowth/tasks/superresolution/tools/paths.py
from pathlib import Path


def ensure_dir(path: Path) -> Path:
    """Create *path* (parents too) if it does not exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
