# src/mgmGrowth/preprocessing/__init__.py
from __future__ import annotations

import logging
import sys
from typing import Final

_FMT: Final = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
)

logging.basicConfig(
    stream=sys.stdout,
    format=_FMT,
    level=logging.INFO,
    force=False,
)

LOGGER: Final = logging.getLogger("mgmGrowth.preprocessing")
