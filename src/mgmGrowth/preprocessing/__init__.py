# src/mgmGrowth/preprocessing/__init__.py

import logging

# Set up module-level logger
LOGGER = logging.getLogger("mgmGrowth.preprocessing")
_hdlr = logging.StreamHandler()
_hdlr.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
LOGGER.addHandler(_hdlr)
LOGGER.setLevel(logging.INFO)