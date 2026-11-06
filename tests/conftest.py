"""Test-wide settings, loaded by pytest before any test module imports deepof."""

import os

# Never open GUI windows during tests, also on machines with a display (CI has none anyway)
os.environ.setdefault("DEEPOF_HEADLESS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
