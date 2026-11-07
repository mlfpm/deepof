"""Test-wide settings, loaded by pytest before any test module imports deepof."""

import os
import sys

import pytest

# Never open GUI windows during tests, also on machines with a display (CI has none anyway)
os.environ.setdefault("DEEPOF_HEADLESS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture(autouse=True)
def _remove_test_folders():
    """Remove the example projects and output folders a test created with tests.test_objects.test_objects, also
    when it failed.

    The module is looked up instead of imported: test runs that never use it (e.g. tests/regression, started with plain
    pytest, where tests is not importable) do not need it.
    """
    yield
    test_objects = sys.modules.get("tests.test_objects.test_objects")
    if test_objects is not None:
        test_objects.remove_test_folders()
