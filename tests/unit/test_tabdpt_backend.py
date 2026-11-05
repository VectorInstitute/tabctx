"""The parts of the TabDPT backend that need no torch or weights; the
model itself is tested in tests/integration/test_tabdpt_real.py."""

import numpy as np

from tabctx.backends.tabdpt import _context_bytes, served_model_id
from tabctx.client import KNOWN_MODELS


def test_model_id_follows_the_checkpoint():
    assert served_model_id("1_3") == "tabdpt-1.3"
    assert served_model_id("1_3") in KNOWN_MODELS


def test_a_context_costs_its_table_not_the_network():
    class Fitted:
        X_train = np.zeros((100, 4))
        y_train = np.zeros(100)
        projection = None
        model = object()  # the shared network: never counted

    assert _context_bytes(Fitted()) == 100 * 4 * 8 + 100 * 8
