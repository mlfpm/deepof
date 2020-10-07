# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Testing module for deepof.train_utils

"""

from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
import deepof.train_utils
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.framework.ops import EagerTensor

# For coverage.py to work with @tf.function decorated functions and methods,
# graph execution is disabled when running this script with pytest

tf.config.experimental_run_functions_eagerly(True)
tfpl = tfp.layers
tfd = tfp.distributions


@given(encoding=st.integers(min_value=1, max_value=128))
def test_load_hparams(encoding):
    params = deepof.train_utils.load_hparams(None, encoding)
    assert type(params) == dict
    assert params["encoding"] == encoding


def test_load_treatments():
    pass


def test_get_callbacks():
    pass


def test_tune_search():
    pass
