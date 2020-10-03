# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Testing module for deepof.model_utils

"""

from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
import deepof.model_utils
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.framework.ops import EagerTensor

# For coverage.py to work with @tf.function decorated functions and methods,
# graph execution is disabled when running this script with pytest

tf.config.experimental_run_functions_eagerly(True)
tfpl = tfp.layers
tfd = tfp.distributions


@settings(deadline=None)
@given(
    shape=st.tuples(
        st.integers(min_value=5, max_value=10), st.integers(min_value=5, max_value=10)
    )
)
def test_far_away_uniform_initialiser(shape):
    far = deepof.model_utils.far_away_uniform_initialiser(shape, 0, 15, 1000)
    random = tf.random.uniform(shape, 0, 15)
    assert far.shape == shape
    assert tf.abs(tf.norm(tf.math.subtract(far[1:], far[:1]))) > tf.abs(
        tf.norm(tf.math.subtract(random[1:], random[:1]))
    )


@settings(deadline=None)
@given(
    tensor=arrays(
        shape=(10, 10),
        dtype=float,
        unique=True,
        elements=st.floats(min_value=-300, max_value=300),
    ),
)
def test_compute_mmd(tensor):
    tensor1 = tf.cast(tf.convert_to_tensor(tensor), dtype=tf.float32)
    tensor2 = tf.random.uniform(tensor1.shape, -300, 300, dtype=tf.float32)

    mmd_kernel = deepof.model_utils.compute_mmd(tuple([tensor1, tensor2]))
    null_kernel = deepof.model_utils.compute_mmd(tuple([tensor1, tensor1]))

    assert type(mmd_kernel) == EagerTensor
    assert null_kernel == 0


def test_one_cycle_scheduler():
    cycle1 = deepof.model_utils.one_cycle_scheduler(
        iterations=5, max_rate=1.0, start_rate=0.1, last_iterations=2, last_rate=0.3
    )
    assert type(cycle1._interpolate(1, 2, 0.2, 0.5)) == float

    X = np.random.uniform(0, 10, [1500, 5])
    y = np.random.randint(0, 2, [1500, 1])

    test_model = tf.keras.Sequential()
    test_model.add(tf.keras.layers.Dense(1))

    test_model.compile(
        loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.SGD(),
    )

    onecycle = deepof.model_utils.one_cycle_scheduler(
        X.shape[0] // 100 * 10, max_rate=0.005,
    )

    fit = test_model.fit(X, y, callbacks=[onecycle], epochs=10, batch_size=100)
    assert type(fit) == tf.python.keras.callbacks.History
    assert onecycle.history["lr"][4] > onecycle.history["lr"][1]
    assert onecycle.history["lr"][4] > onecycle.history["lr"][-1]


def test_uncorrelated_features_constraint():
    X = np.random.uniform(0, 10, [1500, 5])
    y = np.random.randint(0, 2, [1500, 1])

    correlations = []

    for w in range(2):
        test_model = tf.keras.Sequential()
        test_model.add(
            tf.keras.layers.Dense(
                10,
                kernel_constraint=tf.keras.constraints.UnitNorm(axis=1),
                activity_regularizer=deepof.model_utils.uncorrelated_features_constraint(
                    2, weightage=w
                ),
            )
        )

        test_model.compile(
            loss=tf.keras.losses.binary_crossentropy,
            optimizer=tf.keras.optimizers.SGD(),
        )

        fit = test_model.fit(X, y, epochs=10, batch_size=100)
        assert type(fit) == tf.python.keras.callbacks.History

        correlations.append(np.mean(np.corrcoef(test_model.get_weights()[0])))

    assert correlations[0] > correlations[1]


def test_MCDropout():
    X = np.random.uniform(0, 10, [1500, 5])
    y = np.random.randint(0, 2, [1500, 1])

    test_model = tf.keras.Sequential()
    test_model.add(tf.keras.layers.Dense(10))
    test_model.add(deepof.model_utils.MCDropout(0.5))

    test_model.compile(
        loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.SGD(),
    )

    fit = test_model.fit(X, y, epochs=10, batch_size=100)
    assert type(fit) == tf.python.keras.callbacks.History


def test_dense_transpose():
    X = np.random.uniform(0, 10, [1500, 10])
    y = np.random.randint(0, 2, [1500, 1])

    dense_1 = tf.keras.layers.Dense(10)

    dense_input = tf.keras.layers.Input(shape=(10,))
    dense_test = dense_1(dense_input)
    dense_tran = deepof.model_utils.DenseTranspose(dense_1, output_dim=10)(dense_test)
    test_model = tf.keras.Model(dense_input, dense_tran)

    test_model.compile(
        loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.SGD(),
    )

    fit = test_model.fit(X, y, epochs=10, batch_size=100)
    assert type(fit) == tf.python.keras.callbacks.History


def test_KLDivergenceLayer():
    X = tf.random.uniform([1500, 10], 0, 10)
    y = np.random.randint(0, 2, [1500, 1])

    prior = tfd.Independent(
        tfd.Normal(loc=tf.zeros(10), scale=1,), reinterpreted_batch_ndims=1,
    )

    dense_1 = tf.keras.layers.Dense(10)

    i = tf.keras.layers.Input(shape=(10,))
    d = dense_1(i)
    x = tfpl.DistributionLambda(
        lambda dense: tfd.Independent(
            tfd.Normal(loc=dense, scale=1,), reinterpreted_batch_ndims=1,
        )
    )(d)
    x = deepof.model_utils.KLDivergenceLayer(
        prior, weight=tf.keras.backend.variable(1.0, name="kl_beta")
    )(x)
    test_model = tf.keras.Model(i, x)

    test_model.compile(
        loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.SGD(),
    )

    fit = test_model.fit(X, y, epochs=10, batch_size=100)
    assert type(fit) == tf.python.keras.callbacks.History


def test_MMDiscrepancyLayer():
    X = tf.random.uniform([1500, 10], 0, 10)
    y = np.random.randint(0, 2, [1500, 1])

    prior = tfd.Independent(
        tfd.Normal(loc=tf.zeros(10), scale=1,), reinterpreted_batch_ndims=1,
    )

    dense_1 = tf.keras.layers.Dense(10)

    i = tf.keras.layers.Input(shape=(10,))
    d = dense_1(i)
    x = tfpl.DistributionLambda(
        lambda dense: tfd.Independent(
            tfd.Normal(loc=dense, scale=1,), reinterpreted_batch_ndims=1,
        )
    )(d)

    x = deepof.model_utils.MMDiscrepancyLayer(
        100, prior, beta=tf.keras.backend.variable(1.0, name="kl_beta")
    )(x)
    test_model = tf.keras.Model(i, x)

    test_model.compile(
        loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.SGD(),
    )

    fit = test_model.fit(X, y, epochs=10, batch_size=100)
    assert type(fit) == tf.python.keras.callbacks.History


def test_dead_neuron_control():
    X = np.random.uniform(0, 10, [1500, 5])
    y = np.random.randint(0, 2, [1500, 1])

    test_model = tf.keras.Sequential()
    test_model.add(tf.keras.layers.Dense(1))
    test_model.add(deepof.model_utils.Dead_neuron_control())

    test_model.compile(
        loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.SGD(),
    )

    fit = test_model.fit(X, y, epochs=10, batch_size=100)
    assert type(fit) == tf.python.keras.callbacks.History


def test_entropy_regulariser():
    X = np.random.uniform(0, 10, [1500, 5])
    y = np.random.randint(0, 2, [1500, 1])

    test_model = tf.keras.Sequential()
    test_model.add(tf.keras.layers.Dense(1))
    test_model.add(deepof.model_utils.Entropy_regulariser(1.0))

    test_model.compile(
        loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.SGD(),
    )

    fit = test_model.fit(X, y, epochs=10, batch_size=100)
    assert type(fit) == tf.python.keras.callbacks.History


def test_find_learning_rate():
    X = np.random.uniform(0, 10, [1500, 5])
    y = np.random.randint(0, 2, [1500, 1])

    test_model = tf.keras.Sequential()
    test_model.add(tf.keras.layers.Dense(1))
    test_model.add(deepof.model_utils.Entropy_regulariser(1.0))

    test_model.compile(
        loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.SGD(),
    )
    test_model.build(X.shape)

    deepof.model_utils.find_learning_rate(test_model, X, y)
