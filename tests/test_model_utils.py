# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Testing module for deepof.model_utils

"""

from hypothesis import given
from hypothesis import HealthCheck
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
import deepof.models
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


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
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

    assert isinstance(mmd_kernel, EagerTensor)
    assert null_kernel == 0


# noinspection PyUnresolvedReferences
def test_one_cycle_scheduler():
    cycle1 = deepof.model_utils.one_cycle_scheduler(
        iterations=5, max_rate=1.0, start_rate=0.1, last_iterations=2, last_rate=0.3
    )
    assert isinstance(cycle1._interpolate(1, 2, 0.2, 0.5), float)

    X = np.random.uniform(0, 10, [1500, 5])
    y = np.random.randint(0, 2, [1500, 1])

    test_model = tf.keras.Sequential()
    test_model.add(tf.keras.layers.Dense(1))

    test_model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.SGD(),
    )

    onecycle = deepof.model_utils.one_cycle_scheduler(
        X.shape[0] // 100 * 10,
        max_rate=0.005,
    )

    fit = test_model.fit(
        X, y, callbacks=[onecycle], epochs=10, batch_size=100, verbose=0
    )
    assert isinstance(fit, tf.keras.callbacks.History)
    assert onecycle.history["lr"][4] > onecycle.history["lr"][1]
    assert onecycle.history["lr"][4] > onecycle.history["lr"][-1]


# noinspection PyUnresolvedReferences
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

        fit = test_model.fit(X, y, epochs=25, batch_size=100, verbose=0)
        assert isinstance(fit, tf.keras.callbacks.History)

        correlations.append(np.mean(np.corrcoef(test_model.get_weights()[0])))

    assert correlations[0] > correlations[1]


# noinspection PyUnresolvedReferences
def test_MCDropout():
    X = np.random.uniform(0, 10, [1500, 5])
    y = np.random.randint(0, 2, [1500, 1])

    test_model = tf.keras.Sequential()
    test_model.add(tf.keras.layers.Dense(10))
    test_model.add(deepof.model_utils.MCDropout(0.5))

    test_model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.SGD(),
    )

    fit = test_model.fit(X, y, epochs=10, batch_size=100, verbose=0)
    assert isinstance(fit, tf.keras.callbacks.History)


# noinspection PyUnresolvedReferences
def test_dense_transpose():
    X = np.random.uniform(0, 10, [1500, 10])
    y = np.random.randint(0, 2, [1500, 1])

    dense_1 = tf.keras.layers.Dense(10)

    dense_input = tf.keras.layers.Input(shape=(10,))
    dense_test = dense_1(dense_input)
    dense_tran = deepof.model_utils.DenseTranspose(dense_1, output_dim=10)(dense_test)
    test_model = tf.keras.Model(dense_input, dense_tran)

    test_model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.SGD(),
    )

    fit = test_model.fit(X, y, epochs=10, batch_size=100, verbose=0)
    assert isinstance(fit, tf.keras.callbacks.History)


# noinspection PyCallingNonCallable,PyUnresolvedReferences
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(annealing_mode=st.one_of(st.just("linear"), st.just("sigmoid")))
def test_KLDivergenceLayer(annealing_mode):
    X = tf.random.uniform([10, 2], 0, 10)
    y = np.random.randint(0, 1, [10, 1])

    prior = tfd.Independent(
        tfd.Normal(
            loc=tf.zeros(2),
            scale=1,
        ),
        reinterpreted_batch_ndims=1,
    )

    dense_1 = tf.keras.layers.Dense(2)

    i = tf.keras.layers.Input(shape=(2,))
    d = dense_1(i)
    x = tfpl.DistributionLambda(
        lambda dense: tfd.Independent(
            tfd.Normal(
                loc=dense,
                scale=1,
            ),
            reinterpreted_batch_ndims=1,
        )
    )(d)
    kl_canon = tfpl.KLDivergenceAddLoss(
        prior,
        weight=1.0,
    )(x)
    kl_deepof = deepof.model_utils.KLDivergenceLayer(
        distribution_b=prior,
        iters=1,
        warm_up_iters=0,
        annealing_mode=annealing_mode,
    )(x)
    test_model = tf.keras.Model(i, [kl_canon, kl_deepof])

    test_model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.SGD(),
    )

    fit = test_model.fit(X, [y, y], epochs=1, batch_size=100, verbose=0)
    assert isinstance(fit, tf.keras.callbacks.History)
    assert test_model.losses[0] == test_model.losses[1]


# noinspection PyUnresolvedReferences
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(annealing_mode=st.one_of(st.just("linear"), st.just("sigmoid")))
def test_MMDiscrepancyLayer(annealing_mode):
    X = tf.random.uniform([1500, 10], 0, 10)
    y = np.random.randint(0, 2, [1500, 1])

    prior = tfd.Independent(
        tfd.Normal(
            loc=tf.zeros(10),
            scale=1,
        ),
        reinterpreted_batch_ndims=1,
    )

    dense_1 = tf.keras.layers.Dense(10)

    i = tf.keras.layers.Input(shape=(10,))
    d = dense_1(i)
    x = tfpl.DistributionLambda(
        lambda dense: tfd.Independent(
            tfd.Normal(
                loc=dense,
                scale=1,
            ),
            reinterpreted_batch_ndims=1,
        )
    )(d)

    x = deepof.model_utils.MMDiscrepancyLayer(
        batch_size=100,
        prior=prior,
        iters=1,
        warm_up_iters=0,
        annealing_mode=annealing_mode,
    )(x)
    test_model = tf.keras.Model(i, x)

    test_model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.SGD(),
    )

    fit = test_model.fit(X, y, epochs=10, batch_size=100, verbose=0)
    assert isinstance(fit, tf.keras.callbacks.History)


# noinspection PyUnresolvedReferences
def test_dead_neuron_control():
    X = np.random.uniform(0, 10, [1500, 5])
    y = np.random.randint(0, 2, [1500, 1])

    test_model = tf.keras.Sequential()
    test_model.add(tf.keras.layers.Dense(1))
    test_model.add(deepof.model_utils.Dead_neuron_control())

    test_model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.SGD(),
    )

    fit = test_model.fit(X, y, epochs=10, batch_size=100, verbose=0)
    assert isinstance(fit, tf.keras.callbacks.History)


def test_find_learning_rate():
    X = np.random.uniform(0, 10, [1500, 5])
    y = np.random.randint(0, 2, [1500, 1])

    test_model = tf.keras.Sequential()
    test_model.add(tf.keras.layers.Dense(1))

    test_model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.SGD(),
    )
    test_model.build(X.shape)

    deepof.model_utils.find_learning_rate(test_model, X, y)


def test_neighbor_latent_entropy():
    X = np.random.normal(0, 1, [1500, 25, 6])

    test_model = deepof.models.GMVAE()
    gmvaep = test_model.build(X.shape)[3]

    gmvaep.fit(
        X,
        X,
        callbacks=deepof.model_utils.neighbor_latent_entropy(
            k=10,
            encoding_dim=6,
            validation_data=X,
            variational=True,
        ),
    )
