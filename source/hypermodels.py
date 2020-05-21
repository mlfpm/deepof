# @author lucasmiranda42

from kerastuner import HyperModel
from keras import backend as K
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.constraints import Constraint, UnitNorm
from tensorflow.keras.layers import Bidirectional, Dense, Dropout
from tensorflow.keras.layers import Lambda, Layer, LSTM
from tensorflow.keras.layers import RepeatVector, TimeDistributed
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


# Helper functions
def sampling(args, epsilon_std=1.0):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=K.shape(z_mean), mean=0.0, stddev=epsilon_std)
    return z_mean + K.exp(z_log_sigma) * epsilon


def compute_kernel(x, y):
    x_size = K.shape(x)[0]
    y_size = K.shape(y)[0]
    dim = K.shape(x)[1]
    tiled_x = K.tile(K.reshape(x, K.stack([x_size, 1, dim])), K.stack([1, y_size, 1]))
    tiled_y = K.tile(K.reshape(y, K.stack([1, y_size, dim])), K.stack([x_size, 1, 1]))
    return K.exp(
        -tf.reduce_mean(K.square(tiled_x - tiled_y), axis=2) / K.cast(dim, tf.float32)
    )


def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return (
        tf.reduce_mean(x_kernel)
        + tf.reduce_mean(y_kernel)
        - 2 * tf.reduce_mean(xy_kernel)
    )


# Custom layers for efficiency/losses
class DenseTranspose(Layer):
    def __init__(self, dense, output_dim, activation=None, **kwargs):
        self.dense = dense
        self.output_dim = output_dim
        self.activation = tf.keras.activations.get(activation)
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "dense": self.dense,
                "output_dim": self.output_dim,
                "activation": self.activation,
            }
        )
        return config

    def build(self, batch_input_shape):
        self.biases = self.add_weight(
            name="bias", shape=[self.dense.input_shape[-1]], initializer="zeros"
        )
        super().build(batch_input_shape)

    def call(self, inputs, **kwargs):
        z = tf.matmul(inputs, self.dense.weights[0], transpose_b=True)
        return self.activation(z + self.biases)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim


class UncorrelatedFeaturesConstraint(Constraint):
    def __init__(self, encoding_dim, weightage=1.0):
        self.encoding_dim = encoding_dim
        self.weightage = weightage

    def get_config(self):

        config = super().get_config().copy()
        config.update(
            {"encoding_dim": self.encoding_dim, "weightage": self.weightage,}
        )
        return config

    def get_covariance(self, x):
        x_centered_list = []

        for i in range(self.encoding_dim):
            x_centered_list.append(x[:, i] - K.mean(x[:, i]))

        x_centered = tf.stack(x_centered_list)
        covariance = K.dot(x_centered, K.transpose(x_centered)) / tf.cast(
            x_centered.get_shape()[0], tf.float32
        )

        return covariance

    # Constraint penalty
    def uncorrelated_feature(self, x):
        if self.encoding_dim <= 1:
            return 0.0
        else:
            output = K.sum(
                K.square(
                    self.covariance
                    - tf.math.multiply(self.covariance, K.eye(self.encoding_dim))
                )
            )
            return output

    def __call__(self, x):
        self.covariance = self.get_covariance(x)
        return self.weightage * self.uncorrelated_feature(x)


class KLDivergenceLayer(Layer):

    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):

        mu, log_var = inputs

        kl_batch = -0.5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs


class MMDiscrepancyLayer(Layer):
    """ Identity transform layer that adds MM discrepancy
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(MMDiscrepancyLayer, self).__init__(*args, **kwargs)

    def call(self, z, **kwargs):
        true_samples = K.random_normal(
            K.shape(z), mean=0.0, stddev=2.0 / K.cast_to_floatx(K.shape(z)[1])
        )
        mmd_batch = compute_mmd(z, true_samples)

        self.add_loss(K.mean(mmd_batch), inputs=z)

        return z


# Hypermodels
class SEQ_2_SEQ_AE(HyperModel):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

    def build(self, hp):
        # Hyperparameters to tune
        CONV_filters = hp.Int(
            "units_conv", min_value=32, max_value=256, step=32, default=256
        )
        LSTM_units_1 = hp.Int(
            "units_lstm", min_value=128, max_value=512, step=32, default=256
        )
        LSTM_units_2 = int(LSTM_units_1 / 2)
        DENSE_1 = int(LSTM_units_2)
        DENSE_2 = hp.Int(
            "units_dense1", min_value=32, max_value=256, step=32, default=64
        )
        DROPOUT_RATE = hp.Float(
            "dropout_rate", min_value=0.0, max_value=0.5, default=0.25, step=0.05
        )
        ENCODING = hp.Int(
            "units_dense2", min_value=32, max_value=128, step=32, default=32
        )

        # Encoder Layers
        Model_E0 = tf.keras.layers.Conv1D(
            filters=CONV_filters,
            kernel_size=5,
            strides=1,
            padding="causal",
            activation="relu",
            input_shape=self.input_shape[1:],
        )
        Model_E1 = Bidirectional(
            LSTM(
                LSTM_units_1,
                activation="tanh",
                return_sequences=True,
                kernel_constraint=UnitNorm(axis=0),
            )
        )
        Model_E2 = Bidirectional(
            LSTM(
                LSTM_units_2,
                activation="tanh",
                return_sequences=False,
                kernel_constraint=UnitNorm(axis=0),
            )
        )
        Model_E3 = Dense(DENSE_1, activation="relu", kernel_constraint=UnitNorm(axis=0))
        Model_E4 = Dense(DENSE_2, activation="relu", kernel_constraint=UnitNorm(axis=0))
        Model_E5 = Dense(
            ENCODING,
            activation="relu",
            kernel_constraint=UnitNorm(axis=1),
            activity_regularizer=UncorrelatedFeaturesConstraint(3, weightage=1.0),
        )

        # Decoder layers
        Model_D4 = Bidirectional(
            LSTM(
                LSTM_units_1,
                activation="tanh",
                return_sequences=True,
                kernel_constraint=UnitNorm(axis=1),
            )
        )
        Model_D5 = Bidirectional(
            LSTM(
                LSTM_units_1,
                activation="sigmoid",
                return_sequences=True,
                kernel_constraint=UnitNorm(axis=1),
            )
        )

        # Define and instanciate encoder
        encoder = Sequential(name="DLC_encoder")
        encoder.add(Model_E0)
        encoder.add(Model_E1)
        encoder.add(Model_E2)
        encoder.add(Model_E3)
        encoder.add(Dropout(DROPOUT_RATE))
        encoder.add(Model_E4)
        encoder.add(Model_E5)

        # Define and instanciate decoder
        decoder = Sequential(name="DLC_Decoder")
        decoder.add(
            DenseTranspose(
                Model_E5, activation="relu", input_shape=(ENCODING,), output_dim=64
            )
        )
        decoder.add(DenseTranspose(Model_E4, activation="relu", output_dim=128))
        decoder.add(DenseTranspose(Model_E3, activation="relu", output_dim=256))
        decoder.add(RepeatVector(self.input_shape[1]))
        decoder.add(Model_D4)
        decoder.add(Model_D5)
        decoder.add(TimeDistributed(Dense(self.input_shape[2])))

        model = Sequential([encoder, decoder], name="DLC_Autoencoder")

        model.compile(
            loss=Huber(reduction="sum", delta=100.0),
            optimizer=Adam(
                lr=hp.Float(
                    "learning_rate",
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling="LOG",
                    default=1e-3,
                ),
                clipvalue=0.5,
            ),
            metrics=["mae"],
        )

        return model


class SEQ_2_SEQ_VAE(HyperModel):
    def __init__(self, input_shape, loss="ELBO+MMD"):
        super().__init__()
        self.input_shape = input_shape
        self.loss = loss

        assert self.loss in [
            "MMD",
            "ELBO",
            "ELBO+MMD",
        ], "Loss function not recognised. Select one of ELBO, MMD and ELBO+MMD"

    def build(self, hp):
        # Hyperparameters to tune
        CONV_filters = hp.Int(
            "units_conv", min_value=32, max_value=256, step=32, default=256
        )
        LSTM_units_1 = hp.Int(
            "units_lstm", min_value=128, max_value=512, step=32, default=256
        )
        LSTM_units_2 = int(LSTM_units_1 / 2)
        DENSE_1 = int(LSTM_units_2)
        DENSE_2 = hp.Int(
            "units_dense1", min_value=32, max_value=256, step=32, default=64
        )
        DROPOUT_RATE = hp.Float(
            "dropout_rate", min_value=0.0, max_value=0.5, default=0.25, step=0.05
        )
        ENCODING = hp.Int(
            "units_dense2", min_value=32, max_value=128, step=32, default=32
        )

        # Encoder Layers
        Model_E0 = tf.keras.layers.Conv1D(
            filters=CONV_filters,
            kernel_size=5,
            strides=1,
            padding="causal",
            activation="relu",
        )
        Model_E1 = Bidirectional(
            LSTM(
                LSTM_units_1,
                activation="tanh",
                return_sequences=True,
                kernel_constraint=UnitNorm(axis=0),
            )
        )
        Model_E2 = Bidirectional(
            LSTM(
                LSTM_units_2,
                activation="tanh",
                return_sequences=False,
                kernel_constraint=UnitNorm(axis=0),
            )
        )
        Model_E3 = Dense(DENSE_1, activation="relu", kernel_constraint=UnitNorm(axis=0))
        Model_E4 = Dense(DENSE_2, activation="relu", kernel_constraint=UnitNorm(axis=0))
        Model_E5 = Dense(
            ENCODING,
            activation="relu",
            kernel_constraint=UnitNorm(axis=1),
            activity_regularizer=UncorrelatedFeaturesConstraint(3, weightage=1.0),
        )

        # Decoder layers
        Model_D4 = Bidirectional(
            LSTM(
                LSTM_units_1,
                activation="tanh",
                return_sequences=True,
                kernel_constraint=UnitNorm(axis=1),
            )
        )
        Model_D5 = Bidirectional(
            LSTM(
                LSTM_units_1,
                activation="sigmoid",
                return_sequences=True,
                kernel_constraint=UnitNorm(axis=1),
            )
        )

        # Define and instanciate encoder
        x = Input(shape=self.input_shape[1:])
        encoder = Model_E0(x)
        encoder = Model_E1(encoder)
        encoder = Model_E2(encoder)
        encoder = Model_E3(encoder)
        encoder = Dropout(DROPOUT_RATE)(encoder)
        encoder = Model_E4(encoder)
        encoder = Model_E5(encoder)

        z_mean = Dense(ENCODING)(encoder)
        z_log_sigma = Dense(ENCODING)(encoder)

        if "ELBO" in self.loss:
            z_mean, z_log_sigma = KLDivergenceLayer()([z_mean, z_log_sigma])

        z = Lambda(sampling)([z_mean, z_log_sigma])

        if "MMD" in self.loss:
            z = MMDiscrepancyLayer()(z)

        # Define and instanciate decoder
        decoder = DenseTranspose(Model_E5, activation="relu", output_dim=ENCODING)(z)
        decoder = DenseTranspose(Model_E4, activation="relu", output_dim=DENSE_2)(
            decoder
        )
        decoder = DenseTranspose(Model_E3, activation="relu", output_dim=DENSE_1)(
            decoder
        )
        decoder = RepeatVector(self.input_shape[1])(decoder)
        decoder = Model_D4(decoder)
        decoder = Model_D5(decoder)
        x_decoded_mean = TimeDistributed(Dense(self.input_shape[2]))(decoder)

        # end-to-end autoencoder
        vae = Model(x, x_decoded_mean)

        def huber_loss(x, x_decoded_mean):
            huber_loss = Huber(reduction="sum", delta=100.0)
            return self.input_shape[1:] * huber_loss(x, x_decoded_mean)

        vae.compile(
            loss=huber_loss,
            optimizer=Adam(
                lr=hp.Float(
                    "learning_rate",
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling="LOG",
                    default=1e-3,
                ),
            ),
            metrics=["mae"],
            experimental_run_tf_function=False,
        )

        return vae


class SEQ_2_SEQ_MVAE(HyperModel):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

    def build(self, hp):
        pass


class SEQ_2_SEQ_MMVAE(HyperModel):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

    def build(self, hp):
        pass
