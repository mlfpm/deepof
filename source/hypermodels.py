# @author lucasmiranda42

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.layers import RepeatVector, Dropout
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.constraints import UnitNorm, Constraint
from tensorflow.keras import Sequential
from keras import backend as K
from kerastuner import HyperModel

# Custom layers for efficiency
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

    def call(self, inputs):
        z = tf.matmul(inputs, self.dense.weights[0], transpose_b=True)
        return self.activation(z + self.biases)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


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


class SEQ_2_SEQ_AE(HyperModel):
    def __init__(self, input_shape):
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
            loss=tf.keras.losses.Huber(reduction="sum", delta=100.0),
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
