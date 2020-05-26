# @author lucasmiranda42

from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.constraints import UnitNorm
from tensorflow.keras.layers import Bidirectional, Dense, Dropout
from tensorflow.keras.layers import Lambda, LSTM, GRU
from tensorflow.keras.layers import RepeatVector, TimeDistributed
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from source.model_utils import *
import tensorflow as tf


class SEQ_2_SEQ_AE:
    def __init__(
        self,
        input_shape,
        CONV_filters=256,
        LSTM_units_1=256,
        LSTM_units_2=64,
        DENSE_2=64,
        DROPOUT_RATE=0.25,
        ENCODING=32,
        learn_rate=1e-3,
    ):
        self.input_shape = input_shape
        self.CONV_filters = CONV_filters
        self.LSTM_units_1 = LSTM_units_1
        self.LSTM_units_2 = LSTM_units_2
        self.DENSE_1 = LSTM_units_2
        self.DENSE_2 = DENSE_2
        self.DROPOUT_RATE = DROPOUT_RATE
        self.ENCODING = ENCODING
        self.learn_rate = learn_rate

    def build(self):
        # Encoder Layers
        Model_E0 = tf.keras.layers.Conv1D(
            filters=self.CONV_filters,
            kernel_size=5,
            strides=1,
            padding="causal",
            activation="relu",
        )
        Model_E1 = Bidirectional(
            GRU(
                self.LSTM_units_1,
                activation="tanh",
                return_sequences=True,
                kernel_constraint=UnitNorm(axis=0),
            )
        )
        Model_E2 = Bidirectional(
            GRU(
                self.LSTM_units_2,
                activation="tanh",
                return_sequences=False,
                kernel_constraint=UnitNorm(axis=0),
            )
        )
        Model_E3 = Dense(
            self.DENSE_1, activation="relu", kernel_constraint=UnitNorm(axis=0)
        )
        Model_E4 = Dense(
            self.DENSE_2, activation="relu", kernel_constraint=UnitNorm(axis=0)
        )
        Model_E5 = Dense(
            self.ENCODING,
            activation="relu",
            kernel_constraint=UnitNorm(axis=1),
            activity_regularizer=UncorrelatedFeaturesConstraint(3, weightage=1.0),
        )

        # Decoder layers
        Model_D4 = Bidirectional(
            GRU(
                self.LSTM_units_1,
                activation="tanh",
                return_sequences=True,
                kernel_constraint=UnitNorm(axis=1),
            )
        )
        Model_D5 = Bidirectional(
            GRU(
                self.LSTM_units_1,
                activation="sigmoid",
                return_sequences=True,
                kernel_constraint=UnitNorm(axis=1),
            )
        )

        # Define and instanciate encoder
        encoder = Sequential(name="SEQ_2_SEQ_Encoder")
        encoder.add(Input(shape=self.input_shape[1:]))
        encoder.add(Model_E0)
        encoder.add(Model_E1)
        encoder.add(Model_E2)
        encoder.add(Model_E3)
        encoder.add(Dropout(self.DROPOUT_RATE))
        encoder.add(Model_E4)
        encoder.add(Model_E5)

        # Define and instanciate decoder
        decoder = Sequential(name="SEQ_2_SEQ_Decoder")
        decoder.add(
            DenseTranspose(
                Model_E5, activation="relu", input_shape=(self.ENCODING,), output_dim=64
            )
        )
        decoder.add(DenseTranspose(Model_E4, activation="relu", output_dim=128))
        decoder.add(DenseTranspose(Model_E3, activation="relu", output_dim=256))
        decoder.add(RepeatVector(self.input_shape[1]))
        decoder.add(Model_D4)
        decoder.add(Model_D5)
        decoder.add(TimeDistributed(Dense(self.input_shape[2])))

        model = Sequential([encoder, decoder], name="SEQ_2_SEQ_AE")

        model.compile(
            loss=Huber(reduction="sum", delta=100.0),
            optimizer=Adam(lr=self.learn_rate, clipvalue=0.5,),
            metrics=["mae"],
        )

        return encoder, decoder, model


class SEQ_2_SEQ_VAE:
    def __init__(
        self,
        input_shape,
        CONV_filters=256,
        LSTM_units_1=256,
        LSTM_units_2=64,
        DENSE_2=64,
        DROPOUT_RATE=0.25,
        ENCODING=32,
        learn_rate=1e-3,
        loss="ELBO+MMD",
    ):
        self.input_shape = input_shape
        self.CONV_filters = CONV_filters
        self.LSTM_units_1 = LSTM_units_1
        self.LSTM_units_2 = LSTM_units_2
        self.DENSE_1 = LSTM_units_2
        self.DENSE_2 = DENSE_2
        self.DROPOUT_RATE = DROPOUT_RATE
        self.ENCODING = ENCODING
        self.learn_rate = learn_rate
        self.loss = loss

    def build(self):
        # Encoder Layers
        Model_E0 = tf.keras.layers.Conv1D(
            filters=self.CONV_filters,
            kernel_size=5,
            strides=1,
            padding="causal",
            activation="relu",
        )
        Model_E1 = Bidirectional(
            GRU(
                self.LSTM_units_1,
                activation="tanh",
                return_sequences=True,
                kernel_constraint=UnitNorm(axis=0),
            )
        )
        Model_E2 = Bidirectional(
            GRU(
                self.LSTM_units_2,
                activation="tanh",
                return_sequences=False,
                kernel_constraint=UnitNorm(axis=0),
            )
        )
        Model_E3 = Dense(
            self.DENSE_1, activation="relu", kernel_constraint=UnitNorm(axis=0)
        )
        Model_E4 = Dense(
            self.DENSE_2, activation="relu", kernel_constraint=UnitNorm(axis=0)
        )
        Model_E5 = Dense(
            self.ENCODING,
            activation="relu",
            kernel_constraint=UnitNorm(axis=1),
            activity_regularizer=UncorrelatedFeaturesConstraint(3, weightage=1.0),
        )

        # Decoder layers

        Model_D0 = DenseTranspose(Model_E5, activation="relu", output_dim=self.ENCODING)
        Model_D1 = DenseTranspose(Model_E4, activation="relu", output_dim=self.DENSE_2)
        Model_D2 = DenseTranspose(Model_E3, activation="relu", output_dim=self.DENSE_1)
        Model_D3 = RepeatVector(self.input_shape[1])
        Model_D4 = Bidirectional(
            GRU(
                self.LSTM_units_1,
                activation="tanh",
                return_sequences=True,
                kernel_constraint=UnitNorm(axis=1),
            )
        )
        Model_D5 = Bidirectional(
            GRU(
                self.LSTM_units_1,
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
        encoder = Dropout(self.DROPOUT_RATE)(encoder)
        encoder = Model_E4(encoder)
        encoder = Model_E5(encoder)

        z_mean = Dense(self.ENCODING)(encoder)
        z_log_sigma = Dense(self.ENCODING)(encoder)

        if "ELBO" in self.loss:
            z_mean, z_log_sigma = KLDivergenceLayer()([z_mean, z_log_sigma])

        z = Lambda(sampling)([z_mean, z_log_sigma])

        if "MMD" in self.loss:
            z = MMDiscrepancyLayer()(z)

        # Define and instanciate generator
        generator = Model_D0(z)
        generator = Model_D1(generator)
        generator = Model_D2(generator)
        generator = Model_D3(generator)
        generator = Model_D4(generator)
        generator = Model_D5(generator)
        x_decoded_mean = TimeDistributed(Dense(self.input_shape[2]))(generator)

        # end-to-end autoencoder
        encoder = Model(x, z_mean, name="SEQ_2_SEQ_VEncoder")
        vae = Model(x, x_decoded_mean, name="SEQ_2_SEQ_VAE")

        # Build generator as a separate entity
        g = Input(shape=self.ENCODING)
        _generator = Model_D0(g)
        _generator = Model_D1(_generator)
        _generator = Model_D2(_generator)
        _generator = Model_D3(_generator)
        _generator = Model_D4(_generator)
        _generator = Model_D5(_generator)
        _x_decoded_mean = TimeDistributed(Dense(self.input_shape[2]))(_generator)
        generator = Model(g, _x_decoded_mean, name="SEQ_2_SEQ_VGenerator")

        def huber_loss(x, x_decoded_mean):
            huber_loss = Huber(reduction="sum", delta=100.0)
            return self.input_shape[1:] * huber_loss(x, x_decoded_mean)

        vae.compile(
            loss=huber_loss,
            optimizer=Adam(lr=self.learn_rate,),
            metrics=["mae"],
            experimental_run_tf_function=False,
        )

        return encoder, generator, vae


class SEQ_2_SEQ_MVAE:
    pass


class SEQ_2_SEQ_MMVAE:
    pass


# TODO:
#      - Baseline networks (done!)
#      - Initial Convnet switch (done!)
#      - Bidirectional LSTM switches (done!)
#      - Change LSTMs for GRU
#      - Tied/Untied weights
#      - VAE loss function (though this should be analysed later on taking the encodings into account)
#      - Smaller input sliding window (10-15 frames)
