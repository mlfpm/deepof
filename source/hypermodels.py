# @author lucasmiranda42

from kerastuner import HyperModel
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.activations import softplus
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.constraints import UnitNorm
from tensorflow.keras.initializers import he_uniform, Orthogonal, RandomNormal
from tensorflow.keras.layers import BatchNormalization, Bidirectional
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.layers import RepeatVector, Reshape, TimeDistributed
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from source.model_utils import *
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfpl = tfp.layers


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
            "units_dense2", min_value=32, max_value=256, step=32, default=64
        )
        DROPOUT_RATE = hp.Float(
            "dropout_rate", min_value=0.0, max_value=0.5, default=0.25, step=0.05
        )
        ENCODING = hp.Int("encoding", min_value=32, max_value=128, step=32, default=32)

        # Encoder Layers
        Model_E0 = tf.keras.layers.Conv1D(
            filters=CONV_filters,
            kernel_size=5,
            strides=1,
            padding="causal",
            activation="relu",
            kernel_initializer=he_uniform(),
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
        Model_E3 = Dense(
            DENSE_1,
            activation="relu",
            kernel_constraint=UnitNorm(axis=0),
            kernel_initializer=he_uniform(),
        )
        Model_E4 = Dense(
            DENSE_2,
            activation="relu",
            kernel_constraint=UnitNorm(axis=0),
            kernel_initializer=he_uniform(),
        )
        Model_E5 = Dense(
            ENCODING,
            activation="relu",
            kernel_constraint=UnitNorm(axis=1),
            activity_regularizer=UncorrelatedFeaturesConstraint(3, weightage=1.0),
            kernel_initializer=Orthogonal(),
        )

        # Decoder layers
        Model_D0 = DenseTranspose(Model_E5, activation="relu", output_dim=ENCODING,)
        Model_D1 = DenseTranspose(Model_E4, activation="relu", output_dim=DENSE_2,)
        Model_D2 = DenseTranspose(Model_E3, activation="relu", output_dim=DENSE_1,)
        Model_D3 = RepeatVector(self.input_shape[1])
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

        # Define and instantiate encoder
        encoder = Sequential(name="SEQ_2_SEQ_Encoder")
        encoder.add(Input(shape=self.input_shape[1:]))
        encoder.add(Model_E0)
        encoder.add(BatchNormalization())
        encoder.add(Model_E1)
        encoder.add(BatchNormalization())
        encoder.add(Model_E2)
        encoder.add(BatchNormalization())
        encoder.add(Model_E3)
        encoder.add(BatchNormalization())
        encoder.add(Dropout(DROPOUT_RATE))
        encoder.add(Model_E4)
        encoder.add(BatchNormalization())
        encoder.add(Model_E5)

        # Define and instantiate decoder
        decoder = Sequential(name="SEQ_2_SEQ_Decoder")
        decoder.add(Model_D0)
        decoder.add(BatchNormalization())
        decoder.add(Model_D1)
        decoder.add(BatchNormalization())
        decoder.add(Model_D2)
        decoder.add(BatchNormalization())
        decoder.add(Model_D3)
        decoder.add(Model_D4)
        decoder.add(BatchNormalization())
        decoder.add(Model_D5)
        decoder.add(TimeDistributed(Dense(self.input_shape[2])))

        model = Sequential([encoder, decoder], name="SEQ_2_SEQ_AE")

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


class SEQ_2_SEQ_GMVAE(HyperModel):
    def __init__(
        self,
        input_shape,
        CONV_filters=256,
        LSTM_units_1=256,
        LSTM_units_2=128,
        DENSE_2=64,
        learn_rate=1e-3,
        loss="ELBO+MMD",
        kl_warmup_epochs=0,
        mmd_warmup_epochs=0,
        prior="standard_normal",
        number_of_components=1,
        predictor=True,
    ):
        self.input_shape = input_shape
        self.CONV_filters = CONV_filters
        self.LSTM_units_1 = LSTM_units_1
        self.LSTM_units_2 = LSTM_units_2
        self.DENSE_1 = LSTM_units_2
        self.DENSE_2 = DENSE_2
        self.learn_rate = learn_rate
        self.loss = loss
        self.prior = prior
        self.kl_warmup = kl_warmup_epochs
        self.mmd_warmup = mmd_warmup_epochs
        self.number_of_components = number_of_components
        self.predictor = predictor

        assert (
            "ELBO" in self.loss or "MMD" in self.loss
        ), "loss must be one of ELBO, MMD or ELBO+MMD (default)"

    def build(self, hp):
        # Hyperparameters to tune
        DROPOUT_RATE = hp.Float(
            "dropout_rate", min_value=0.0, max_value=0.5, default=0.25, step=0.05
        )
        ENCODING = hp.Int(
            "units_dense2", min_value=32, max_value=128, step=32, default=32
        )

        if self.prior == "standard_normal":
            self.prior = tfd.mixture.Mixture(
                tfd.categorical.Categorical(
                    probs=tf.ones(self.number_of_components) / self.number_of_components
                ),
                [
                    tfd.Independent(
                        tfd.Normal(loc=tf.zeros(ENCODING), scale=1),
                        reinterpreted_batch_ndims=1,
                    )
                    for _ in range(self.number_of_components)
                ],
            )

        # Encoder Layers
        Model_E0 = tf.keras.layers.Conv1D(
            filters=self.CONV_filters,
            kernel_size=5,
            strides=1,
            padding="causal",
            activation="relu",
            kernel_initializer=he_uniform(),
        )
        Model_E1 = Bidirectional(
            LSTM(
                self.LSTM_units_1,
                activation="tanh",
                return_sequences=True,
                kernel_constraint=UnitNorm(axis=0),
            )
        )
        Model_E2 = Bidirectional(
            LSTM(
                self.LSTM_units_2,
                activation="tanh",
                return_sequences=False,
                kernel_constraint=UnitNorm(axis=0),
            )
        )
        Model_E3 = Dense(
            self.DENSE_1,
            activation="relu",
            kernel_constraint=UnitNorm(axis=0),
            kernel_initializer=he_uniform(),
        )
        Model_E4 = Dense(
            self.DENSE_2,
            activation="relu",
            kernel_constraint=UnitNorm(axis=0),
            kernel_initializer=he_uniform(),
        )

        # Decoder layers
        Model_B1 = BatchNormalization()
        Model_B2 = BatchNormalization()
        Model_B3 = BatchNormalization()
        Model_B4 = BatchNormalization()
        Model_D1 = Dense(
            self.DENSE_2, activation="relu", kernel_initializer=he_uniform()
        )
        Model_D2 = Dense(
            self.DENSE_1, activation="relu", kernel_initializer=he_uniform()
        )
        Model_D3 = RepeatVector(self.input_shape[1])
        Model_D4 = Bidirectional(
            LSTM(
                self.LSTM_units_2,
                activation="tanh",
                return_sequences=True,
                kernel_constraint=UnitNorm(axis=1),
            )
        )
        Model_D5 = Bidirectional(
            LSTM(
                self.LSTM_units_1,
                activation="sigmoid",
                return_sequences=True,
                kernel_constraint=UnitNorm(axis=1),
            )
        )

        # Define and instantiate encoder
        x = Input(shape=self.input_shape[1:])
        encoder = Model_E0(x)
        encoder = BatchNormalization()(encoder)
        encoder = Model_E1(encoder)
        encoder = BatchNormalization()(encoder)
        encoder = Model_E2(encoder)
        encoder = BatchNormalization()(encoder)
        encoder = Model_E3(encoder)
        encoder = BatchNormalization()(encoder)
        encoder = Dropout(DROPOUT_RATE)(encoder)
        encoder = Model_E4(encoder)
        encoder = BatchNormalization()(encoder)

        z_cat = Dense(
            self.number_of_components,
            activation="softmax",
            kernel_initializer=RandomNormal(
                mean=(1 / self.number_of_components), stddev=0.05, seed=None
            ),
        )(encoder)
        z_gauss = Dense(
            tfpl.IndependentNormal.params_size(ENCODING * self.number_of_components),
            activation=None,
        )(encoder)

        # Define and control custom loss functions
        kl_warmup_callback = False
        if "ELBO" in self.loss:

            kl_beta = K.variable(1.0, name="kl_beta")
            kl_beta._trainable = False
            if self.kl_warmup:
                kl_warmup_callback = LambdaCallback(
                    on_epoch_begin=lambda epoch, logs: K.set_value(
                        kl_beta, K.min([epoch / self.kl_warmup, 1])
                    )
                )

        z_gauss = Reshape([2 * ENCODING, self.number_of_components])(z_gauss)
        z = tfpl.DistributionLambda(
            lambda gauss: tfd.mixture.Mixture(
                cat=tfd.categorical.Categorical(probs=gauss[0],),
                components=[
                    tfd.Independent(
                        tfd.Normal(
                            loc=gauss[1][..., :ENCODING, k],
                            scale=softplus(gauss[1][..., ENCODING:, k]),
                        ),
                        reinterpreted_batch_ndims=1,
                    )
                    for k in range(self.number_of_components)
                ],
            ),
            activity_regularizer=UncorrelatedFeaturesConstraint(3, weightage=1.0),
        )([z_cat, z_gauss])

        if "ELBO" in self.loss:
            z = KLDivergenceLayer(self.prior, weight=kl_beta)(z)

        mmd_warmup_callback = False
        if "MMD" in self.loss:

            mmd_beta = K.variable(1.0, name="mmd_beta")
            mmd_beta._trainable = False
            if self.mmd_warmup:
                mmd_warmup_callback = LambdaCallback(
                    on_epoch_begin=lambda epoch, logs: K.set_value(
                        mmd_beta, K.min([epoch / self.mmd_warmup, 1])
                    )
                )

            z = MMDiscrepancyLayer(prior=self.prior, beta=mmd_beta)(z)

        # Define and instantiate generator
        generator = Model_D1(z)
        generator = Model_B1(generator)
        generator = Model_D2(generator)
        generator = Model_B2(generator)
        generator = Model_D3(generator)
        generator = Model_D4(generator)
        generator = Model_B3(generator)
        generator = Model_D5(generator)
        generator = Model_B4(generator)
        x_decoded_mean = TimeDistributed(
            Dense(self.input_shape[2]), name="vaep_reconstruction"
        )(generator)

        if self.predictor:
            # Define and instantiate predictor
            predictor = Dense(
                self.DENSE_2, activation="relu", kernel_initializer=he_uniform()
            )(z)
            predictor = BatchNormalization()(predictor)
            predictor = Dense(
                self.DENSE_1, activation="relu", kernel_initializer=he_uniform()
            )(predictor)
            predictor = BatchNormalization()(predictor)
            predictor = RepeatVector(self.input_shape[1])(predictor)
            predictor = Bidirectional(
                LSTM(
                    self.LSTM_units_1,
                    activation="tanh",
                    return_sequences=True,
                    kernel_constraint=UnitNorm(axis=1),
                )
            )(predictor)
            predictor = BatchNormalization()(predictor)
            predictor = Bidirectional(
                LSTM(
                    self.LSTM_units_1,
                    activation="sigmoid",
                    return_sequences=True,
                    kernel_constraint=UnitNorm(axis=1),
                )
            )(predictor)
            predictor = BatchNormalization()(predictor)
            x_predicted_mean = TimeDistributed(
                Dense(self.input_shape[2]), name="vaep_prediction"
            )(predictor)

        # end-to-end autoencoder
        gmvaep = Model(
            inputs=x,
            outputs=(
                [x_decoded_mean, x_predicted_mean] if self.predictor else x_decoded_mean
            ),
            name="SEQ_2_SEQ_VAE",
        )

        def huber_loss(x_, x_decoded_mean_):
            huber = Huber(reduction="sum", delta=100.0)
            return self.input_shape[1:] * huber(x_, x_decoded_mean_)

        gmvaep.compile(
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

        return gmvaep
