# @author lucasmiranda42

from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.activations import softplus
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.constraints import UnitNorm
from tensorflow.keras.initializers import he_uniform, Orthogonal
from tensorflow.keras.layers import BatchNormalization, Bidirectional
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.layers import RepeatVector, Reshape, TimeDistributed
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Nadam
from source.model_utils import *
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfpl = tfp.layers


class SEQ_2_SEQ_AE:
    def __init__(
        self,
        input_shape,
        units_conv=256,
        units_lstm=256,
        units_dense2=64,
        dropout_rate=0.25,
        encoding=16,
        learning_rate=1e-5,
    ):
        self.input_shape = input_shape
        self.CONV_filters = units_conv
        self.LSTM_units_1 = units_lstm
        self.LSTM_units_2 = int(units_lstm / 2)
        self.DENSE_1 = int(units_lstm / 2)
        self.DENSE_2 = units_dense2
        self.DROPOUT_RATE = dropout_rate
        self.ENCODING = encoding
        self.learn_rate = learning_rate

    def build(self):
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
                recurrent_activation="sigmoid",
                return_sequences=True,
                kernel_constraint=UnitNorm(axis=0),
            )
        )
        Model_E2 = Bidirectional(
            LSTM(
                self.LSTM_units_2,
                activation="tanh",
                recurrent_activation="sigmoid",
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
        Model_E5 = Dense(
            self.ENCODING,
            activation="relu",
            kernel_constraint=UnitNorm(axis=1),
            activity_regularizer=UncorrelatedFeaturesConstraint(3, weightage=1.0),
            kernel_initializer=Orthogonal(),
        )

        # Decoder layers
        Model_D0 = DenseTranspose(
            Model_E5, activation="relu", output_dim=self.ENCODING,
        )
        Model_D1 = DenseTranspose(Model_E4, activation="relu", output_dim=self.DENSE_2,)
        Model_D2 = DenseTranspose(Model_E3, activation="relu", output_dim=self.DENSE_1,)
        Model_D3 = RepeatVector(self.input_shape[1])
        Model_D4 = Bidirectional(
            LSTM(
                self.LSTM_units_1,
                activation="tanh",
                recurrent_activation="sigmoid",
                return_sequences=True,
                kernel_constraint=UnitNorm(axis=1),
            )
        )
        Model_D5 = Bidirectional(
            LSTM(
                self.LSTM_units_1,
                activation="sigmoid",
                recurrent_activation="sigmoid",
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
        encoder.add(Dropout(self.DROPOUT_RATE))
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
            optimizer=Nadam(lr=self.learn_rate, decay=1e-2, clipvalue=0.5,),
            metrics=["mae"],
        )

        return encoder, decoder, model


class SEQ_2_SEQ_GMVAE:
    def __init__(
        self,
        input_shape,
        batch_size=512,
        units_conv=256,
        units_lstm=256,
        units_dense2=64,
        dropout_rate=0.25,
        encoding=16,
        learning_rate=1e-3,
        loss="ELBO+MMD",
        kl_warmup_epochs=0,
        mmd_warmup_epochs=0,
        prior="standard_normal",
        number_of_components=1,
        predictor=True,
        overlap_loss=False,
        entropy_reg_weight=1.0,
    ):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.CONV_filters = units_conv
        self.LSTM_units_1 = units_lstm
        self.LSTM_units_2 = int(units_lstm / 2)
        self.DENSE_1 = int(units_lstm / 2)
        self.DENSE_2 = units_dense2
        self.DROPOUT_RATE = dropout_rate
        self.ENCODING = encoding
        self.learn_rate = learning_rate
        self.loss = loss
        self.prior = prior
        self.kl_warmup = kl_warmup_epochs
        self.mmd_warmup = mmd_warmup_epochs
        self.number_of_components = number_of_components
        self.predictor = predictor
        self.overlap_loss = overlap_loss
        self.entropy_reg_weight = entropy_reg_weight

        if self.prior == "standard_normal":

            init_means = far_away_uniform_initialiser(
                [self.number_of_components, self.ENCODING], minval=0, maxval=15
            )

            self.prior = tfd.mixture.Mixture(
                cat=tfd.categorical.Categorical(
                    probs=tf.ones(self.number_of_components) / self.number_of_components
                ),
                components=[
                    tfd.Independent(
                        tfd.Normal(loc=init_means[k], scale=1,),
                        reinterpreted_batch_ndims=1,
                    )
                    for k in range(self.number_of_components)
                ],
            )

        assert (
            "ELBO" in self.loss or "MMD" in self.loss
        ), "loss must be one of ELBO, MMD or ELBO+MMD (default)"

    def build(self):
        # Encoder Layers
        Model_E0 = tf.keras.layers.Conv1D(
            filters=self.CONV_filters,
            kernel_size=5,
            strides=1,
            padding="causal",
            activation="relu",
            kernel_initializer=he_uniform(),
            use_bias=False,
        )
        Model_E1 = Bidirectional(
            LSTM(
                self.LSTM_units_1,
                activation="tanh",
                recurrent_activation="sigmoid",
                return_sequences=True,
                kernel_constraint=UnitNorm(axis=0),
                use_bias=False,
            )
        )
        Model_E2 = Bidirectional(
            LSTM(
                self.LSTM_units_2,
                activation="tanh",
                recurrent_activation="sigmoid",
                return_sequences=False,
                kernel_constraint=UnitNorm(axis=0),
                use_bias=False,
            )
        )
        Model_E3 = Dense(
            self.DENSE_1,
            activation="relu",
            kernel_constraint=UnitNorm(axis=0),
            kernel_initializer=he_uniform(),
            use_bias=False,
        )
        Model_E4 = Dense(
            self.DENSE_2,
            activation="relu",
            kernel_constraint=UnitNorm(axis=0),
            kernel_initializer=he_uniform(),
            use_bias=False,
        )

        # Decoder layers
        Model_B1 = BatchNormalization()
        Model_B2 = BatchNormalization()
        Model_B3 = BatchNormalization()
        Model_B4 = BatchNormalization()
        Model_D1 = Dense(
            self.DENSE_2,
            activation="relu",
            kernel_initializer=he_uniform(),
            use_bias=False,
        )
        Model_D2 = Dense(
            self.DENSE_1,
            activation="relu",
            kernel_initializer=he_uniform(),
            use_bias=False,
        )
        Model_D3 = RepeatVector(self.input_shape[1])
        Model_D4 = Bidirectional(
            LSTM(
                self.LSTM_units_2,
                activation="tanh",
                recurrent_activation="sigmoid",
                return_sequences=True,
                kernel_constraint=UnitNorm(axis=1),
                use_bias=False,
            )
        )
        Model_D5 = Bidirectional(
            LSTM(
                self.LSTM_units_1,
                activation="sigmoid",
                recurrent_activation="sigmoid",
                return_sequences=True,
                kernel_constraint=UnitNorm(axis=1),
                use_bias=False,
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
        encoder = Dropout(self.DROPOUT_RATE)(encoder)
        encoder = Model_E4(encoder)
        encoder = BatchNormalization()(encoder)

        z_cat = Dense(self.number_of_components, activation="softmax",)(encoder)
        z_cat = Entropy_regulariser(self.entropy_reg_weight)(z_cat)
        z_gauss = Dense(
            tfpl.IndependentNormal.params_size(
                self.ENCODING * self.number_of_components
            ),
            activation=None,
        )(encoder)

        z_gauss = Reshape([2 * self.ENCODING, self.number_of_components])(z_gauss)

        if self.overlap_loss:
            z_gauss = Gaussian_mixture_overlap(
                self.ENCODING, self.number_of_components, loss=self.overlap_loss,
            )(z_gauss)

        z = tfpl.DistributionLambda(
            lambda gauss: tfd.mixture.Mixture(
                cat=tfd.categorical.Categorical(probs=gauss[0],),
                components=[
                    tfd.Independent(
                        tfd.Normal(
                            loc=gauss[1][..., : self.ENCODING, k],
                            scale=softplus(gauss[1][..., self.ENCODING :, k]),
                        ),
                        reinterpreted_batch_ndims=1,
                    )
                    for k in range(self.number_of_components)
                ],
            ),
            activity_regularizer=UncorrelatedFeaturesConstraint(3, weightage=1.0),
        )([z_cat, z_gauss])

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

            z = MMDiscrepancyLayer(
                batch_size=self.batch_size, prior=self.prior, beta=mmd_beta
            )(z)

        # Identity layer controlling clustering and latent space statistics
        z = Latent_space_control(loss=self.overlap_loss)(z, z_gauss, z_cat)

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

        if self.predictor > 0:
            # Define and instantiate predictor
            predictor = Dense(
                self.DENSE_2, activation="relu", kernel_initializer=he_uniform()
            )(z)
            predictor = BatchNormalization()(predictor)
            predictor = Dense(
                self.DENSE_1,
                activation="relu",
                kernel_initializer=he_uniform(),
                use_bias=False,
            )(predictor)
            predictor = BatchNormalization()(predictor)
            predictor = RepeatVector(self.input_shape[1])(predictor)
            predictor = Bidirectional(
                LSTM(
                    self.LSTM_units_1,
                    activation="tanh",
                    recurrent_activation="sigmoid",
                    return_sequences=True,
                    kernel_constraint=UnitNorm(axis=1),
                    use_bias=False,
                )
            )(predictor)
            predictor = BatchNormalization()(predictor)
            predictor = Bidirectional(
                LSTM(
                    self.LSTM_units_1,
                    activation="tanh",
                    recurrent_activation="sigmoid",
                    return_sequences=True,
                    kernel_constraint=UnitNorm(axis=1),
                    use_bias=False,
                )
            )(predictor)
            predictor = BatchNormalization()(predictor)
            x_predicted_mean = TimeDistributed(
                Dense(self.input_shape[2]), name="vaep_prediction"
            )(predictor)

        # end-to-end autoencoder
        encoder = Model(x, z, name="SEQ_2_SEQ_VEncoder")
        grouper = Model(x, z_cat, name="Deep_Gaussian_Mixture_clustering")
        gmvaep = Model(
            inputs=x,
            outputs=(
                [x_decoded_mean, x_predicted_mean]
                if self.predictor > 0
                else x_decoded_mean
            ),
            name="SEQ_2_SEQ_VAE",
        )

        # Build generator as a separate entity
        g = Input(shape=self.ENCODING)
        _generator = Model_D1(g)
        _generator = Model_B1(_generator)
        _generator = Model_D2(_generator)
        _generator = Model_B2(_generator)
        _generator = Model_D3(_generator)
        _generator = Model_D4(_generator)
        _generator = Model_B3(_generator)
        _generator = Model_D5(_generator)
        _generator = Model_B4(_generator)
        _x_decoded_mean = TimeDistributed(Dense(self.input_shape[2]))(_generator)
        generator = Model(g, _x_decoded_mean, name="SEQ_2_SEQ_VGenerator")

        def huber_loss(x_, x_decoded_mean_):
            huber = Huber(reduction="sum", delta=100.0)
            return self.input_shape[1:] * huber(x_, x_decoded_mean_)

        gmvaep.compile(
            loss=huber_loss,
            optimizer=Nadam(lr=self.learn_rate, decay=1e-2),
            metrics=["mae"],
            loss_weights=([1, self.predictor] if self.predictor > 0 else [1]),
        )

        return (
            encoder,
            generator,
            grouper,
            gmvaep,
            kl_warmup_callback,
            mmd_warmup_callback,
        )


# TODO:
#       - Investigate posterior collapse
#       - Learning rate scheduler (for faster / better convergence)
#       - data augmentation with rotation / always align fist frame with an axis
#       - design clustering-conscious hyperparameter tuning pipeline
#       - execute the pipeline ;)
