# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

deep autoencoder models for unsupervised pose detection

"""

from typing import Any, Dict, Tuple
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
from deepof.model_utils import *
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfpl = tfp.layers


# noinspection PyDefaultArgument
class SEQ_2_SEQ_AE:
    """

        Simple sequence to sequence autoencoder implemented with tf.keras

            Parameters:
                -

            Returns:
                -

        """

    def __init__(
        self, architecture_hparams: Dict = {}, huber_delta: float = 100.0,
    ):
        self.hparams = self.get_hparams(architecture_hparams)
        self.CONV_filters = self.hparams["units_conv"]
        self.LSTM_units_1 = self.hparams["units_lstm"]
        self.LSTM_units_2 = int(self.hparams["units_lstm"] / 2)
        self.DENSE_1 = int(self.hparams["units_lstm"] / 2)
        self.DENSE_2 = self.hparams["units_dense2"]
        self.DROPOUT_RATE = self.hparams["dropout_rate"]
        self.ENCODING = self.hparams["encoding"]
        self.learn_rate = self.hparams["learning_rate"]
        self.delta = huber_delta

    @staticmethod
    def get_hparams(hparams):
        """Sets the default parameters for the model. Overwritable with a dictionary"""

        defaults = {
            "units_conv": 256,
            "units_lstm": 256,
            "units_dense2": 64,
            "dropout_rate": 0.25,
            "encoding": 16,
            "learning_rate": 1e-5,
        }

        for k, v in hparams.items():
            defaults[k] = v

        return defaults

    def get_layers(self, input_shape):
        """Instanciate all layers in the model"""

        # Encoder Layers
        Model_E0 = tf.keras.layers.Conv1D(
            filters=self.CONV_filters,
            kernel_size=5,
            strides=1,
            padding="causal",
            activation="elu",
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
            activation="elu",
            kernel_constraint=UnitNorm(axis=0),
            kernel_initializer=he_uniform(),
        )
        Model_E4 = Dense(
            self.DENSE_2,
            activation="elu",
            kernel_constraint=UnitNorm(axis=0),
            kernel_initializer=he_uniform(),
        )
        Model_E5 = Dense(
            self.ENCODING,
            activation="elu",
            kernel_constraint=UnitNorm(axis=1),
            activity_regularizer=uncorrelated_features_constraint(2, weightage=1.0),
            kernel_initializer=Orthogonal(),
        )

        # Decoder layers
        Model_D0 = DenseTranspose(Model_E5, activation="elu", output_dim=self.ENCODING,)
        Model_D1 = DenseTranspose(Model_E4, activation="elu", output_dim=self.DENSE_2,)
        Model_D2 = DenseTranspose(Model_E3, activation="elu", output_dim=self.DENSE_1,)
        Model_D3 = RepeatVector(input_shape[1])
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

        return (
            Model_E0,
            Model_E1,
            Model_E2,
            Model_E3,
            Model_E4,
            Model_E5,
            Model_D0,
            Model_D1,
            Model_D2,
            Model_D3,
            Model_D4,
            Model_D5,
        )

    def build(self, input_shape: tuple,) -> Tuple[Any, Any, Any]:
        """Builds the tf.keras model"""

        (
            Model_E0,
            Model_E1,
            Model_E2,
            Model_E3,
            Model_E4,
            Model_E5,
            Model_D0,
            Model_D1,
            Model_D2,
            Model_D3,
            Model_D4,
            Model_D5,
        ) = self.get_layers(input_shape)

        # Define and instantiate encoder
        encoder = Sequential(name="SEQ_2_SEQ_Encoder")
        encoder.add(Input(shape=input_shape[1:]))
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
        decoder.add(TimeDistributed(Dense(input_shape[2])))

        model = Sequential([encoder, decoder], name="SEQ_2_SEQ_AE")

        model.compile(
            loss=Huber(reduction="sum", delta=self.delta),
            optimizer=Nadam(lr=self.learn_rate, clipvalue=0.5,),
            metrics=["mae"],
        )

        return encoder, decoder, model


# noinspection PyDefaultArgument
class SEQ_2_SEQ_GMVAE:
    """

    Gaussian Mixture Variational Autoencoder for pose motif elucidation.

        Parameters:
            -

        Returns:
            -

    """

    def __init__(
        self,
        architecture_hparams: dict = {},
        loss: str = "ELBO+MMD",
        kl_warmup_epochs: int = 0,
        mmd_warmup_epochs: int = 0,
        number_of_components: int = 1,
        predictor: bool = True,
        overlap_loss: bool = False,
        entropy_reg_weight: float = 0.0,
        initialiser_iters: int = int(1e5),
        huber_delta: float = 100.0,
    ):
        self.hparams = self.get_hparams(architecture_hparams)
        self.batch_size = self.hparams["batch_size"]
        self.CONV_filters = self.hparams["units_conv"]
        self.LSTM_units_1 = self.hparams["units_lstm"]
        self.LSTM_units_2 = int(self.hparams["units_lstm"] / 2)
        self.DENSE_1 = int(self.hparams["units_lstm"] / 2)
        self.DENSE_2 = self.hparams["units_dense2"]
        self.DROPOUT_RATE = self.hparams["dropout_rate"]
        self.ENCODING = self.hparams["encoding"]
        self.learn_rate = self.hparams["learning_rate"]
        self.loss = loss
        self.prior = "standard_normal"
        self.kl_warmup = kl_warmup_epochs
        self.mmd_warmup = mmd_warmup_epochs
        self.number_of_components = number_of_components
        self.predictor = predictor
        self.overlap_loss = overlap_loss
        self.entropy_reg_weight = entropy_reg_weight
        self.initialiser_iters = initialiser_iters
        self.delta = huber_delta

        assert (
            "ELBO" in self.loss or "MMD" in self.loss
        ), "loss must be one of ELBO, MMD or ELBO+MMD (default)"

    @property
    def prior(self):
        return self._prior

    def get_prior(self):
        """Sets the Variational Autoencoder prior distribution"""

        if self.prior == "standard_normal":
            init_means = far_away_uniform_initialiser(
                shape=(self.number_of_components, self.ENCODING),
                minval=0,
                maxval=5,
                iters=self.initialiser_iters,
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

        else:
            raise NotImplementedError(
                "Gaussian Mixtures are currently the only supported prior"
            )

    @staticmethod
    def get_hparams(params: Dict) -> Dict:
        """Sets the default parameters for the model. Overwritable with a dictionary"""

        defaults = {
            "batch_size": 512,
            "units_conv": 256,
            "units_lstm": 256,
            "units_dense2": 64,
            "dropout_rate": 0.25,
            "encoding": 16,
            "learning_rate": 1e-3,
        }

        for k, v in params.items():
            defaults[k] = v

        return defaults

    def get_layers(self, input_shape):
        """Instanciate all layers in the model"""

        # Encoder Layers
        Model_E0 = tf.keras.layers.Conv1D(
            filters=self.CONV_filters,
            kernel_size=5,
            strides=1,
            padding="causal",
            activation="elu",
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
            activation="elu",
            kernel_constraint=UnitNorm(axis=0),
            kernel_initializer=he_uniform(),
            use_bias=False,
        )
        Model_E4 = Dense(
            self.DENSE_2,
            activation="elu",
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
            activation="elu",
            kernel_initializer=he_uniform(),
            use_bias=False,
        )
        Model_D2 = Dense(
            self.DENSE_1,
            activation="elu",
            kernel_initializer=he_uniform(),
            use_bias=False,
        )
        Model_D3 = RepeatVector(input_shape[1])
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
        Model_P1 = Dense(
            self.DENSE_1,
            activation="elu",
            kernel_initializer=he_uniform(),
            use_bias=False,
        )
        Model_P2 = Bidirectional(
            LSTM(
                self.LSTM_units_1,
                activation="tanh",
                recurrent_activation="sigmoid",
                return_sequences=True,
                kernel_constraint=UnitNorm(axis=1),
                use_bias=False,
            )
        )
        Model_P3 = Bidirectional(
            LSTM(
                self.LSTM_units_1,
                activation="tanh",
                recurrent_activation="sigmoid",
                return_sequences=True,
                kernel_constraint=UnitNorm(axis=1),
                use_bias=False,
            )
        )

        return (
            Model_E0,
            Model_E1,
            Model_E2,
            Model_E3,
            Model_E4,
            Model_B1,
            Model_B2,
            Model_B3,
            Model_B4,
            Model_D1,
            Model_D2,
            Model_D3,
            Model_D4,
            Model_D5,
            Model_P1,
            Model_P2,
            Model_P3,
        )

    def build(self, input_shape: Tuple):

        # Instanciate prior
        self.get_prior()

        # Get model layers
        (
            Model_E0,
            Model_E1,
            Model_E2,
            Model_E3,
            Model_E4,
            Model_B1,
            Model_B2,
            Model_B3,
            Model_B4,
            Model_D1,
            Model_D2,
            Model_D3,
            Model_D4,
            Model_D5,
            Model_P1,
            Model_P2,
            Model_P3,
        ) = self.get_layers(input_shape)

        # Define and instantiate encoder
        x = Input(shape=input_shape[1:])
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

        encoding_shuffle = MCDropout(self.DROPOUT_RATE)(encoder)
        z_cat = Dense(self.number_of_components, activation="softmax",)(
            encoding_shuffle
        )
        z_cat = Entropy_regulariser(self.entropy_reg_weight)(z_cat)
        z_gauss = Dense(
            tfpl.IndependentNormal.params_size(
                self.ENCODING * self.number_of_components
            ),
            activation=None,
        )(encoder)

        z_gauss = Reshape([2 * self.ENCODING, self.number_of_components])(z_gauss)

        # Identity layer controlling for dead neurons in the Gaussian Mixture posterior
        z_gauss = Dead_neuron_control()(z_gauss)

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
            Dense(input_shape[2]), name="vaep_reconstruction"
        )(generator)

        if self.predictor > 0:
            # Define and instantiate predictor
            predictor = Dense(
                self.DENSE_2, activation="elu", kernel_initializer=he_uniform()
            )(z)
            predictor = BatchNormalization()(predictor)
            predictor = Model_P1(predictor)
            predictor = BatchNormalization()(predictor)
            predictor = RepeatVector(input_shape[1])(predictor)
            predictor = Model_P2(predictor)
            predictor = BatchNormalization()(predictor)
            predictor = Model_P3(predictor)
            predictor = BatchNormalization()(predictor)
            x_predicted_mean = TimeDistributed(
                Dense(input_shape[2]), name="vaep_prediction"
            )(predictor)

        # end-to-end autoencoder
        encoder = Model(x, z, name="SEQ_2_SEQ_VEncoder")
        grouper = Model(x, z_cat, name="Deep_Gaussian_Mixture_clustering")
        # noinspection PyUnboundLocalVariable
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
        _x_decoded_mean = TimeDistributed(Dense(input_shape[2]))(_generator)
        generator = Model(g, _x_decoded_mean, name="SEQ_2_SEQ_VGenerator")

        def huber_loss(x_, x_decoded_mean_):  # pragma: no cover
            """Computes huber loss with a fixed delta"""

            huber = Huber(reduction="sum", delta=self.delta)
            return input_shape[1:] * huber(x_, x_decoded_mean_)

        gmvaep.compile(
            loss=huber_loss,
            optimizer=Nadam(lr=self.learn_rate, clipvalue=0.5,),
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

    @prior.setter
    def prior(self, value):
        self._prior = value


# TODO:
#       - Investigate posterior collapse (L1 as kernel/activity regulariser does not work)
#       - Random horizontal flip for data augmentation
#       - Align first frame and untamper sliding window (reduce window stride)
#       - design clustering-conscious hyperparameter tuning pipeline
#       - execute the pipeline ;)
