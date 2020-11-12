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
from tensorflow.keras.losses import BinaryCrossentropy, Huber
from tensorflow.keras.optimizers import Nadam
import deepof.model_utils
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
        self, architecture_hparams: Dict = {}, huber_delta: float = 1.0,
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
            "learning_rate": 1e-3,
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
            activity_regularizer=deepof.model_utils.uncorrelated_features_constraint(
                2, weightage=1.0
            ),
            kernel_initializer=Orthogonal(),
        )

        # Decoder layers
        Model_D0 = deepof.model_utils.DenseTranspose(
            Model_E5, activation="elu", output_dim=self.ENCODING,
        )
        Model_D1 = deepof.model_utils.DenseTranspose(
            Model_E4, activation="elu", output_dim=self.DENSE_2,
        )
        Model_D2 = deepof.model_utils.DenseTranspose(
            Model_E3, activation="elu", output_dim=self.DENSE_1,
        )
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
            loss=Huber(delta=self.delta),
            optimizer=Nadam(lr=self.learn_rate, clipvalue=0.5,),
            metrics=["mae"],
        )

        model.build(input_shape)

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
        batch_size: int = 256,
        compile: bool = True,
        entropy_reg_weight: float = 0.0,
        huber_delta: float = 1.0,
        initialiser_iters: int = int(1e4),
        kl_warmup_epochs: int = 0,
        loss: str = "ELBO+MMD",
        mmd_warmup_epochs: int = 0,
        number_of_components: int = 1,
        overlap_loss: bool = False,
        phenotype_prediction: float = 0.0,
        predictor: float = 0.0,
    ):
        self.hparams = self.get_hparams(architecture_hparams)
        self.batch_size = batch_size
        self.CONV_filters = self.hparams["units_conv"]
        self.LSTM_units_1 = self.hparams["units_lstm"]
        self.LSTM_units_2 = int(self.hparams["units_lstm"] / 2)
        self.DENSE_1 = int(self.hparams["units_lstm"] / 2)
        self.DENSE_2 = self.hparams["units_dense2"]
        self.DROPOUT_RATE = self.hparams["dropout_rate"]
        self.ENCODING = self.hparams["encoding"]
        self.learn_rate = self.hparams["learning_rate"]
        self.compile = compile
        self.delta = huber_delta
        self.entropy_reg_weight = entropy_reg_weight
        self.initialiser_iters = initialiser_iters
        self.kl_warmup = kl_warmup_epochs
        self.loss = loss
        self.mmd_warmup = mmd_warmup_epochs
        self.number_of_components = number_of_components
        self.overlap_loss = overlap_loss
        self.phenotype_prediction = phenotype_prediction
        self.predictor = predictor
        self.prior = "standard_normal"

        assert (
            "ELBO" in self.loss or "MMD" in self.loss
        ), "loss must be one of ELBO, MMD or ELBO+MMD (default)"

    @property
    def prior(self):
        """Property to set the value of the prior
        once the class is instanciated"""

        return self._prior

    def get_prior(self):
        """Sets the Variational Autoencoder prior distribution"""

        if self.prior == "standard_normal":
            init_means = deepof.model_utils.far_away_uniform_initialiser(
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

        else:  # pragma: no cover
            raise NotImplementedError(
                "Gaussian Mixtures are currently the only supported prior"
            )

    @staticmethod
    def get_hparams(params: Dict) -> Dict:
        """Sets the default parameters for the model. Overwritable with a dictionary"""

        defaults = {
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

        # Predictor layers
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

        # Phenotype classification layers
        Model_PC1 = Dense(
            self.number_of_components, activation="elu", kernel_initializer=he_uniform()
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
            Model_PC1,
        )

    def build(self, input_shape: Tuple):
        """Builds the tf.keras model"""

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
            Model_PC1,
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

        encoding_shuffle = deepof.model_utils.MCDropout(self.DROPOUT_RATE)(encoder)
        z_cat = Dense(self.number_of_components, activation="softmax",)(
            encoding_shuffle
        )
        z_cat = deepof.model_utils.Entropy_regulariser(self.entropy_reg_weight)(z_cat)
        z_gauss = Dense(
            deepof.model_utils.tfpl.IndependentNormal.params_size(
                self.ENCODING * self.number_of_components
            ),
            activation=None,
        )(encoder)

        z_gauss = Reshape([2 * self.ENCODING, self.number_of_components])(z_gauss)

        # Identity layer controlling for dead neurons in the Gaussian Mixture posterior
        z_gauss = deepof.model_utils.Dead_neuron_control()(z_gauss)

        if self.overlap_loss:
            z_gauss = deepof.model_utils.Gaussian_mixture_overlap(
                self.ENCODING, self.number_of_components, loss=self.overlap_loss,
            )(z_gauss)

        z = deepof.model_utils.tfpl.DistributionLambda(
            lambda gauss: tfd.mixture.Mixture(
                cat=tfd.categorical.Categorical(probs=gauss[0],),
                components=[
                    tfd.Independent(
                        tfd.Normal(
                            loc=gauss[1][..., : self.ENCODING, k],
                            scale=softplus(gauss[1][..., self.ENCODING:, k]),
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

            kl_beta = deepof.model_utils.K.variable(1.0, name="kl_beta")
            kl_beta._trainable = False
            if self.kl_warmup:
                kl_warmup_callback = LambdaCallback(
                    on_epoch_begin=lambda epoch, logs: deepof.model_utils.K.set_value(
                        kl_beta, deepof.model_utils.K.min([epoch / self.kl_warmup, 1])
                    )
                )

            z = deepof.model_utils.KLDivergenceLayer(self.prior, weight=kl_beta)(z)

        mmd_warmup_callback = False
        if "MMD" in self.loss:

            mmd_beta = deepof.model_utils.K.variable(1.0, name="mmd_beta")
            mmd_beta._trainable = False
            if self.mmd_warmup:
                mmd_warmup_callback = LambdaCallback(
                    on_epoch_begin=lambda epoch, logs: deepof.model_utils.K.set_value(
                        mmd_beta, deepof.model_utils.K.min([epoch / self.mmd_warmup, 1])
                    )
                )

            z = deepof.model_utils.MMDiscrepancyLayer(
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

        model_outs = [x_decoded_mean]
        model_losses = [Huber(delta=self.delta, reduction="sum")]
        loss_weights = [1.0]

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

            model_outs.append(x_predicted_mean)
            model_losses.append(Huber(delta=self.delta, reduction="sum"))
            loss_weights.append(self.predictor)

        if self.phenotype_prediction > 0:
            pheno_pred = Model_PC1(z)
            pheno_pred = Dense(1, activation="sigmoid", name="phenotype_prediction")(pheno_pred)

            model_outs.append(pheno_pred)
            model_losses.append(BinaryCrossentropy())
            loss_weights.append(self.phenotype_prediction)

        # end-to-end autoencoder
        encoder = Model(x, z, name="SEQ_2_SEQ_VEncoder")
        grouper = Model(x, z_cat, name="Deep_Gaussian_Mixture_clustering")
        # noinspection PyUnboundLocalVariable

        gmvaep = Model(inputs=x, outputs=model_outs, name="SEQ_2_SEQ_GMVAE",)

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

        if self.compile:
            gmvaep.compile(
                loss=model_losses,
                optimizer=Nadam(lr=self.learn_rate, clipvalue=0.5,),
                metrics=["mae"],
                loss_weights=loss_weights,
            )

        gmvaep.build(input_shape)

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
#       - Check KL weight in the overal loss function! Are we scaling the loss components correctly?
#       - Check batch and event shapes of all distributions involved. Incorrect shapes (batch >1) could bring
#         problems with the KL.
#       - Check merge mode in LSTM layers. Maybe we can drastically reduce model size!
#       - Check usefulness of stateful sequential layers! (stateful=True in the LSTMs)
#       - Investigate posterior collapse (L1 as kernel/activity regulariser does not work)
#       - design clustering-conscious hyperparameter tuning pipeline
#       - execute the pipeline ;)
