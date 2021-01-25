# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

deep autoencoder models for unsupervised pose detection

"""

from typing import Any, Dict, Tuple
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
import deepof.model_utils
import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions
tfpl = tfp.layers


# noinspection PyDefaultArgument
class SEQ_2_SEQ_AE:
    """  Simple sequence to sequence autoencoder implemented with tf.keras """

    def __init__(
        self,
        architecture_hparams: Dict = {},
        huber_delta: float = 1.0,
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
            Model_E5,
            activation="elu",
            output_dim=self.ENCODING,
        )
        Model_D1 = deepof.model_utils.DenseTranspose(
            Model_E4,
            activation="elu",
            output_dim=self.DENSE_2,
        )
        Model_D2 = deepof.model_utils.DenseTranspose(
            Model_E3,
            activation="elu",
            output_dim=self.DENSE_1,
        )
        Model_D3 = RepeatVector(input_shape[1])
        Model_D4 = Bidirectional(
            LSTM(
                self.LSTM_units_1,
                activation="tanh",
                recurrent_activation="sigmoid",
                return_sequences=True,
                # kernel_constraint=UnitNorm(axis=1),
            )
        )
        Model_D5 = Bidirectional(
            LSTM(
                self.LSTM_units_1,
                activation="sigmoid",
                recurrent_activation="sigmoid",
                return_sequences=True,
                # kernel_constraint=UnitNorm(axis=1),
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

    def build(
        self,
        input_shape: tuple,
    ) -> Tuple[Any, Any, Any]:
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
            optimizer=Nadam(
                lr=self.learn_rate,
                clipvalue=0.5,
            ),
            metrics=["mae"],
        )

        model.build(input_shape)

        return encoder, decoder, model


# noinspection PyDefaultArgument
class SEQ_2_SEQ_GMVAE:
    """  Gaussian Mixture Variational Autoencoder for pose motif elucidation.  """

    def __init__(
        self,
        architecture_hparams: dict = {},
        batch_size: int = 256,
        compile_model: bool = True,
        encoding: int = 16,
        entropy_reg_weight: float = 0.0,
        huber_delta: float = 1.0,
        initialiser_iters: int = int(1),
        kl_warmup_epochs: int = 20,
        loss: str = "ELBO+MMD",
        mmd_warmup_epochs: int = 20,
        montecarlo_kl: int = 1,
        neuron_control: bool = False,
        number_of_components: int = 1,
        overlap_loss: float = False,
        phenotype_prediction: float = 0.0,
        predictor: float = 0.0,
    ):
        self.hparams = self.get_hparams(architecture_hparams)
        self.batch_size = batch_size
        self.bidirectional_merge = self.hparams["bidirectional_merge"]
        self.CONV_filters = self.hparams["units_conv"]
        self.DENSE_1 = int(self.hparams["units_lstm"] / 2)
        self.DENSE_2 = self.hparams["units_dense2"]
        self.DROPOUT_RATE = self.hparams["dropout_rate"]
        self.ENCODING = encoding
        self.LSTM_units_1 = self.hparams["units_lstm"]
        self.LSTM_units_2 = int(self.hparams["units_lstm"] / 2)
        self.clipvalue = self.hparams["clipvalue"]
        self.dense_activation = self.hparams["dense_activation"]
        self.dense_layers_per_branch = self.hparams["dense_layers_per_branch"]
        self.learn_rate = self.hparams["learning_rate"]
        self.lstm_unroll = True
        self.compile = compile_model
        self.delta = huber_delta
        self.entropy_reg_weight = entropy_reg_weight
        self.initialiser_iters = initialiser_iters
        self.kl_warmup = kl_warmup_epochs
        self.loss = loss
        self.mc_kl = montecarlo_kl
        self.mmd_warmup = mmd_warmup_epochs
        self.neuron_control = neuron_control
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
            # init_means = deepof.model_utils.far_away_uniform_initialiser(
            #     shape=(self.number_of_components, self.ENCODING),
            #     minval=0,
            #     maxval=5,
            #     iters=self.initialiser_iters,
            # )

            self.prior = tfd.MixtureSameFamily(
                mixture_distribution=tfd.categorical.Categorical(
                    probs=tf.ones(self.number_of_components) / self.number_of_components
                ),
                components_distribution=tfd.MultivariateNormalDiag(
                    loc=tf.Variable(
                        tf.random.normal(
                            [self.number_of_components, self.ENCODING],
                            name="prior_means",
                        )
                    ),
                    scale_diag=tfp.util.TransformedVariable(
                        tf.ones([self.number_of_components, self.ENCODING]),
                        tfb.Softplus(),
                        name="prior_scales",
                    ),
                ),
            )

        else:  # pragma: no cover
            raise NotImplementedError(
                "Gaussian Mixtures are currently the only supported prior"
            )

    @staticmethod
    def get_hparams(params: Dict) -> Dict:
        """Sets the default parameters for the model. Overwritable with a dictionary"""

        defaults = {
            "bidirectional_merge": "ave",
            "clipvalue": 1.0,
            "dense_activation": "relu",
            "dense_layers_per_branch": 1,
            "dropout_rate": 1e-3,
            "learning_rate": 1e-3,
            "units_conv": 160,
            "units_dense2": 120,
            "units_lstm": 300,
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
            activation=self.dense_activation,
            kernel_initializer=he_uniform(),
            use_bias=True,
        )
        Model_E1 = Bidirectional(
            LSTM(
                self.LSTM_units_1,
                activation="tanh",
                recurrent_activation="sigmoid",
                return_sequences=True,
                unroll=self.lstm_unroll,
                # kernel_constraint=UnitNorm(axis=0),
                use_bias=True,
            ),
            merge_mode=self.bidirectional_merge,
        )
        Model_E2 = Bidirectional(
            LSTM(
                self.LSTM_units_2,
                activation="tanh",
                recurrent_activation="sigmoid",
                return_sequences=False,
                unroll=self.lstm_unroll,
                # kernel_constraint=UnitNorm(axis=0),
                use_bias=True,
            ),
            merge_mode=self.bidirectional_merge,
        )
        Model_E3 = Dense(
            self.DENSE_1,
            activation=self.dense_activation,
            # kernel_constraint=UnitNorm(axis=0),
            kernel_initializer=he_uniform(),
            use_bias=True,
        )

        Model_E4 = [
            Dense(
                self.DENSE_2,
                activation=self.dense_activation,
                # kernel_constraint=UnitNorm(axis=0),
                kernel_initializer=he_uniform(),
                use_bias=True,
            )
            for _ in range(self.dense_layers_per_branch)
        ]

        # Decoder layers
        Model_B1 = BatchNormalization()
        Model_B2 = BatchNormalization()
        Model_B3 = BatchNormalization()
        Model_B4 = BatchNormalization()
        Model_D1 = [
            Dense(
                self.DENSE_2,
                activation=self.dense_activation,
                kernel_initializer=he_uniform(),
                use_bias=True,
            )
            for _ in range(self.dense_layers_per_branch)
        ]
        Model_D2 = Dense(
            self.DENSE_1,
            activation=self.dense_activation,
            kernel_initializer=he_uniform(),
            use_bias=True,
        )
        Model_D3 = RepeatVector(input_shape[1])
        Model_D4 = Bidirectional(
            LSTM(
                self.LSTM_units_2,
                activation="tanh",
                recurrent_activation="sigmoid",
                return_sequences=True,
                unroll=self.lstm_unroll,
                # kernel_constraint=UnitNorm(axis=1),
                use_bias=True,
            ),
            merge_mode=self.bidirectional_merge,
        )
        Model_D5 = Bidirectional(
            LSTM(
                self.LSTM_units_1,
                activation="sigmoid",
                recurrent_activation="sigmoid",
                return_sequences=True,
                unroll=self.lstm_unroll,
                # kernel_constraint=UnitNorm(axis=1),
                use_bias=True,
            ),
            merge_mode=self.bidirectional_merge,
        )

        # Predictor layers
        Model_P1 = Dense(
            self.DENSE_1,
            activation=self.dense_activation,
            kernel_initializer=he_uniform(),
            use_bias=True,
        )
        Model_P2 = Bidirectional(
            LSTM(
                self.LSTM_units_1,
                activation="tanh",
                recurrent_activation="sigmoid",
                return_sequences=True,
                unroll=self.lstm_unroll,
                # kernel_constraint=UnitNorm(axis=1),
                use_bias=True,
            ),
            merge_mode=self.bidirectional_merge,
        )
        Model_P3 = Bidirectional(
            LSTM(
                self.LSTM_units_1,
                activation="tanh",
                recurrent_activation="sigmoid",
                return_sequences=True,
                unroll=self.lstm_unroll,
                # kernel_constraint=UnitNorm(axis=1),
                use_bias=True,
            ),
            merge_mode=self.bidirectional_merge,
        )

        # Phenotype classification layers
        Model_PC1 = Dense(
            self.number_of_components,
            activation=self.dense_activation,
            kernel_initializer=he_uniform(),
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

        print(input_shape)
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
        encoder = Sequential(Model_E4)(encoder)
        encoder = BatchNormalization()(encoder)

        # encoding_shuffle = deepof.model_utils.MCDropout(self.DROPOUT_RATE)(encoder)
        z_cat = Dense(
            self.number_of_components,
            activation="softmax",
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01),
        )(encoder)

        if self.entropy_reg_weight > 0:
            z_cat = deepof.model_utils.Entropy_regulariser(self.entropy_reg_weight)(
                z_cat
            )

        z_gauss = Dense(
            tfpl.IndependentNormal.params_size(
                self.ENCODING * self.number_of_components
            ),
            activation=None,
        )(
            encoder
        )  # REMOVE BIAS FROM HERE! WHAT's THE POINT?

        z_gauss = Reshape([2 * self.ENCODING, self.number_of_components])(z_gauss)

        # Identity layer controlling for dead neurons in the Gaussian Mixture posterior
        if self.neuron_control:
            z_gauss = deepof.model_utils.Dead_neuron_control()(z_gauss)

        if self.overlap_loss:
            z_gauss = deepof.model_utils.Gaussian_mixture_overlap(
                self.ENCODING,
                self.number_of_components,
                loss=self.overlap_loss,
            )(z_gauss)

        z = tfpl.DistributionLambda(
            lambda gauss: tfd.mixture.Mixture(
                cat=tfd.categorical.Categorical(
                    probs=gauss[0],
                ),
                components=[
                    tfd.Independent(
                        tfd.Normal(
                            loc=gauss[1][..., : self.ENCODING, k],
                            scale=softplus(gauss[1][..., self.ENCODING :, k]) + 1e-5,
                        ),
                        reinterpreted_batch_ndims=1,
                    )
                    for k in range(self.number_of_components)
                ],
            ),
            convert_to_tensor_fn="sample",
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

            # noinspection PyCallingNonCallable
            z = deepof.model_utils.KLDivergenceLayer(
                self.prior,
                test_points_fn=lambda q: q.sample(self.mc_kl),
                test_points_reduce_axis=0,
                weight=kl_beta,
            )(z)

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
        g = Input(shape=self.ENCODING)
        generator = Sequential(Model_D1)(g)
        generator = Model_B1(generator)
        generator = Model_D2(generator)
        generator = Model_B2(generator)
        generator = Model_D3(generator)
        generator = Model_D4(generator)
        generator = Model_B3(generator)
        generator = Model_D5(generator)
        generator = Model_B4(generator)
        generator = Dense(tfpl.IndependentNormal.params_size(input_shape[2:]))(
            generator
        )
        x_decoded_mean = tfpl.IndependentNormal(
            event_shape=input_shape[2:],
            convert_to_tensor_fn=tfp.distributions.Distribution.mean,
            name="vae_reconstruction",
        )(generator)

        # define individual branches as models
        encoder = Model(x, z, name="SEQ_2_SEQ_VEncoder")
        generator = Model(g, x_decoded_mean, name="vae_reconstruction")

        def log_loss(x_true, p_x_q_given_z):
            """Computes the negative log likelihood of the data given
            the output distribution"""
            return -tf.reduce_sum(p_x_q_given_z.log_prob(x_true))

        model_outs = [generator(encoder.outputs)]
        model_losses = [log_loss]
        model_metrics = {"vae_reconstruction": ["mae", "mse"]}
        loss_weights = [1.0]

        if self.predictor > 0:
            # Define and instantiate predictor
            predictor = Dense(
                self.DENSE_2,
                activation=self.dense_activation,
                kernel_initializer=he_uniform(),
            )(z)
            predictor = BatchNormalization()(predictor)
            predictor = Model_P1(predictor)
            predictor = BatchNormalization()(predictor)
            predictor = RepeatVector(input_shape[1])(predictor)
            predictor = Model_P2(predictor)
            predictor = BatchNormalization()(predictor)
            predictor = Model_P3(predictor)
            predictor = BatchNormalization()(predictor)
            predictor = Dense(tfpl.IndependentNormal.params_size(input_shape[2:]))(
                predictor
            )
            x_predicted_mean = tfpl.IndependentNormal(
                event_shape=input_shape[2:],
                convert_to_tensor_fn=tfp.distributions.Distribution.mean,
                name="vae_prediction",
            )(predictor)

            model_outs.append(x_predicted_mean)
            model_losses.append(log_loss)
            model_metrics["vae_prediction"] = ["mae", "mse"]
            loss_weights.append(self.predictor)

        if self.phenotype_prediction > 0:
            pheno_pred = Model_PC1(z)
            pheno_pred = Dense(tfpl.IndependentBernoulli.params_size(1))(pheno_pred)
            pheno_pred = tfpl.IndependentBernoulli(
                event_shape=1,
                convert_to_tensor_fn=tfp.distributions.Distribution.mean,
                name="phenotype_prediction",
            )(pheno_pred)

            model_outs.append(pheno_pred)
            model_losses.append(log_loss)
            model_metrics["phenotype_prediction"] = ["AUC", "accuracy"]
            loss_weights.append(self.phenotype_prediction)

        # define grouper and end-to-end autoencoder model
        grouper = Model(encoder.inputs, z_cat, name="Deep_Gaussian_Mixture_clustering")
        gmvaep = Model(
            inputs=encoder.inputs,
            outputs=model_outs,
            name="SEQ_2_SEQ_GMVAE",
        )

        if self.compile:
            gmvaep.compile(
                loss=model_losses,
                optimizer=Nadam(
                    lr=self.learn_rate,
                    clipvalue=self.clipvalue,
                ),
                metrics=model_metrics,
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
#       - Check usefulness of stateful sequential layers! (stateful=True in the LSTMs)
#       - Investigate full covariance matrix approximation for the latent space! (details on tfp course) :)
#       - Explore expanding the event dims of the final reconstruction layer
#       - Gaussian Mixture as output layer? One component per bodypart (makes sense?)
