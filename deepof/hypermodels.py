# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

keras hypermodels for hyperparameter tuning of deep autoencoders

"""

import tensorflow_probability as tfp
from keras_tuner import HyperModel

import deepof.model_utils
import deepof.models

tfd = tfp.distributions
tfpl = tfp.layers


class VQVAE(HyperModel):
    """Hyperparameter tuning pipeline for deepof.models.SEQ_2_SEQ_GMVAE"""

    def __init__(
        self,
        input_shape: tuple,
        encoding: int,
        batch_size: int,
        learn_rate: float = 1e-3,
        n_components: int = 10,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.encoding = encoding
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        self.n_components = n_components

    def get_hparams(self, hp):
        """Retrieve hyperparameters to tune"""

        # Architectural hyperparameters
        bidirectional_merge = "ave"
        clipvalue = 1.0
        conv_filters = hp.Int("conv_units", min_value=32, max_value=512, step=32)
        dense_2 = hp.Int("dense_units", min_value=32, max_value=512, step=32)
        dense_activation = "relu"
        dense_layers_per_branch = hp.Int(
            "dense_layers", min_value=1, max_value=3, step=1
        )
        dropout_rate = hp.Float(
            "dropout_rate", min_value=0.0, max_value=1.0, sampling="linear"
        )
        k = self.n_components
        lstm_units_1 = hp.Int("units", min_value=32, max_value=512, step=32)

        return (
            bidirectional_merge,
            clipvalue,
            conv_filters,
            dense_2,
            dense_activation,
            dense_layers_per_branch,
            dropout_rate,
            k,
            lstm_units_1,
        )

    def build(self, hp):
        """Overrides Hypermodel's build method"""

        # Hyperparameters to tune
        (
            bidirectional_merge,
            clipvalue,
            conv_filters,
            dense_2,
            dense_activation,
            dense_layers_per_branch,
            dropout_rate,
            k,
            lstm_units_1,
        ) = self.get_hparams(hp)

        gmvaep = deepof.models.GMVAE(
            architecture_hparams={
                "bidirectional_merge": "concat",
                "clipvalue": clipvalue,
                "dense_activation": dense_activation,
                "dense_layers_per_branch": dense_layers_per_branch,
                "dropout_rate": dropout_rate,
                "units_conv": conv_filters,
                "units_dense_2": dense_2,
                "units_lstm": lstm_units_1,
            },
            input_shape=self.input_shape,
            latent_dim=self.encoding,
            batch_size=self.batch_size,
            kl_warmup_epochs=self.kl_warmup_epochs,
            loss=self.loss,
            mmd_warmup_epochs=self.mmd_warmup_epochs,
            n_components=k,
            overlap_loss=self.overlap_loss,
            next_sequence_prediction=self.next_sequence_prediction,
            phenotype_prediction=self.phenotype_prediction,
            supervised_prediction=self.supervised_prediction,
            supervised_features=self.supervised_features,
        ).gmvae

        return gmvaep


class GMVAE(HyperModel):
    """Hyperparameter tuning pipeline for deepof.models.SEQ_2_SEQ_GMVAE"""

    def __init__(
        self,
        input_shape: tuple,
        encoding: int,
        batch_size: int,
        kl_warmup_epochs: int = 0,
        learn_rate: float = 1e-3,
        loss: str = "ELBO+MMD",
        mmd_warmup_epochs: int = 0,
        n_components: int = 10,
        overlap_loss: float = False,
        next_sequence_prediction: float = 0.0,
        phenotype_prediction: float = 0.0,
        supervised_prediction: float = 0.0,
        supervised_features: int = 6,
        prior: str = "standard_normal",
    ):
        super().__init__()
        self.input_shape = input_shape
        self.encoding = encoding
        self.batch_size = batch_size
        self.kl_warmup_epochs = kl_warmup_epochs
        self.learn_rate = learn_rate
        self.loss = loss
        self.mmd_warmup_epochs = mmd_warmup_epochs
        self.n_components = n_components
        self.overlap_loss = overlap_loss
        self.next_sequence_prediction = next_sequence_prediction
        self.phenotype_prediction = phenotype_prediction
        self.supervised_prediction = supervised_prediction
        self.supervised_features = supervised_features
        self.prior = prior

        assert (
            "ELBO" in self.loss or "MMD" in self.loss
        ), "loss must be one of ELBO, MMD or ELBO+MMD (default)"

    def get_hparams(self, hp):
        """Retrieve hyperparameters to tune"""

        # Architectural hyperparameters
        bidirectional_merge = "ave"
        clipvalue = 1.0
        conv_filters = hp.Int("conv_units", min_value=32, max_value=512, step=32)
        dense_2 = hp.Int("dense_units", min_value=32, max_value=512, step=32)
        dense_activation = "relu"
        dense_layers_per_branch = hp.Int(
            "dense_layers", min_value=1, max_value=3, step=1
        )
        dropout_rate = hp.Float(
            "dropout_rate", min_value=0.0, max_value=1.0, sampling="linear"
        )
        k = self.n_components
        lstm_units_1 = hp.Int("units", min_value=32, max_value=512, step=32)

        return (
            bidirectional_merge,
            clipvalue,
            conv_filters,
            dense_2,
            dense_activation,
            dense_layers_per_branch,
            dropout_rate,
            k,
            lstm_units_1,
        )

    def build(self, hp):
        """Overrides Hypermodel's build method"""

        # Hyperparameters to tune
        (
            bidirectional_merge,
            clipvalue,
            conv_filters,
            dense_2,
            dense_activation,
            dense_layers_per_branch,
            dropout_rate,
            k,
            lstm_units_1,
        ) = self.get_hparams(hp)

        gmvaep = deepof.models.GMVAE(
            architecture_hparams={
                "bidirectional_merge": "concat",
                "clipvalue": clipvalue,
                "dense_activation": dense_activation,
                "dense_layers_per_branch": dense_layers_per_branch,
                "dropout_rate": dropout_rate,
                "units_conv": conv_filters,
                "units_dense_2": dense_2,
                "units_lstm": lstm_units_1,
            },
            input_shape=self.input_shape,
            latent_dim=self.encoding,
            batch_size=self.batch_size,
            kl_warmup_epochs=self.kl_warmup_epochs,
            loss=self.loss,
            mmd_warmup_epochs=self.mmd_warmup_epochs,
            n_components=k,
            overlap_loss=self.overlap_loss,
            next_sequence_prediction=self.next_sequence_prediction,
            phenotype_prediction=self.phenotype_prediction,
            supervised_prediction=self.supervised_prediction,
            supervised_features=self.supervised_features,
        ).gmvae

        return gmvaep
