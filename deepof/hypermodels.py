# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

keras_tuner hypermodels for hyperparameter tuning of deep autoencoders in deepof.models

"""

import tensorflow_probability as tfp
from keras_tuner import HyperModel

import deepof.model_utils
import deepof.models

tfd = tfp.distributions
tfpl = tfp.layers


class VQVAE(HyperModel):
    """

    Hyperparameter tuning pipeline for deepof.models.SEQ_2_SEQ_VQVAE

    """

    def __init__(
        self,
        input_shape: tuple,
        latent_dim: int,
        learn_rate: float = 1e-3,
        n_components: int = 10,
        reg_gram: float = 1.0,
    ):
        """

        VQVAE hypermodel for hyperparameter tuning.

        Args:
            input_shape (tuple): shape of the input tensor.
            latent_dim (int): dimension of the latent space.
            learn_rate (float): learning rate for the optimizer.
            n_components (int): number of components in the quantization space.
            reg_gram (float): regularization parameter for the Gram matrix of the latent embeddings.

        """

        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.learn_rate = learn_rate
        self.n_components = n_components
        self.reg_gram = reg_gram

    def get_hparams(self, hp):
        """

        Retrieve hyperparameters to tune

        """

        # Architectural hyperparameters
        bidirectional_merge = "concat"
        clipvalue = 1.0
        conv_filters = hp.Int("conv_units", min_value=32, max_value=512, step=32)
        dense_activation = "relu"
        k = self.n_components
        rnn_units_1 = hp.Int("units", min_value=32, max_value=512, step=32)

        return (
            bidirectional_merge,
            clipvalue,
            conv_filters,
            dense_activation,
            k,
            rnn_units_1,
        )

    def build(self, hp):
        """

        Overrides Hypermodel's build method

        """

        # Hyperparameters to tune
        (
            bidirectional_merge,
            clipvalue,
            conv_filters,
            dense_activation,
            k,
            lstm_units_1,
        ) = self.get_hparams(hp)

        vqvae = deepof.models.VQVAE(
            architecture_hparams={
                "bidirectional_merge": "concat",
                "clipvalue": clipvalue,
                "dense_activation": dense_activation,
                "units_conv": conv_filters,
                "units_lstm": lstm_units_1,
            },
            input_shape=self.input_shape,
            latent_dim=self.latent_dim,
            n_components=k,
            reg_gram=self.reg_gram,
        ).vqvae
        vqvae.compile()

        return vqvae


class GMVAE(HyperModel):
    """

    Hyperparameter tuning pipeline for deepof.models.SEQ_2_SEQ_GMVAE

    """

    def __init__(
        self,
        input_shape: tuple,
        latent_dim: int,
        batch_size: int,
        kl_warmup_epochs: int = 0,
        learn_rate: float = 1e-3,
        loss: str = "ELBO+MMD",
        mmd_warmup_epochs: int = 0,
        n_components: int = 10,
        n_cluster_loss: float = False,
        reg_gram: float = 1.0,
        next_sequence_prediction: float = 0.0,
        phenotype_prediction: float = 0.0,
        supervised_prediction: float = 0.0,
        supervised_features: int = 6,
    ):
        """

        GMVAE hypermodel for hyperparameter tuning.

        Args:
            input_shape (tuple): shape of the input tensor.
            latent_dim (int): dimension of the latent space.
            batch_size (int): batch size for training.
            kl_warmup_epochs (int): number of epochs to warmup KL loss.
            learn_rate (float): learning rate for the optimizer.
            loss (str): loss function to use.
            mmd_warmup_epochs (int): number of epochs to warmup MMD loss.
            n_components (int): number of components in the quantization space.
            n_cluster_loss (float): weight of the n_cluster_loss.
            reg_gram (float): regularization parameter for the Gram matrix of the latent embeddings.
            next_sequence_prediction (float): weight of the next sequence prediction loss.
            phenotype_prediction (float): weight of the phenotype prediction loss.
            supervised_prediction (float): weight of the supervised prediction loss.
            supervised_features (int): number of features in the supervised prediction loss.

        """

        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.kl_warmup_epochs = kl_warmup_epochs
        self.learn_rate = learn_rate
        self.loss = loss
        self.mmd_warmup_epochs = mmd_warmup_epochs
        self.n_components = n_components
        self.n_cluster_loss = n_cluster_loss
        self.reg_gram = reg_gram
        self.next_sequence_prediction = next_sequence_prediction
        self.phenotype_prediction = phenotype_prediction
        self.supervised_prediction = supervised_prediction
        self.supervised_features = supervised_features

        assert (
            "SIWAE" in self.latent_loss
            or "SELBO" in self.latent_loss
            or "MMD" in self.latent_loss
        ), "loss must be one of SIWAE, SELBO (default), MMD, SIWAE+MMD, or SELBO+MMD"

    def get_hparams(self, hp):
        """

        Retrieve hyperparameters to tune

        """

        # Architectural hyperparameters
        bidirectional_merge = "concat"
        clipvalue = 1.0
        conv_filters = hp.Int("conv_units", min_value=32, max_value=512, step=32)
        dense_activation = "relu"
        k = self.n_components
        rnn_units_1 = hp.Int("units", min_value=32, max_value=512, step=32)

        return (
            bidirectional_merge,
            clipvalue,
            conv_filters,
            dense_activation,
            k,
            rnn_units_1,
        )

    def build(self, hp):
        """

        Overrides Hypermodel's build method

        """

        # Hyperparameters to tune
        (
            bidirectional_merge,
            clipvalue,
            conv_filters,
            dense_activation,
            k,
            lstm_units_1,
        ) = self.get_hparams(hp)

        gmvae = deepof.models.GMVAE(
            architecture_hparams={
                "bidirectional_merge": "concat",
                "clipvalue": clipvalue,
                "dense_activation": dense_activation,
                "units_conv": conv_filters,
                "units_lstm": lstm_units_1,
            },
            input_shape=self.input_shape,
            latent_dim=self.latent_dim,
            batch_size=self.batch_size,
            kl_warmup_epochs=self.kl_warmup_epochs,
            latent_loss=self.loss,
            mmd_warmup_epochs=self.mmd_warmup_epochs,
            n_components=k,
            n_cluster_loss=self.n_cluster_loss,
            reg_gram=self.reg_gram,
            next_sequence_prediction=self.next_sequence_prediction,
            phenotype_prediction=self.phenotype_prediction,
            supervised_prediction=self.supervised_prediction,
            supervised_features=self.supervised_features,
        ).gmvae
        gmvae.compile()

        return gmvae
