# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

keras_tuner hypermodels for hyperparameter tuning of deep autoencoders in deepof.models

"""

import tensorflow_probability as tfp
from keras_tuner import HyperModel

import deepof.unsupervised_utils
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
        n_components: int = 12,
    ):
        """

        VQVAE hypermodel for hyperparameter tuning.

        Args:
            input_shape (tuple): shape of the input tensor.
            latent_dim (int): dimension of the latent space.
            learn_rate (float): learning rate for the optimizer.
            n_components (int): number of components in the quantization space.

        """

        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.learn_rate = learn_rate
        self.n_components = n_components

    def get_hparams(self, hp):
        """

        Retrieve hyperparameters to tune, including the encoder type and the weight of the kmeans loss.

        """

        # Architectural hyperparameters
        encoder = hp.Choice(
            "encoder", ["recurrent", "TCN", "transformer"], default="recurrent"
        )
        kmeans_loss = hp.Float(
            "kmeans_loss", min_value=0.0, max_value=1.0, sampling="linear"
        )

        return (encoder, kmeans_loss)

    def build(self, hp):
        """

        Overrides Hypermodel's build method

        """

        # Hyperparameters to tune
        (encoder, kmeans_loss) = self.get_hparams(hp)

        vqvae = deepof.models.VQVAE(
            input_shape=self.input_shape,
            latent_dim=self.latent_dim,
            n_components=self.n_components,
            # hyperparameters to tune
            encoder_type=encoder,
            kmeans_loss=kmeans_loss,
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
        n_components: int = 10,
        learn_rate: float = 1e-3,
    ):
        """

        GMVAE hypermodel for hyperparameter tuning.

        Args:
            input_shape (tuple): shape of the input tensor.
            latent_dim (int): dimension of the latent space.
            batch_size (int): batch size for training.
            learn_rate (float): learning rate for the optimizer.
            n_components (int): number of components in the quantization space.

        """

        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        self.n_components = n_components

    def get_hparams(self, hp):
        """

        Retrieve hyperparameters to tune

        """

        # Architectural hyperparameters
        encoder = hp.Choice(
            "encoder", ["recurrent", "TCN", "transformer"], default="recurrent"
        )
        kmeans_loss = hp.Float(
            "kmeans_loss", min_value=0.0, max_value=1.0, sampling="linear"
        )
        kl_annealing_mode = hp.Choice(
            "kl_annealing_mode", ["linear", "sigmoid"], default="linear"
        )
        kl_warmup_epochs = hp.Int(
            "kl_warmup_epochs", min_value=0, max_value=100, sampling="linear"
        )
        cluster_assignment_regularizer = hp.Float(
            "cluster_assignment_regularizer",
            min_value=0.0,
            max_value=1.0,
            sampling="linear",
        )

        return (
            encoder,
            kmeans_loss,
            kl_annealing_mode,
            kl_warmup_epochs,
            cluster_assignment_regularizer,
        )

    def build(self, hp):
        """

        Overrides Hypermodel's build method

        """

        # Hyperparameters to tune
        (
            encoder,
            kmeans_loss,
            kl_annealing_mode,
            kl_warmup_epochs,
            cluster_assignment_regularizer,
        ) = self.get_hparams(hp)

        gmvae = deepof.models.GMVAE(
            input_shape=self.input_shape,
            latent_dim=self.latent_dim,
            n_components=self.n_components,
            batch_size=self.batch_size,
            # hyperparameters to tune
            encoder_type=encoder,
            kmeans_loss=kmeans_loss,
            kl_annealing_mode=kl_annealing_mode,
            kl_warmup_epochs=kl_warmup_epochs,
            reg_cat_clusters=cluster_assignment_regularizer,
        ).gmvae
        gmvae.compile()

        return gmvae
