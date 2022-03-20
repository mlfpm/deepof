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
        reg_gram: float = 0.0,
        phenotype_prediction: float = 0.0,
    ):
        """

        VQVAE hypermodel for hyperparameter tuning.

        Args:
            input_shape (tuple): shape of the input tensor.
            latent_dim (int): dimension of the latent space.
            learn_rate (float): learning rate for the optimizer.
            n_components (int): number of components in the quantization space.
            reg_gram (float): regularization parameter for the Gram matrix of the latent embeddings.
            phenotype_prediction (float): weight of the phenotype prediction loss.

        """

        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.learn_rate = learn_rate
        self.n_components = n_components
        self.reg_gram = reg_gram
        self.phenotype_prediction = phenotype_prediction

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
            phenotype_prediction=self.phenotype_prediction,
        ).vqvae
        vqvae.compile()

        return vqvae
