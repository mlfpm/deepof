# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

keras hypermodels for hyperparameter tuning of deep autoencoders

"""

from kerastuner import HyperModel
import deepof.models
import deepof.model_utils
import tensorflow_probability as tfp

tfd = tfp.distributions
tfpl = tfp.layers


class SEQ_2_SEQ_AE(HyperModel):
    """Hyperparameter tuning pipeline for deepof.models.SEQ_2_SEQ_AE"""

    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

    @staticmethod
    def get_hparams(hp):
        """Retrieve hyperparameters to tune"""

        conv_filters = hp.Int(
            "units_conv",
            min_value=32,
            max_value=256,
            step=32,
            default=256,
        )
        lstm_units_1 = hp.Int(
            "units_lstm",
            min_value=128,
            max_value=512,
            step=32,
            default=256,
        )
        dense_2 = hp.Int(
            "units_dense2",
            min_value=32,
            max_value=256,
            step=32,
            default=64,
        )
        dropout_rate = hp.Float(
            "dropout_rate",
            min_value=0.0,
            max_value=0.5,
            default=0.25,
            step=0.05,
        )
        encoding = hp.Int(
            "encoding",
            min_value=16,
            max_value=64,
            step=8,
            default=24,
        )

        return conv_filters, lstm_units_1, dense_2, dropout_rate, encoding

    def build(self, hp):
        """Overrides Hypermodel's build method"""

        # HYPERPARAMETERS TO TUNE
        conv_filters, lstm_units_1, dense_2, dropout_rate, encoding = self.get_hparams(
            hp
        )

        # INSTANCIATED MODEL
        model = deepof.models.SEQ_2_SEQ_AE(
            architecture_hparams={
                "units_conv": conv_filters,
                "units_lstm": lstm_units_1,
                "units_dense_2": dense_2,
                "dropout_rate": dropout_rate,
                "encoding": encoding,
            }
        ).build(self.input_shape)[2]

        return model


class SEQ_2_SEQ_GMVAE(HyperModel):
    """Hyperparameter tuning pipeline for deepof.models.SEQ_2_SEQ_GMVAE"""

    def __init__(
        self,
        input_shape: tuple,
        encoding: int,
        entropy_reg_weight: float = 0.0,
        huber_delta: float = 1.0,
        kl_warmup_epochs: int = 0,
        learn_rate: float = 1e-3,
        loss: str = "ELBO+MMD",
        mmd_warmup_epochs: int = 0,
        number_of_components: int = 10,
        overlap_loss: float = False,
        phenotype_predictor: float = 0.0,
        predictor: float = 0.0,
        prior: str = "standard_normal",
    ):
        super().__init__()
        self.input_shape = input_shape
        self.encoding = encoding
        self.entropy_reg_weight = entropy_reg_weight
        self.huber_delta = huber_delta
        self.kl_warmup_epochs = kl_warmup_epochs
        self.learn_rate = learn_rate
        self.loss = loss
        self.mmd_warmup_epochs = mmd_warmup_epochs
        self.number_of_components = number_of_components
        self.overlap_loss = overlap_loss
        self.pheno_class = phenotype_predictor
        self.predictor = predictor
        self.prior = prior

        assert (
            "ELBO" in self.loss or "MMD" in self.loss
        ), "loss must be one of ELBO, MMD or ELBO+MMD (default)"

    def get_hparams(self, hp):
        """Retrieve hyperparameters to tune"""

        # Architectural hyperparameters
        bidirectional_merge = "ave"
        clipvalue = 1.0
        conv_filters = 160
        dense_2 = 120
        dense_activation = "relu"
        dense_layers_per_branch = 1
        dropout_rate = 1e-3
        k = self.number_of_components
        lstm_units_1 = 300

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

        gmvaep, kl_warmup_callback, mmd_warmup_callback = deepof.models.SEQ_2_SEQ_GMVAE(
            architecture_hparams={
                "bidirectional_merge": "ave",
                "clipvalue": clipvalue,
                "dense_activation": dense_activation,
                "dense_layers_per_branch": dense_layers_per_branch,
                "dropout_rate": dropout_rate,
                "units_conv": conv_filters,
                "units_dense_2": dense_2,
                "units_lstm": lstm_units_1,
            },
            encoding=self.encoding,
            entropy_reg_weight=self.entropy_reg_weight,
            huber_delta=self.huber_delta,
            kl_warmup_epochs=self.kl_warmup_epochs,
            loss=self.loss,
            mmd_warmup_epochs=self.mmd_warmup_epochs,
            number_of_components=k,
            overlap_loss=self.overlap_loss,
            phenotype_prediction=self.pheno_class,
            predictor=self.predictor,
        ).build(self.input_shape)[3:]

        return gmvaep


# TODO:
#    - We can add as many parameters as we want to the hypermodel!
#    with this implementation, predictor, warmup, loss and even number of components can be tuned using BayOpt
#    - Number of dense layers close to the latent space as a hyperparameter (!)
