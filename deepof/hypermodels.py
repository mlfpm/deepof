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

    def build(self, hp):
        """Overrides Hypermodel's build method"""

        # HYPERPARAMETERS TO TUNE
        conv_filters = hp.Int(
            "units_conv", min_value=32, max_value=256, step=32, default=256
        )
        lstm_units_1 = hp.Int(
            "units_lstm", min_value=128, max_value=512, step=32, default=256
        )
        dense_2 = hp.Int(
            "units_dense2", min_value=32, max_value=256, step=32, default=64
        )
        dropout_rate = hp.Float(
            "dropout_rate", min_value=0.0, max_value=0.5, default=0.25, step=0.05
        )
        encoding = hp.Int("encoding", min_value=16, max_value=64, step=8, default=24)

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
        input_shape,
        CONV_filters=256,
        DENSE_2=64,
        entropy_reg_weight=0.0,
        huber_delta=100,
        LSTM_units_1=256,
        LSTM_units_2=128,
        kl_warmup_epochs=0,
        learn_rate=1e-3,
        loss="ELBO+MMD",
        mmd_warmup_epochs=0,
        number_of_components=1,
        overlap_loss=False,
        predictor=0.0,
        prior="standard_normal",
    ):
        super().__init__()
        self.input_shape = input_shape
        self.CONV_filters = CONV_filters
        self.DENSE_1 = LSTM_units_2
        self.DENSE_2 = DENSE_2
        self.entropy_reg_weight = entropy_reg_weight
        self.huber_delta = huber_delta
        self.LSTM_units_1 = LSTM_units_1
        self.LSTM_units_2 = LSTM_units_2
        self.kl_warmup = kl_warmup_epochs
        self.kl_warmup_callback = None
        self.learn_rate = learn_rate
        self.loss = loss
        self.mmd_warmup = mmd_warmup_epochs
        self.mmd_warmup_callback = None
        self.number_of_components = number_of_components
        self.overlap_loss = overlap_loss
        self.predictor = predictor
        self.prior = prior

        assert (
            "ELBO" in self.loss or "MMD" in self.loss
        ), "loss must be one of ELBO, MMD or ELBO+MMD (default)"

    def build(self, hp):
        """Overrides Hypermodel's build method"""

        # Hyperparameters to tune
        conv_filters = hp.Int(
            "units_conv", min_value=32, max_value=256, step=32, default=256
        )
        lstm_units_1 = hp.Int(
            "units_lstm", min_value=128, max_value=512, step=32, default=256
        )
        dense_2 = hp.Int(
            "units_dense2", min_value=32, max_value=256, step=32, default=64
        )
        dropout_rate = hp.Float(
            "dropout_rate", min_value=0.0, max_value=0.5, default=0.25, step=0.05
        )
        encoding = hp.Int("encoding", min_value=16, max_value=64, step=8, default=24)

        gmvaep, kl_warmup_callback, mmd_warmup_callback = deepof.models.SEQ_2_SEQ_GMVAE(
            architecture_hparams={
                "units_conv": conv_filters,
                "units_lstm": lstm_units_1,
                "units_dense_2": dense_2,
                "dropout_rate": dropout_rate,
                "encoding": encoding,
            },
            entropy_reg_weight=self.entropy_reg_weight,
            huber_delta=self.huber_delta,
            kl_warmup_epochs=self.kl_warmup,
            loss=self.loss,
            mmd_warmup_epochs=self.mmd_warmup,
            number_of_components=self.number_of_components,
            overlap_loss=self.overlap_loss,
            predictor=self.predictor,
        ).build(self.input_shape)[3:]

        self.kl_warmup_callback = kl_warmup_callback
        self.mmd_warmup_callback = mmd_warmup_callback

        return gmvaep


# TODO:
#    - We can add as many parameters as we want to the hypermodel!
#    with this implementation, predictor, warmup, loss and even number of components can be tuned using BayOpt
#    - Number of dense layers close to the latent space as a hyperparameter (!)
