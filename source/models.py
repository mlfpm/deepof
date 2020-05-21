# @author lucasmiranda42

from keras import backend as K
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.constraints import Constraint, UnitNorm
from tensorflow.keras.layers import Bidirectional, Dense, Dropout
from tensorflow.keras.layers import Lambda, Layer, LSTM
from tensorflow.keras.layers import RepeatVector, TimeDistributed
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


class SEQ_2_SEQ_AE:
    pass


class SEQ_2_SEQ_VAE:
    pass


class SEQ_2_SEQ_MVAE(HyperModel):
    pass


class SEQ_2_SEQ_MMVAE(HyperModel):
    pass
