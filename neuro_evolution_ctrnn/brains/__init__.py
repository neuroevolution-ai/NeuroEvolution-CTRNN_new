# Import IBrain explicitly so it's a little bit easier to access from outside
from brains.brain import Brain, register_brain, registered_brains

# __all__ doesn't work, because classes are only loaded when they are actually imported.
# but most of the brains are never explicitly imported, and instead accessed via  Brain.__subclasses__()
# thus we force loading all modules in this folder once.
from brains.feed_forward.feed_forward import FeedForward
from brains.feed_forward.pytorch.feed_forward_pytorch import FeedForwardPyTorch
from brains.recurrent.elman import Elman
from brains.recurrent.gru import GRU
from brains.recurrent.lstm import LSTM
from brains.recurrent.pytorch.elman_pytorch import ElmanPyTorch
from brains.recurrent.pytorch.gru_pytorch import GRUPyTorch
from brains.recurrent.pytorch.lstm_pytorch import LSTMPyTorch
from brains.concatenated.concatenated_lstm import ConcatenatedLSTM
from brains.concatenated.cnn_ctrnn import CNNCTRNN
from brains.continuous_time_rnn import CTRNN
from brains.cnn import CNN


# # from concatenated import cnn_ctrnn, concatenated_lstm
# # from concatenated import *
# # from feed_forward import *
# # from recurrent import *
# # from .cnn import CNN
# # from .continuous_time_rnn import CTRNN
# # from .continuous_time_rnn import CTRNN
# # from .cnn import CNN
#
