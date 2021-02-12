# import IBrain explicitly so it's a little bit easier to access from outside
from .i_brain import IBrain, register_brain, registered_brains

# __all__ doesn't work, because classes are only loaded when they are actually imported.
# but most of the brains are never explicitly imported, and instead accessed via  IBrain.__subclasses__()
# thus we force loading all modules in this folder once.
from . import (
    CNN_CTRNN,
    concatenated_brains,
    continuous_time_rnn,
    elman,
    ffnn,
    gru,
    i_layer_based_brain,
    i_pytorch_brain,
    lstm,
)
