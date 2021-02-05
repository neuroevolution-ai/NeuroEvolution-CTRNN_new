
# import IBrain explicitly so it's a little bit easier to access from outside
from .i_brain import IBrain


# __all__ doesn't work, because classes are only loaded when they are actually imported.
# but most of the brains are never explicitly imported, and instead accessed via  IBrain.__subclasses__()
# thus we force loading all modules in this folder once.

import os
import importlib

for module in os.listdir(os.path.dirname(__file__)):
    if module == '__init__.py' or module[-3:] != '.py':
        continue
    importlib.import_module("." + module[:-3], "neuro_evolution_ctrnn.brains")
del module

