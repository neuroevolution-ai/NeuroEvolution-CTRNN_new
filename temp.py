

import pickle
import os
from deap import base
from deap import creator
from deap import tools
import numpy as np

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, typecode='b', fitness=creator.FitnessMax)
with open(os.path.join("../checkpoint_4000.pkl"), "rb") as read_file_hof:
    # creator is needed to unpickle HOF
    # creator is registered when loading experiment
    x = pickle.load(read_file_hof)

y = []
for ind1 in x['population']:
    half = len(ind1) // 2
    # y.append(ind1[half:])
    y.append(ind1[-2:])

y = np.array(y)
print(y)
