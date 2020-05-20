import random

from deap import tools
import pickle
from pathlib import Path
import os
import numpy as np
from deap.algorithms import varOr
from tools.helper import set_random_seeds
from typing import Iterable, Sized, Collection
from deap.tools.support import Logbook


def eaGenerateUpdate(toolbox, ngen: int, halloffame=None, stats=None):
    start_gen: int = 0
    logbook: Logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    for gen in range(start_gen, ngen + 1):
        population: Collection = toolbox.generate()
        seed_after_map: int = random.randint(1, 10000)
        seeds_for_evaluation: np.ndarray = np.random.randint(1, 10000, size=len(population))
        finesses: Iterable = toolbox.map(toolbox.evaluate, population, seeds_for_evaluation)
        for ind, fit in zip(population, finesses):
            ind.fitness.values = fit
        # reseed because workers seem to affect the global state
        # also this must happen AFTER fitness-values have been processes, because futures
        set_random_seeds(seed_after_map, env=None)
        if halloffame is not None:
            halloffame.update(population)
        toolbox.update(population)
        record: dict = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        print(logbook.stream)

    return logbook


def _write_checkpoint(data, generation):
    cp_dir = "checkpoints"
    Path(cp_dir).mkdir(parents=True, exist_ok=True)
    filename = os.path.join(cp_dir, "checkpoint_" + str(generation) + ".pkl")
    print("writing checkpoint " + filename)
    with open(filename, "wb") as cp_file:
        pickle.dump(data, cp_file, protocol=pickle.HIGHEST_PROTOCOL, fix_imports=False)


def get_checkpoint(checkpoint):
    with open(checkpoint, "rb") as cp_file:
        cp = pickle.load(cp_file, fix_imports=False)
    return cp
