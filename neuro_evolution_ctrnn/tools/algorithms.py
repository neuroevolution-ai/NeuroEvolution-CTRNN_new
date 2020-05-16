import random

from deap import tools
import pickle
from pathlib import Path
import os
from deap.algorithms import varOr


def eaGenerateUpdate(toolbox, ngen, halloffame=None, stats=None):
    population = None
    start_gen = 0
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    for gen in range(start_gen, ngen + 1):
        if toolbox.cb_before_each_generation:
            toolbox.cb_before_each_generation()
        population = toolbox.generate()
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        if halloffame is not None:
            halloffame.update(population)
        toolbox.update(population)
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(population), **record)

    return population, logbook


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
