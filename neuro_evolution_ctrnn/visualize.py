import pickle
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from deap import base
from deap import creator
import argparse
import threading

from neuro_evolution_ctrnn.tools.experiment import Experiment
from neuro_evolution_ctrnn.brain_visualizer import BrainVisualizerHandler


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='visualize CTRNN')
    parser.add_argument('--configuration', metavar='dir', type=str,
                        help='use an alternative configuration file',
                        default=os.path.join('results', '2020-05-19_10-10-31'))
    return parser.parse_args(args)


args = parse_args()
directory = os.path.join(args.configuration)

with open(os.path.join(directory, 'HallOfFame.pickle'), "rb") as read_file:
    # creator is needed to unpickle HOF
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, typecode='b', fitness=creator.FitnessMax)
    hall_of_fame = pickle.load(read_file)
with open(os.path.join(directory, 'Log.json'), 'r') as read_file:
    log = json.load(read_file)

experiment = Experiment(configuration_path=os.path.join(directory, 'Configuration.json'), result_path="asdasd",
                        from_checkpoint=None)

t = threading.Thread(target=experiment.visualize, args=[hall_of_fame[0:2], BrainVisualizerHandler()])
t.start()

generations = [i for i in range(len(log))]
avg = [generation["avg"] for generation in log]
maximum = [generation["max"] for generation in log]

# Plot results
plt.plot(generations, avg, 'r-')
plt.plot(generations, maximum, 'y--')
plt.xlabel('Generations')
plt.legend(['avg', 'std high', 'std low'])
plt.grid()
plt.show()
