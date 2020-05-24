#!/usr/bin/env python3

import pickle
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from deap import base
from deap import creator
import argparse
import threading

from tools.experiment import Experiment
from brain_visualizer import BrainVisualizerHandler
from tools.helper import config_from_file


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='visualize CTRNN')
    parser.add_argument('--dir', metavar='dir', type=str,
                        help='path to the simulation result',
                        default=os.path.join('results/data', '2020-05-22_16-38-09'))

    parser.add_argument('--plot', metavar='dir', type=bool,
                        help='show plot?',
                        default=True)
    return parser.parse_args(args)


args = parse_args()
directory = os.path.join(args.dir)

with open(os.path.join(directory, 'HallOfFame.pickle'), "rb") as read_file_hof:
    # creator is needed to unpickle HOF
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, typecode='b', fitness=creator.FitnessMax)
    hall_of_fame = pickle.load(read_file_hof)
with open(os.path.join(directory, 'Log.json'), 'r') as read_file_log:
    log = json.load(read_file_log)

experiment = Experiment(configuration=config_from_file(os.path.join(directory, 'Configuration.json')),
                        result_path="asdasd",
                        from_checkpoint=None)

t = threading.Thread(target=experiment.visualize, args=[hall_of_fame[0:2], BrainVisualizerHandler()])
t.start()

generations = [i for i in range(len(log))]
avg = [generation["avg"] for generation in log]
maximum = [generation["max"] for generation in log]

# Plot results
if args.plot:
    plt.plot(generations, avg, 'r-')
    plt.plot(generations, maximum, 'y--')
    plt.xlabel('Generations')
    plt.legend(['avg', 'std high', 'std low'])
    plt.grid()
    plt.show()
