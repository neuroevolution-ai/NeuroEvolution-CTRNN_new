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
from operator import add, sub
from scipy.ndimage.filters import gaussian_filter1d

from tools.experiment import Experiment
from brain_visualizer import BrainVisualizerHandler
from tools.helper import config_from_file


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='visualize CTRNN')
    parser.add_argument('--dir', metavar='dir', type=str,
                        help='path to the simulation result',
                        default=os.path.join('results/data', '2020-05-22_16-38-09'))

    parser.add_argument('--plot', metavar='bool', type=bool,
                        help='show plot?',
                        default=True)
    parser.add_argument('--neuron_vis', metavar='bool', type=bool,
                        help='show neuron visualizer?',
                        default=False)
    parser.add_argument('--hof', metavar='int', type=int,
                        help='show how many individuals in environment?',
                        default=0)
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


if args.neuron_vis or args.hof:
    experiment = Experiment(configuration=config_from_file(os.path.join(directory, 'Configuration.json')),
                            result_path="asdasd",
                            from_checkpoint=None)
    t = threading.Thread(target=experiment.visualize, args=[hall_of_fame[0:args.hof], BrainVisualizerHandler()])
    t.start()

# Plot results
if args.plot:
    generations = [i for i in range(len(log))]
    average = [generation["avg"] for generation in log]
    maximum = [generation["max"] for generation in log]
    std = [generation["std"] for generation in log]
    std_low = list(map(add, average, std))
    std_high = list(map(sub, average, std))
    minimum = [generation["min"] for generation in log]


    def my_plot(*nargs, **kwargs, ):
        lst = list(nargs)
        lst[1] = gaussian_filter1d(nargs[1], sigma=15)
        t = tuple(lst)
        kwargs["alpha"] = 0.8
        plt.plot(*t, **kwargs, )
        kwargs["alpha"] = 0.3
        del kwargs["label"]
        plt.plot(*nargs, **kwargs, )


    my_plot(generations, minimum, '-', color="pink", label="minimum")
    my_plot(generations, maximum, '-', color="teal", label="maximum")
    my_plot(generations, average, '-', color="green", label="average")
    plt.fill_between(generations, std_low, std_high, facecolor='green', alpha=0.2,
                     label='variance')

    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()
