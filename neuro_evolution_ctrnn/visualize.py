#!/usr/bin/env python3

import pickle
import os
import json
import matplotlib.pyplot as plt
from deap import base
from deap import creator
import argparse
import threading
from operator import add, sub
from scipy.ndimage.filters import gaussian_filter1d
import logging

from tools.experiment import Experiment
from brain_visualizer import BrainVisualizerHandler
from tools.helper import config_from_file
from tools.episode_runner import MemoryEpisodeRunner
from brains.lstm import LSTMNumPy
import gym

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='visualize CTRNN')
    parser.add_argument('--dir', metavar='dir', type=str,
                        help='path to the simulation result',
                        default=os.path.join('..', 'CTRNN_Simulation_Results', 'data', '2020-05-26_10-51-17'))

    parser.add_argument('--plot', dest='plot', action='store_true')
    parser.add_argument('--no-plot', dest='plot', action='store_false')
    parser.set_defaults(plot=True)
    parser.add_argument('--plot-save', type=str,
                        help='a filename where the plot should be saved',
                        default=None)


    parser.add_argument('--neuron-vis', dest='neuron_vis', action='store_true')
    parser.add_argument('--no-neuron-vis', dest='neuron_vis', action='store_false')
    parser.set_defaults(neuron_vis=True)

    parser.add_argument('--hof', type=int,
                        help='show how many individuals in environment?',
                        default=0)
    parser.add_argument('--slow-down', type=int,
                        help='Insert a pause between iteration (milliseconds)',
                        default=0)
    parser.add_argument('--rounds', metavar='int', type=int,
                        help='how many rounds per individual?',
                        default=1)
    parser.add_argument('--style', metavar='int', type=str,
                        help='Which plot-style should be used? ',
                        default='seaborn-paper')

    parser.add_argument('--memory', type=bool, default=False)
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

with open(os.path.join(directory, 'Configuration.json'), "r") as read_file:
    conf = json.load(read_file)

if args.neuron_vis and args.hof:
    experiment = Experiment(configuration=config_from_file(os.path.join(directory, 'Configuration.json')),
                            result_path="asdasd",
                            from_checkpoint=None)
    t = threading.Thread(target=experiment.visualize, args=[hall_of_fame[0:args.hof], BrainVisualizerHandler(), args.rounds, args.neuron_vis, args.slow_down])
    t.start()

if args.memory:
    config = config_from_file(os.path.join(directory, 'Configuration.json'))
    env = gym.make(config.environment)
    episode_runner = MemoryEpisodeRunner(config=config.episode_runner, brain_conf=config.brain, discrete_actions=False,
                                         brain_class=LSTMNumPy, input_space=env.observation_space,
                                         output_space=env.action_space, env_template=env, render=True,
                                         record_directory=directory)

    individual = hall_of_fame[0:1][0]
    episode_runner.eval_fitness(individual, config.random_seed)

# Plot results
if args.plot_save or args.plot:
    generations = [i for i in range(len(log))]
    average = [generation["avg"] for generation in log]
    maximum = [generation["max"] for generation in log]
    std = [generation["std"] for generation in log]
    std_low = list(map(add, average, std))
    std_high = list(map(sub, average, std))
    minimum = [generation["min"] for generation in log]

    base_dir = os.path.basename(args.dir)
    params_display = conf['environment'] + "\n" + conf['brain']['type'] + " + " + conf['optimizer'][
        'type'] + "\nneurons: " + str(conf['brain']['number_neurons'])

    plt.style.use('seaborn-paper')


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
    plt.title(base_dir)
    plt.text(0.96, 0.05, params_display, ha='right',
             fontsize=8, fontname='Ubuntu', transform=plt.axes().transAxes,
             bbox={'facecolor': 'white', 'alpha': 0.4, 'pad': 8})

    if args.plot_save:
        logging.info("saving plot to: " + str(args.plot_save))
        plt.savefig(args.plot_save)

    if args.plot:
        plt.show()
