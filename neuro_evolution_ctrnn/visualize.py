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
import numpy as np

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

parser = argparse.ArgumentParser(description="Visualize Experiments")

parser.add_argument("--dir", metavar="dir", type=str, help="Directory path to the simulation result",
                    default=os.path.join("..", "CTRNN_Simulation_Results", "data", "2020-05-26_10-51-17"))

parser.add_argument("--plot", dest="plot", action="store_true")

parser.add_argument("--plot-save", type=str, help="A filename where the plot should be saved", default=None)

parser.add_argument("--plot-novelty", dest="plot_novelty", action="store_true")

parser.add_argument("--render", action="store_true")

parser.add_argument("--record", action="store_true")

parser.add_argument("--record-force", action="store_true")

parser.add_argument("--smooth", type=int, help="How strong should the lines be smoothed? (0 to disable)", default=0)

parser.add_argument("--neuron-vis", dest="neuron_vis", action="store_true")

parser.add_argument("--hof", type=int, help="How many individuals shall be visualized?", default=0)

parser.add_argument("--slow-down", type=int, help="Insert a pause between the iterations in milliseconds", default=0)

parser.add_argument("--rounds", metavar="int", type=int, help="How many episodes shall be conducted per individual?",
                    default=1)

parser.add_argument("--style", metavar="int", type=str, help="Which plot style should be used?",
                    default="seaborn-paper")

args = parser.parse_args()

with open(os.path.join(args.dir, "HallOfFame.pickle"), "rb") as read_file_hof:
    # creator is needed to unpickle HOF
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, typecode="b", fitness=creator.FitnessMax)
    hall_of_fame = pickle.load(read_file_hof)

with open(os.path.join(args.dir, "Log.pkl"), "rb") as read_file_log:
    log = pickle.load(read_file_log)

with open(os.path.join(args.dir, "Configuration.json"), "r") as read_file:
    conf = json.load(read_file)
    config = config_from_file(os.path.join(args.dir, "Configuration.json"))

if args.neuron_vis or args.hof or args.render or args.record:
    experiment = Experiment(configuration=config,
                            result_path="/tmp/not-used",
                            from_checkpoint=None)

    if len(hall_of_fame) < args.hof:
        raise RuntimeError(
            "The 'hof' value {} is too large as the hall of fame has a size of {}.".format(args.hof, len(hall_of_fame)))

    individuals = hall_of_fame[0:args.hof]

    if args.rounds > 1:
        # If multiple episodes are used per individual then we need to have either different recording directories
        # or simply overwrite the old recording (force=True)
        record_force = True
    else:
        record_force = args.record_force

    if hasattr(config.optimizer, "mutation_learned"):
        # sometimes there are also optimizing strategies encoded in the genome. These parameters
        # are not part of the brain and need to be removed from the genome before initializing the brain.
        individuals = experiment.optimizer.strip_strategy_from_population(individuals,
                                                                          config.optimizer.mutation_learned)
    elif config.optimizer.type == "MU_ES":
        # later version of don't have the "mutation_learned" option anymore and instead always use that option
        individuals = experiment.optimizer.strip_strategy_from_population(individuals, True)

    for i, individual in enumerate(individuals):
        if args.record:
            record = os.path.join(args.dir, "video_{}".format(i))
            logging.info("Recording an individual to {}".format(record))
        else:
            record = None
        experiment.ep_runner.eval_fitness(individual, config.random_seed, args.render, record, record_force, args.rounds,
                                   BrainVisualizerHandler(), args.neuron_vis, args.slow_down)



# Plot results
def my_plot(axis, *nargs, **kwargs, ):
    lst = list(nargs)
    if args.smooth:
        lst[1] = gaussian_filter1d(nargs[1], sigma=args.smooth)
        t = tuple(lst)
        kwargs["alpha"] = 0.8
        axis.plot(*t, **kwargs, )
        kwargs["alpha"] = 0.2
        del kwargs["label"]
    axis.plot(*nargs, **kwargs)


def plot_chapter(axis, chapter, gens, colors):
    fit_min, fit_avg, fit_max, fit_std = chapter.select('min', 'avg', 'max', 'std')

    std_low = list(map(add, fit_avg, np.array(fit_std) / 2))
    std_high = list(map(sub, fit_avg, np.array(fit_std) / 2))

    my_plot(axis, gens, fit_max, '-', color=colors[0], label="maximum")
    my_plot(axis, gens, fit_avg, '-', color=colors[1], label="average")
    axis.fill_between(generations, std_low, std_high, facecolor=colors[2], alpha=0.15,
                      label='variance')
    my_plot(axis, gens, fit_min, '-', color=colors[3], label="minimum")


if args.plot_save or args.plot:
    generations = [i for i in range(len(log))]

    base_dir = os.path.basename(args.dir)
    params_display = conf['environment'] + "\n" + conf['brain']['type'] + " + " + conf['optimizer'][
        'type'] + "\nneurons: " + str(conf['brain']['number_neurons'])

    fig, ax1 = plt.subplots()
    plt.style.use('seaborn-paper')

    plot_chapter(ax1, log.chapters["fitness"], generations, ("green", "teal", "teal", 'blue'))

    ax1.set_xlabel('Generations')
    ax1.set_ylabel('Fitness')
    ax1.legend(loc='upper left')

    ax1.grid()
    plt.title(base_dir)
    ax1.text(0.96, 0.05, params_display, ha='right',
             fontsize=8, fontname='Ubuntu', transform=ax1.transAxes,
             bbox={'facecolor': 'white', 'alpha': 0.4, 'pad': 8})

    if args.plot_novelty:
        # quickfix because first value is bugged
        log.chapters["novelty"][0]["min"] = log.chapters["novelty"][1]["min"]
        log.chapters["novelty"][0]["avg"] = log.chapters["novelty"][1]["avg"]
        log.chapters["novelty"][0]["max"] = log.chapters["novelty"][1]["max"]
        log.chapters["novelty"][0]["std"] = log.chapters["novelty"][1]["std"]

        ax2 = plt.twinx()
        ax2.set_ylabel('Novelty')
        plot_chapter(ax2, log.chapters["novelty"], generations, ("yellow", "orange", "orange", 'pink'))
        ax2.legend(loc='lower left')

    if args.plot_save:
        logging.info("saving plot to: " + str(args.plot_save))
        plt.savefig(args.plot_save)

    if args.plot:
        plt.show()
