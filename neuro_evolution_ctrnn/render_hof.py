#!/usr/bin/env python3
import pickle
import os
import json
import logging
from tap import Tap

from tools.experiment import Experiment
from brain_visualizer.brain_visualizer import BrainVisualizerHandler
from tools.helper import config_from_dict


class RenderArgs(Tap):
    # Note: Comments in lines become helpstring when called with --help
    # for details on parsing see: https://github.com/swansonk14/typed-argument-parser

    dir: str  # Directory path to the simulation result
    no_render: bool = False  # disable rendering to screen?
    record: bool = False  # record rendering to store it to file?
    record_force: bool = False  # force rendering even if file exists
    description = "Visualize Experiments"
    neuron_vis: bool = False  # show neuron visualizer?
    hof: int = 1  # how many members of hall-of-fame should be shown?
    rounds: int = 1  # how many episodes should be shown per HOF-member?
    slow_down: int = 0  # how many milliseconds should be pause between time steps?
    neuron_vis_width: int = 1600  # how wide should the neuron_vis window be?
    neuron_vis_height: int = 900  # how high should the neuron_vis window be?


args = RenderArgs().parse_args()
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
try:
    with open(os.path.join(args.dir, "Log.pkl"), "rb") as read_file_log:
        log = pickle.load(read_file_log)
except:
    with open(os.path.join(args.dir, "Log.json"), "r") as read_file_log:
        log = json.load(read_file_log)
with open(os.path.join(args.dir, "Configuration.json"), "r") as read_file:
    config = config_from_dict(json.load(read_file))

experiment = Experiment(configuration=config,
                        result_path="/tmp/not-used",
                        from_checkpoint=None, processing_framework="sequential")

with open(os.path.join(args.dir, "HallOfFame.pickle"), "rb") as read_file_hof:
    # creator is needed to unpickle HOF
    # creator is registered when loading experiment
    hall_of_fame = pickle.load(read_file_hof)

assert len(hall_of_fame) >= args.hof, "The 'hof' value {} is too large as the hall of fame has a size of {}.".format(
    args.hof, len(hall_of_fame))

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

render = not args.no_render

for i, individual in enumerate(individuals):
    if args.record:
        record = os.path.join(args.dir, "video_{}".format(i))
        logging.info("Recording an individual to {}".format(record))
    else:
        record = None

    experiment.ep_runner.eval_fitness(individual, config.random_seed, render, record, record_force,
                                      BrainVisualizerHandler(), args.neuron_vis, args.slow_down, args.rounds,
                                      args.neuron_vis_width, args.neuron_vis_height)
