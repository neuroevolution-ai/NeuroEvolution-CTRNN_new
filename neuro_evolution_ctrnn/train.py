#!/usr/bin/env python3

import argparse
import os
from datetime import datetime

from tools.experiment import Experiment
from tools.helper import config_from_file

import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='train CTRNN')
    parser.add_argument('--from-checkpoint', metavar='dir', type=str,
                        help='continues training from a checkpoint', default=None)
    parser.add_argument('--configuration', metavar='dir', type=str,
                        help='use an alternative configuration file', default='configurations/default.json')
    parser.add_argument('--result-path', metavar='dir', type=os.path.abspath,
                        help='use an alternative path for simulation results',
                        default=os.path.join('..', 'CTRNN_Simulation_Results', 'data',
                                             datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))

    return parser.parse_args(args)


args = parse_args()
experiment = Experiment(configuration=config_from_file(args.configuration), result_path=args.result_path,
                        from_checkpoint=args.from_checkpoint)

if __name__ == "__main__":  # pragma: no cover
    """Everything outside this block will be executed by every scoop-worker, while this block is only run on the 
    main thread. Every object that is later passed to a worker must be pickle-able, that's why we 
    initialise everything that is not pickle-able before this point. Especially the DEAP-toolbox is not 
    pickle-able. 
    """
    os.mkdir(args.result_path)
    experiment.run()
