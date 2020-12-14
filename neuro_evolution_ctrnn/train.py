#!/usr/bin/env python3

import argparse
import os
from datetime import datetime
import logging

from tools.experiment import Experiment
from tools.config_reader import ConfigReader

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Train CTRNN")
    parser.add_argument("--from-checkpoint", metavar="dir", type=str,
                        help="Continues training from a checkpoint", default=None)
    parser.add_argument("--configuration", metavar="dir", type=str,
                        help="Use an alternative configuration file", default="configurations/default.json")
    parser.add_argument("--result-path", metavar="dir", type=os.path.abspath,
                        help="Use an alternative path for simulation results",
                        default=os.path.join("..", "CTRNN_Simulation_Results", "data",
                                             datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    parser.add_argument("-p", "--processing-framework", metavar="dask/mp/sequential", type=str, default="dask",
                        help="Choose the framework used for the processing")
    parser.add_argument("-n", "--num-workers", metavar="int", type=int, default=os.cpu_count(),
                        help="Specify the amount of workers for the computation")

    return parser.parse_args(args)


if __name__ == "__main__":  # pragma: no cover
    """Everything outside this block will be executed by every worker-thread, while this block is only run on the 
    main thread. Every object that is later passed to a worker must be pickle-able, that's why we 
    initialise everything that is not pickle-able before this point. Especially the DEAP-toolbox's creator-object is not 
    pickle-able. 
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args = parse_args()
    experiment = Experiment(configuration=ConfigReader.config_from_file(args.configuration),
                            result_path=args.result_path, from_checkpoint=args.from_checkpoint,
                            processing_framework=args.processing_framework, number_of_workers=args.num_workers)

    os.mkdir(args.result_path)
    experiment.run()
