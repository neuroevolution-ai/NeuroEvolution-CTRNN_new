#!/usr/bin/env python3

import os
from datetime import datetime
import logging
from tap import Tap

from tools.experiment import Experiment
from tools.config_reader import ConfigReader

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


class TrainArgs(Tap):
    configuration: str  # path to configuration file. See directory "configurations" for example files
    result_path: os.path.abspath  # Use an alternative path for simulation results
    processing_framework: str = 'dask'  # Choose the framework used for the processing. Options are dask/mp/sequential
    num_workers: int = os.cpu_count()  # Specify the amount of workers for the computation
    from_checkpoint: str = None  # Continues training from a checkpoint. Expects path to checkpoint.pkl
    reset_hof: bool = False  # when loading from a checkpoint, should the HoF be resetted before continuing?
    checkpoint_to_result: bool = False  # Should the last checkpoint be stored in the result directory?

    def configure(self):
        self.description = 'Train CTRNN'
        # positional argument:
        self.add_argument('configuration')

        # aliases
        self.add_argument('-p', '--processing_framework')
        self.add_argument('-n', '--num_workers')

        # complex type and default
        self.add_argument("--result_path", type=os.path.abspath,
                          default=os.path.join("..", "CTRNN_Simulation_Results", "data",
                                               datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
        return self


if __name__ == "__main__":  # pragma: no cover
    """Everything outside this block will be executed by every worker-thread, while this block is only run on the 
    main thread. Every object that is later passed to a worker must be pickle-able, that's why we 
    initialise everything that is not pickle-able before this point. Especially the DEAP-toolbox's creator-object is not 
    pickle-able. 
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args = TrainArgs(underscores_to_dashes=True).parse_args()

    experiment = Experiment(configuration=ConfigReader.config_from_file(args.configuration),
                            result_path=args.result_path,
                            from_checkpoint=args.from_checkpoint,
                            processing_framework=args.processing_framework,
                            number_of_workers=args.num_workers,
                            reset_hof=args.reset_hof,
                            checkpoint_to_result=args.checkpoint_to_result)

    os.mkdir(args.result_path)
    experiment.run()
