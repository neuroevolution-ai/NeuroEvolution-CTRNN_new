import argparse
import os
from datetime import datetime

from tools.experiment import Experiment
from tools.helper import config_from_file


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='train CTRNN')
    parser.add_argument('--from-checkpoint', metavar='dir', type=str,
                        help='continues training from a checkpoint', default=None)
    parser.add_argument('--configuration', metavar='dir', type=str,
                        help='use an alternative configuration file', default='configurations/cma_es_basic.json')
    parser.add_argument('--result-path', metavar='dir', type=os.path.abspath,
                        help='use an alternative path for simulation results',
                        default=os.path.join("results", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
                        )
    return parser.parse_args(args)


args = parse_args()
experiment = Experiment(configuration=config_from_file(args.configuration), result_path=args.result_path,
                        from_checkpoint=args.from_checkpoint)

if __name__ == "__main__":
    """Everything outside this block will be executed by every scoop-worker, while this block is only run on the 
    main thread. Every object that is later passed to a worker must be pickle-able, that's why we 
    initialise everything that is not pickle-able before this point. Especially the DAEP-toolbox is not 
    pickle-able. 
    """
    os.makedirs(args.result_path)
    experiment.run()
