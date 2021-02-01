import os
import logging
import argparse
import subprocess
import json
import tempfile
from datetime import datetime
from tools.helper import sample_from_design_space

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

parser = argparse.ArgumentParser(
    description='generate multiple experiments from design space and execute them')

parser.add_argument('--design-space', type=str, dest='design_space_path',
                    help='path to design-space.json',
                    default='configurations/cma_es_basic_design_space.json')
parser.add_argument('--result-base-path', type=str, dest='result_base_path',
                    help='path to where the results will be stored',
                    default=os.path.join('..', 'CTRNN_Simulation_Results', 'data'))
parser.add_argument('--max-experiments', type=int, dest='max_experiments',
                    help='how many experiments should be ran?',
                    default=100)

params = parser.parse_args()


for i in range(params.max_experiments):

    logging.info("Starting experiment number: " + str(i) + " out of " + str(params.max_experiments))
    with open(params.design_space_path, "r") as read_file:
        design_space = json.load(read_file)
    config = sample_from_design_space(design_space)
    result_path = os.path.join(params.result_base_path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    with tempfile.TemporaryDirectory() as tmpdir:
        logging.info('created temporary directory' + tmpdir)
        logging.info('result path: ' + result_path)
        logging.info('config: ' + str(config))
        conf_path = os.path.join(tmpdir, 'Configuration.json')

        with open(conf_path, 'w') as conf_file:
            json.dump(config, conf_file)

        logging.info('starting experiment...')
        subprocess.call(["python",
                         "neuro_evolution_ctrnn/train.py",
                         "--configuration", conf_path,
                         "--result-path", result_path])
        logging.info('... experiment done')

print("done")
