import os
import json
import pickle
from tools.helper import walk_dict
import numpy as np
import subprocess


class ResultHandler(object):

    def __init__(self, result_path, neural_network_type, config_raw):
        self.result_path = result_path
        self.nn_type = neural_network_type
        self.config_raw = config_raw
        self.result_hof = None
        self.result_log = None
        self.result_time_elapsed = None

        self.git_head = subprocess.check_output(["git", "rev-parse", "--verify", "HEAD"])
        self.git_diff = subprocess.check_output(["git", "diff", "HEAD"])

    def check_path(self):
        # checking before hand is not pythonic, but problems would a lot of processing time go to waste
        if not os.path.isdir(self.result_path):
            raise RuntimeError("result path '" + self.result_path + "' is not a directory")

        if len(os.listdir(self.result_path)) != 0:
            raise RuntimeError("result path '" + self.result_path + "' is not empty")

        if not os.access(self.result_path, os.W_OK):
            raise RuntimeError("result path '" + self.result_path + "' is not writable")

    def write_result(self, hof, log, time_elapsed: float, individual_size: int, input_space: np.shape, output_space):
        # store results in object, so it can be accept directly by other modules
        self.result_hof = hof
        self.result_log = log
        self.result_time_elapsed = time_elapsed

        print("output directory: " + str(self.result_path))
        with open(os.path.join(self.result_path, 'Configuration.json'), 'w') as outfile:
            # The indent attribute will pretty print the configuration
            json.dump(self.config_raw, outfile, ensure_ascii=False, indent=4)

        with open(os.path.join(self.result_path, 'HallOfFame.pickle'), "wb") as fp:
            pickle.dump(hof, fp)
        with open(os.path.join(self.result_path, 'Log.json'), 'w') as outfile:
            json.dump(log, outfile)
        with open(os.path.join(self.result_path, 'Log.pkl'), 'wb') as pk_file:
            pickle.dump(log, pk_file)

        with open(os.path.join(self.result_path, 'git.diff'), 'wb') as diff_file:
            diff_file.write(self.git_diff)

        with open(os.path.join(self.result_path, 'Log.txt'), 'w') as write_file:
            def write(key, value, depth, is_leaf):
                pad = ""
                for x in range(depth):
                    pad = pad + "\t"
                if is_leaf:
                    write_file.write(pad + key + ": " + str(value))
                else:
                    write_file.write(pad + key)
                write_file.write('\n')

            walk_dict(self.config_raw, write)

            write_file.write('\n')
            write_file.write('Genome Size: {:d}\n'.format(individual_size))
            write_file.write('Inputs: {:s}\n'.format(str(input_space)))
            write_file.write('Outputs: {:s}\n'.format(str(output_space)))
            write_file.write('Commit: {:s}\n'.format(str(self.git_head.decode("utf-8") )))
            write_file.write('\n')
            dash = '-' * 80
            write_file.write(dash + '\n')
            write_file.write(
                '{:<8s}{:<12s}{:<16s}{:<16s}{:<16s}{:<16s}\n'.format('gen', 'nevals', 'avg', 'std', 'min', 'max'))
            write_file.write(dash + '\n')

            # Write data for each episode
            for idx, line in enumerate(log):
                if log.chapters:
                    avg = log.chapters["fitness"][idx]["avg"]
                    std = log.chapters["fitness"][idx]["std"]
                    min = log.chapters["fitness"][idx]["min"]
                    max = log.chapters["fitness"][idx]["max"]
                else:
                    avg = line["avg"]
                    std = line["std"]
                    min = line["min"]
                    max = line["max"]

                write_file.write(
                    '{:<8d}{:<12d}{:<16.2f}{:<16.2f}{:<16.2f}{:<16.2f}\n'.format(line['gen'], line['nevals'],
                                                                                 avg, std, min, max))

            # Write elapsed time
            write_file.write("\nTime elapsed: %.4f seconds" % (time_elapsed))
