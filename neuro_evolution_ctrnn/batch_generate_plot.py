import os
import logging
import subprocess
from tap import Tap

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


class BatchPlotArgs(Tap):
    # Note: Comments in lines become helpstring when called with --help
    # for details on parsing see: https://github.com/swansonk14/typed-argument-parser

    basedir: str  # path to the parent-folder of simulation results
    filename: str = "plot.png"  # the filename under which to store the plots
    handle_existing: str = "skip"  # what to do when the plot-file already exists? options: skip, replace, raise
    smooth: int = 0  # How strong should the lines be smoothed? (0 to disable)

    def configure(self):
        self.description = "iterate over simulation folders and create a plot for each simulation run by" \
                           "by calling plot_experiment.py for each folder"

        # positional argument:
        # note: for reasons unknown positional arguments can not contain underscores when using underscores_to_dashes
        self.add_argument("basedir")

        # aliases
        self.add_argument("-e", "--handle_existing")


args = BatchPlotArgs(underscores_to_dashes=True).parse_args()

simulation_folders = [f.path for f in os.scandir(args.basedir) if f.is_dir()]

for sim in simulation_folders:
    file = os.path.join(sim, args.filename)
    if os.path.exists(file):
        if args.handle_existing == 'skip':
            logging.info("skipping because file already exist" + str(file))
            continue
        elif args.handle_existing == 'raise':
            raise RuntimeError("File already exists: " + str(file))
        elif args.handle_existing == 'replace':
            logging.info("replacing existing file: " + str(file))

    # instead of calling the other file, it would probably better to move the generation of the plot into a
    # separate module which is used both by plot_experiment.py and this script
    subprocess.run(["python", "neuro_evolution_ctrnn/plot_experiment.py",
                    sim,
                    "--no-show",
                    "--smooth", str(args.smooth),
                    "--save-png", file])
