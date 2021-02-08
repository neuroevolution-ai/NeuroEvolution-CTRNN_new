#!/usr/bin/env python3

import pickle
import os
import json
import matplotlib.pyplot as plt
from operator import add, sub
from scipy.ndimage.filters import gaussian_filter1d
import logging
import numpy as np
import matplotlib
import tikzplotlib
from tap import Tap

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


class PlotArgs(Tap):
    # Note: Comments in lines become helpstring when called with --help
    # for details on parsing see: https://github.com/swansonk14/typed-argument-parser

    dir: str  # Directory path to the simulation result
    no_show: bool = False  # Open a matplotlib window to show the plot
    save_png: str = None  # A filename where the plot should be saved as png
    save_tikz: str = None  # A filename where the plot should be saved as tikz
    plot_novelty: bool = False  # include novelty in the plot?
    smooth: int = 0  # How strong should the lines be smoothed? (0 to disable)
    style: str = 'seaborn-paper'  # Which plot style should be used?
    tex_renderer: bool = False  # Use text to render plot?

    def configure(self):
        self.description = "plot the training graph with matplotlib"
        # positional argument:
        self.add_argument("dir")


args = PlotArgs(underscores_to_dashes=True).parse_args()


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


if args.tex_renderer:
    matplotlib.rcParams.update({
        "pgf.texsystem": "xelatex",
        'font.family': 'serif',
        'pgf.rcfonts': False,
        'text.usetex': True,
    })

with open(os.path.join(args.dir, "Log.pkl"), "rb") as read_file_log:
    log = pickle.load(read_file_log)

with open(os.path.join(args.dir, "Configuration.json"), "r") as read_file:
    conf = json.load(read_file)

if conf['brain']['type'] == 'CNN_CTRNN':
    nn = conf['brain']['ctrnn_conf']['number_neurons']
else:
    nn = conf['brain']['number_neurons']

params_display = conf['environment'] + "\n" + conf['brain']['type'] + " + " + conf['optimizer'][
    'type'].replace('_', ' ') + "\nneurons: " + str()

fig, ax1 = plt.subplots()
plt.style.use('seaborn-paper')

generations = [i for i in range(len(log))]
if "val_fit" in log[0]:
    val_fit = log.select('val_fit')
    my_plot(ax1, generations, val_fit, '-', color='orange', label="val_fit")
plot_chapter(ax1, log.chapters["fitness"], generations, ("green", "teal", "teal", 'blue'))

ax1.set_xlabel('Generations')
ax1.set_ylabel('Fitness')
ax1.legend(loc='upper left')
ax1.grid()
plt.title(os.path.basename(args.dir).replace('_', ' '))
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

if args.save_png:
    logging.info("saving plot to: " + str(args.save_png))
    plt.savefig(args.save_png)

if args.save_tikz:
    tikzplotlib.clean_figure(target_resolution=80)
    tikzplotlib.save(filepath=args.save_tikz, strict=True, axis_height='8cm',
                     axis_width='10cm')
if not args.no_show:
    plt.show()
