import tkinter as tk
import pygame

from neuro_evolution_ctrnn.brain_visualizer.pygame_visualizer import Pygame
#from neuro_evolution_ctrnn.brain_visualizer.pyvis import Pyvis
#from neuro_evolution_ctrnn.brain_visualizer.graphTool import GraphTool
#from neuro_evolution_ctrnn.brain_visualizer.netwulf import Netwulf
from neuro_evolution_ctrnn.brain_visualizer.matplotlib import Matplotlib
#from neuro_evolution_ctrnn.brain_visualizer.graphics import Graphics
#from neuro_evolution_ctrnn.brain_visualizer.plotly import Plotly

import pygame, sys
from pygame.locals import *
import os
import networkx as nx
import json
from json import JSONEncoder
import numpy
import math
import matplotlib.pyplot as plt
import numpy as np
import time
from networkx.drawing.nx_agraph import graphviz_layout
from neuro_evolution_ctrnn.brain_visualizer.position import Position


class BrainVisualizerHandler(object):
    def __init__(self):
        self.current_visualizer = None

    def launch_new_visualization(self, brain):
        self.current_visualizer = PygameBrainVisualizer(brain)
        return self.current_visualizer


class PygameBrainVisualizer(object):
    def __init__(self, brain):
        self.brain = brain



    def process_update(self, in_values, out_values):
        Pygame.pygame(self, in_values, out_values)




class BrainVisualizer(object):

    def __init__(self, brain):
        self.brain = brain

        root = tk.Tk()
        self.root = root
        self.app = Window(root)
        root.wm_title("Tkinter window")
        root.geometry("200x200")

    def process_update(self, in_values, out_values):
        self.app.set_text("new states: \n" + str(self.brain.y))
        self.root.update_idletasks()
        self.root.update()


class Window(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.master = master
        self.pack(fill=tk.BOTH, expand=1)
        self.text = tk.StringVar()
        self.set_text("waiting for update...")
        tk.Label(self, textvariable=self.text).pack()

    def set_text(self, text):
        self.text.set(text)

