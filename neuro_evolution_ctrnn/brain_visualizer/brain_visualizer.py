from brain_visualizer.position import Positions
from brain_visualizer.weights import Weights
from brain_visualizer.neurons import Neurons
from brain_visualizer.events import Events
from brain_visualizer.color import Colour
from tools.configurations import IBrainCfg
from brains.i_brain import IBrain


import json
import pygame, sys
import numpy as np
import os
from typing import Tuple


# class BrainVisualizerHandler(object):
#     def __init__(self):
#         self.current_visualizer = None
#
#     # colorClippingRange for colorClipping Input [0], Graph [1] and Output [2]
#     def launch_new_visualization(self, brain, brain_config, width=1800, height=800, displayColor=(0, 0, 0),
#                                  colorClippingRange=(1, 1, 1), neuronRadius=30):
#         self.current_visualizer = PygameBrainVisualizer(brain, width, height, displayColor, neuronRadius,
#                                                         colorClippingRange, brain_config=brain_config)
#         return self.current_visualizer


class BrainVisualizer:
    def __init__(self, brain: IBrain, brain_config: IBrainCfg, env_id: str, width: int, height: int, displayColor: Tuple[int, int, int], neuronRadius: int, colorClippingRange: Tuple[int, int, int]):
        self.brain = brain
        self.brain_config = brain_config

        # Get directory of results and read Configuration file
        # self.environment = None
        # list = sys.argv
        # for i in list:
        #     if i.startswith('results/data/'):
        #         index = list.index(i)
        #         dir = sys.argv[index]
        #         with open(dir + "/Configuration.json", "r", encoding='utf-8') as read_file:
        #             file = json.load(read_file)
        #             self.environment = file["environment"]
        #             read_file.close()
        # if self.environment == None:
        #     self.environment = "Couldn't read Configuration.json"
        self.environment = env_id

        # Initial pygame module
        successes, failures = pygame.init()
        if failures:
            print("{0} successes and{1} failures".format(successes, failures))

        # Set position of screen (x, y) & create screen (length, width)
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (2500, 510)  # for a fixed position of the window
        self.screen = pygame.display.set_mode([width, height])
        self.w, self.h = pygame.display.get_surface().get_size()

        # Give it a name
        pygame.display.set_caption('Neurobotics - Brain Visualizer')

        self.kitLogo = pygame.image.load("kit_grey_50.png")
        self.kitLogo.convert()
        self.kitRect = self.kitLogo.get_rect()
        self.kitRect.center = 90, 30

        # initialize & set font
        pygame.font.init()
        self.myfont = pygame.font.SysFont("Helvetica", 14)

        ##### Dictionary Graph Neurons
        ##### Create Graph with Spring layout and get Positions of Neurons back
        self.graphPositionsDict = Positions.getGraphPositions(self)

        # define all colors
        Colour.colors(self, displayColor)
        self.colorClippingRange = colorClippingRange

        # variables for events
        self.positiveWeights = True
        self.negativeWeights = True
        self.weightsDirection = False
        self.inputWeights = True
        self.outputWeights = True
        self.weightVal = 0  # Defines how many connections will be drawn, default: every connection
        self.neuronRadius = neuronRadius
        self.neuronText = True
        self.clickedNeuron = None

    def process_update(self, in_values, out_values):
        # Fill screen with color
        self.screen.fill(self.displayColor)

        # Draw Rect for Logo and Blit Logo on the screen
        pygame.draw.rect(self.screen, self.darkGrey, (0, 0, self.w, 60))
        self.screen.blit(self.kitLogo, self.kitRect)

        if self.brain_config.use_bias:
            in_values = np.r_[in_values, [1]]

        ##### Number Neurons
        numberInputNeurons = len(in_values)
        numberNeurons = len(self.brain.W.todense())
        numberOutputNeurons = 1 if isinstance(out_values, np.int64) else len(out_values)

        ########## Draw Legend
        PygameBrainVisualizer.render_InfoText(self, [
            "Input Neurons: " + str(numberInputNeurons),
            "Graph Neurons: " + str(numberNeurons),
            "Output Neurons: " + str(numberOutputNeurons)],
                                              ((self.w / 4) - 80), 5, 18)

        PygameBrainVisualizer.render_InfoText(self, [
            "Positive/Negative Weights [t,w] : " + str(self.positiveWeights) + " / " + str(self.negativeWeights),
            "Input/Output Weights [q,z] : " + str(self.inputWeights) + " / " + str(self.outputWeights),
            "Direction [g] : " + str(self.weightsDirection)],
                                              ((self.w / 2) - 130), 5, 18)

        if self.weightVal == 0:  # If weightVal is 0 every Connection will be drawn
            text = "all"
        else:
            text = str(self.weightVal)
        PygameBrainVisualizer.render_InfoText(self, [
            "Weights [e,r] : " + text,
            "Values [s] : " + str(self.neuronText),
            "Simulation : " + str(self.environment)],
                                              ((3 * self.w / 4) - 80), 5, 18)

        ########## Create Dictionaries with Positions
        ##### Input Dictionary
        inputPositionsDict = Positions.getInputOutputPositions(self, numberInputNeurons, True)

        ##### Dictionary Graph Neurons
        # --> self.graphPositionsDict

        ##### Output Dictionary
        outputPositionsDict = Positions.getInputOutputPositions(self, numberOutputNeurons, False)

        ########## Draw Weights
        ##### n-1 Linien pro Neuron ; Input zu Neuron
        if self.inputWeights:
            Weights.drawWeights(self, inputPositionsDict, self.graphPositionsDict, self.brain.V.todense().T,
                                self.positiveWeights, self.negativeWeights, self.weightsDirection)

        # ##### n-1 Linien pro Neuron ; Neuron zu Neuron
        Weights.drawWeights(self, self.graphPositionsDict, self.graphPositionsDict, self.brain.W.todense(),
                            self.positiveWeights, self.negativeWeights, self.weightsDirection)

        # ##### n-1 Linien pro Neuron ; Neuron zu Output
        if self.outputWeights:
            Weights.drawWeights(self, self.graphPositionsDict, outputPositionsDict, self.brain.T.todense(),
                                self.positiveWeights, self.negativeWeights, self.weightsDirection)

        # #### 1 Kreis pro Neuron ; Neuron zu sich selbst ; Radius +5 damit Kreis größer als Neuron ist
        Neurons.drawNeurons(self, self.graphPositionsDict, self.brain.W, 2, self.colorNegativeWeight,
                            self.colorNeutralWeight, self.colorPositiveWeight, self.neuronRadius + self.weightVal, True,
                            True)

        ########### Draw neurons
        ##### Draw Graph
        Neurons.drawNeurons(self, self.graphPositionsDict, self.brain.y, self.colorClippingRange[1],
                            self.colorNegNeuronGraph, self.colorNeutralNeuron, self.colorPosNeuronGraph,
                            self.neuronRadius - 3, False)

        ##### draw ob-values (input)
        # minMax = clipping Range
        Neurons.drawNeurons(self, inputPositionsDict, in_values, self.colorClippingRange[0], self.colorNegNeuronIn,
                            self.colorNeutralNeuron, self.colorPosNeuronIn, self.neuronRadius)

        ##### draw action-values (out)
        # minMax = clipping Range
        Neurons.drawNeurons(self, outputPositionsDict, out_values, self.colorClippingRange[2], self.colorNegNeuronOut,
                            self.colorNeutralNeuron, self.colorPosNeuronOut, self.neuronRadius)

        ######### Events: Close when x-Button, Show Number of Neuron when click on it
        for event in pygame.event.get():
            Events.handleEvents(self, event, inputPositionsDict, outputPositionsDict)

        # Updates the content of the window
        pygame.display.flip()

    def render_InfoText(self, list_of_strings, x_pos, initial_y_pos, y_step):
        y_pos = initial_y_pos
        for s in list_of_strings:
            textSurface = self.myfont.render(s, False, self.numColor)
            self.screen.blit(textSurface, (x_pos, y_pos))
            y_pos += y_step
