from brain_visualizer.position import Positions
from brain_visualizer.weights import Weights
from brain_visualizer.neurons import Neurons
from brain_visualizer.events import Events

import json

import pygame, sys
from pygame.locals import *


class BrainVisualizerHandler(object):
    def __init__(self):
        self.current_visualizer = None

    def launch_new_visualization(self, brain):
        self.current_visualizer = PygameBrainVisualizer(brain)
        return self.current_visualizer


class PygameBrainVisualizer(object):
    def __init__(self, brain):
        self.brain = brain

        # Read Configuration file
        with open("../CTRNN_Simulation_Results/data/" + ((sys.argv[2])[-19:]) + "/Configuration.json", "r", encoding='utf-8') as read_file:
            file = json.load(read_file)
            self.environment = file["environment"]
            read_file.close()

        # Initial pygame module
        successes, failures = pygame.init()
        if failures:
            print("{0} successes and{1} failures".format(successes, failures))

        # Set position of screen (x, y) & create screen (length, width)
        # os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (115, 510)
        #self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.screen = pygame.display.set_mode([1900, 1000])
        self.w, self.h = pygame.display.get_surface().get_size()

        # Give it a name
        pygame.display.set_caption('Neurobotics - Brain Visualizer')

        self.neuroboticsLogo = pygame.image.load("neurobotics_50.png")
        self.neuroboticsLogo.convert()
        self.neuroboticsRect = self.neuroboticsLogo.get_rect()
        self.neuroboticsRect.center = self.w - 80,  30

        self.kitLogo = pygame.image.load("kit_grey_50.png")
        self.kitLogo.convert()
        self.kitRect = self.kitLogo.get_rect()
        self.kitRect.center = 90, 30

        # initialize & set font
        pygame.font.init()
        self.myfont = pygame.font.SysFont("Helvetica", 14)

        ##### Dictionary Graph Neurons
        ##### Create Graph with Spring layout and get Positions of Neurons back
        self.graphPositionsDict = Positions.getGraphPositions(self, self.w, self.h)

        # colors
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.grey = (169, 169, 169)
        self.brightGrey = (220, 220, 220)
        self.darkGrey = (79, 79, 79)
        # Color for Display
        self.displayColor = self.black
        # Color for Numbers
        self.numColor = (220, 220, 220)
        # Colors for Weights
        self.colorNegativeWeight = (185, 19, 44) # rot
        self.colorNeutralWeight = self.darkGrey
        self.colorPositiveWeight = (188, 255, 169)  # hellgrün
        # Colors Neutral Neurons
        self.colorNeutralNeuron = self.darkGrey
        # Color Neurons in Graph
        self.colorNegNeuronGraph = (187, 209, 251)  # Hellblau
        self.colorPosNeuronGraph = (7, 49, 129)  # Blau
        # Color Input Layer
        self.colorNegNeuronIn = (188, 255, 169)   # hellgrün
        self.colorPosNeuronIn = (49, 173, 14)   # grün
        # Color in Output Layer
        self.colorNegNeuronOut = (255, 181, 118)  # Hell-Orange
        self.colorPosNeuronOut = (255, 142, 46)    # orange


        # variables
        self.positiveWeights = True
        self.negativeWeights = True
        self.weightsDirection = False
        self.inputWeights = True
        self.outputWeights = True
        self.weightVal = 0 # Defines how many connections will be drawn, defaul: every connection
        self.neuronRadius = 30
        self.neuronText = True



    def process_update(self, in_values, out_values):
        # Fill screen with color
        self.screen.fill((self.displayColor))

        # Draw Rect for Logo and Blit Logo on the screen
        pygame.draw.rect(self.screen, self.darkGrey, (0, 0, self.w, 60))
        self.screen.blit(self.neuroboticsLogo, self.neuroboticsRect)
        self.screen.blit(self.kitLogo, self.kitRect)

        ##### Number Neurons
        numberInputNeurons = len(in_values)
        numberNeurons = len(self.brain.W.todense())
        numberOutputNeurons = len(out_values)

        ########## Draw Legend
        PygameBrainVisualizer.render_InfoText(self, [
            "Input Neurons: " + str(numberInputNeurons),
            "Graph Neurons: " + str(numberNeurons),
            "Output Neurons: " + str(numberOutputNeurons)],
                                              ((self.w / 4) - 80), 5, self.numColor, 18)

        PygameBrainVisualizer.render_InfoText(self, [
            "Positive/Negative Weights [t,w] : " + str(self.positiveWeights) + " / " + str(self.negativeWeights),
            "Input/Output Weights [q,z] : " + str(self.inputWeights) + " / " + str(self.outputWeights),
            "Direction [g] : " + str(self.weightsDirection)],
                        ((self.w / 2) - 130), 5, self.numColor, 18)

        if self.weightVal == 0:     # If weightVal is 0 every Connection will be drawn
            text = "all"
        else:
            text = str(self.weightVal)
        PygameBrainVisualizer.render_InfoText(self, [
            "Weights [e,r] : " + text,
            "Values [s] : " + str(self.neuronText),
            "Simulation : " + str(self.environment)],
                        ((3 * self.w / 4) - 80), 5, self.numColor, 18)


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
            Weights.drawWeights(self, inputPositionsDict, self.graphPositionsDict, self.brain.V.todense().T, self.positiveWeights, self.negativeWeights, self.weightsDirection)

        # ##### n-1 Linien pro Neuron ; Neuron zu Neuron
        Weights.drawWeights(self, self.graphPositionsDict, self.graphPositionsDict, self.brain.W.todense(), self.positiveWeights, self.negativeWeights, self.weightsDirection)

        # ##### n-1 Linien pro Neuron ; Neuron zu Output
        if self.outputWeights:
            Weights.drawWeights(self, self.graphPositionsDict, outputPositionsDict, self.brain.T.todense(), self.positiveWeights, self.negativeWeights, self.weightsDirection)

        # #### 1 Kreis pro Neuron ; Neuron zu sich selbst ; Radius +5 damit Kreis größer als Neuron ist
        Neurons.drawNeurons(self, self.graphPositionsDict, self.brain.W, 2, self.colorNegativeWeight, self.colorNeutralWeight, self.colorPositiveWeight, self.neuronRadius + self.weightVal, True, True)


        ########### Draw neurons
        ##### Draw Graph
        Neurons.drawNeurons(self, self.graphPositionsDict, self.brain.y, 1,  self.colorNegNeuronGraph, self.colorNeutralNeuron, self.colorPosNeuronGraph, self.neuronRadius-3, False)

        ##### draw ob-values (input)
        # minMax = clipping Range
        Neurons.drawNeurons(self, inputPositionsDict, in_values, 1, self.colorNegNeuronIn, self.colorNeutralNeuron, self.colorPosNeuronIn, self.neuronRadius)

        ##### draw action-values (out)
        # minMax = clipping Range
        Neurons.drawNeurons(self, outputPositionsDict, out_values, 1, self.colorNegNeuronOut, self.colorNeutralNeuron, self.colorPosNeuronOut, self.neuronRadius)


        ######### Events: Close when x-Button, Show Number of Neuron when click on it
        for event in pygame.event.get():
            Events.handleEvents(self, event, inputPositionsDict, outputPositionsDict)


        # Updates the content of the window
        pygame.display.flip()

    def render_InfoText(self, list_of_strings, x_pos, initial_y_pos, color, y_step):
        y_pos = initial_y_pos
        for s in list_of_strings:
            textSurface = self.myfont.render(s, False, self.numColor)
            self.screen.blit(textSurface, (x_pos, y_pos))
            y_pos += y_step
