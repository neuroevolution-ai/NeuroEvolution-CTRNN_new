from neuro_evolution_ctrnn.brain_visualizer.position import Positions
from neuro_evolution_ctrnn.brain_visualizer.weights import Weights
from neuro_evolution_ctrnn.brain_visualizer.neurons import Neurons
from  neuro_evolution_ctrnn.brain_visualizer.color import Colour
from neuro_evolution_ctrnn.brain_visualizer.events import Events

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

        # Initial pygame module
        successes, failures = pygame.init()
        if failures:
            print("{0} successes and{1} failures".format(successes, failures))

        # Set position of screen (x, y) & create screen (length, width)
        # os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (115, 510)
        self.screen = pygame.display.set_mode([1500, 900])
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


        ##### Dictionary Graph Neurons
        ##### Create Graph with Spring layout and get Positions of Neurons back
        self.graphPositionsDict = Positions.getGraphPositions(self, self.w, self.h)

        # colors
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.grey = (169, 169, 169)
        self.brightGrey = (220, 220, 220)
        self.darkGrey = (79, 79, 79)
        # Set Color for Display
        self.displayColor = self.black
        # Set Color for Numbers
        self.numColor = (220, 220, 220)

        # variables
        self.positiveWeights = True
        self.negativeWeights = True
        self.weightsDirection = False
        self.weightVal = (-2)
        self.neuronRadius = 30



    def process_update(self, in_values, out_values):
        # TODO: Zweite Runde bug oder was ist das?
        # Fill screen with color
        self.screen.fill((self.displayColor))

        # Draw Rect
        pygame.draw.rect(self.screen, self.darkGrey, (0, 0, self.w, 60))

        # Blit Logo on the Screen
        self.screen.blit(self.neuroboticsLogo, self.neuroboticsRect)
        self.screen.blit(self.kitLogo, self.kitRect)

        # initialize & set font
        pygame.font.init()
        myfont = pygame.font.SysFont("Helvetica", 14)

        # Text
        textSurface = myfont.render("Positive Weights [t] : " + str(self.positiveWeights), False, self.numColor)
        self.screen.blit(textSurface, (((self.w / 2) - 80), 5))
        textSurface = myfont.render("Negative Weights [w] : " + str(self.negativeWeights), False, self.numColor)
        self.screen.blit(textSurface, (((self.w / 2) - 80), 23))
        textSurface = myfont.render("Direction [z] : " + str(self.weightsDirection), False, self.numColor)
        self.screen.blit(textSurface, (((self.w / 2) - 80), 41))

        if self.weightVal == 0:
            text = "all"
        else:
            text = str(self.weightVal)
        textSurface = myfont.render("Weights [e,r] : " + text, False, self.numColor)
        self.screen.blit(textSurface, (((3*self.w / 4) - 80), 23))

        ##### Number Neurons
        numberInputNeurons = len(in_values)
        numberNeurons = len(self.brain.W)
        numberOutputNeurons = len(out_values)


        ##### Dictionary Input Neurons
        obPositionsDict = Positions.getInputOutputPositions(self, numberInputNeurons, "input")

        ##### Dictionary Graph Neurons
        # --> self.graphPositionsDict

        ##### Dictionary Output Neurons
        outputPositionsDict = Positions.getInputOutputPositions(self, numberOutputNeurons, "output")


        # draw split lines
        # pygame.draw.line(self.screen, (255, 255, 255), ((w/6), 0), ((w/6), h), 1)
        # pygame.draw.line(self.screen, (255, 255, 255), (((5*w)/6), 0), (((5*w)/6), h), 1)
        #pygame.draw.line(self.screen, (255, 255, 255), (0, 60), (self.w, 60), 1)
        # pygame.draw.aaline(self.screen, (255, 255, 255), (0, 40), (self.w, 40), 3)


        ########## Weights
        # TODO: checken ob directions so stimmen
        ##### n-1 Linien pro Neuron ; Input zu Neuron
        textSurface = myfont.render("Input Neurons: " + str(numberInputNeurons), False, self.numColor)
        self.screen.blit(textSurface, (((self.w / 4) - 55), 5))

        Weights.draw(self, self.graphPositionsDict, obPositionsDict, self.brain.V, self.positiveWeights, self.negativeWeights, self.weightsDirection, False)


        ##### n-1 Linien pro Neuron ; Neuron zu Neuron
        textSurface = myfont.render("Graph Neurons: " + str(numberNeurons), False, self.numColor)
        self.screen.blit(textSurface, (((self.w / 4) - 55), 23))

        Weights.draw(self, self.graphPositionsDict, self.graphPositionsDict, self.brain.W, self.positiveWeights, self.negativeWeights, self.weightsDirection, False)


        ##### n-1 Linien pro Neuron ; Neuron zu Output
        textSurface = myfont.render("Output Neurons: " + str(numberOutputNeurons), False, self.numColor)
        self.screen.blit(textSurface, (((self.w / 4) - 55), 41))

        Weights.draw(self, self.graphPositionsDict, outputPositionsDict, self.brain.T, self.positiveWeights, self.negativeWeights, self.weightsDirection, True)



        ########### Draw neurons
        ##### Draw Graph
        # TODO: clipping range generisch
        # minMax = clipping Range; hell, grau, grell
        Neurons.draw(self, myfont, self.graphPositionsDict, self.brain.W, 8, (255, 218, 184), (91, 98, 89), (255, 142, 34), self.neuronRadius + 5, True)
        Neurons.draw(self, myfont, self.graphPositionsDict, self.brain.y, 0.1,  (187, 209, 251), (91, 98, 89), (7, 49, 129), self.neuronRadius - 5 + 5, False)

        ##### draw ob-values (input)
        # minMax = clipping Range; hell, grau, grell
        Neurons.draw(self, myfont, obPositionsDict, in_values, 1, (188, 255, 169), (91, 98, 89), (49, 173, 14), self.neuronRadius)

        ##### draw action-values (out)
        # minMax = clipping Range; hell, grau, grell
        Neurons.draw(self, myfont, outputPositionsDict, out_values, 1, (249, 200, 207), (91, 98, 89), (185, 19, 44), self.neuronRadius)


        ######### Events: Close when x-Button, Show Number of Neuron when click on it
        for event in pygame.event.get():
            global clickedNeuron
            if event.type == QUIT:
                pygame.quit()
                Positions.clearJSON(self)
                sys.exit()
            if event.type == MOUSEMOTION:
                Events.drawNeuronNumber(self, numberInputNeurons, numberNeurons, numberOutputNeurons, obPositionsDict, self.graphPositionsDict, outputPositionsDict, pygame.mouse.get_pos())
            if event.type == MOUSEBUTTONDOWN:
                clickedNeuron = Events.getNeuronOnClick(self, numberNeurons, self.graphPositionsDict, pygame.mouse.get_pos())
            if event.type == MOUSEBUTTONUP and clickedNeuron != None:
                Events.changeNeuronPos(self, clickedNeuron, pygame.mouse.get_pos(), self.graphPositionsDict)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_e:
                    self.weightVal = self.weightVal - 1
                if event.key == pygame.K_r:
                    self.weightVal = self.weightVal + 1
                if event.key == pygame.K_d:
                    self.neuronRadius= self.neuronRadius - 5
                if event.key == pygame.K_f:
                    self.neuronRadius = self.neuronRadius + 5
                if event.key == pygame.K_t:
                    if self.positiveWeights:
                        self.positiveWeights = False
                    else:
                        self.positiveWeights = True
                if event.key == pygame.K_w:
                    if self.negativeWeights:
                        self.negativeWeights = False
                    else:
                        self.negativeWeights = True
                if event.key == pygame.K_z:
                    if self.weightsDirection:
                        self.weightsDirection = False
                    else:
                        self.weightsDirection = True

        # Updates the content of the window
        pygame.display.flip()

