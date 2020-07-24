from brain_visualizer.position import Positions

import pygame, sys
import math
from pygame.locals import *

class Events():
    def handleEvents(self, event, inputPositionsDict, outputPositionsDict):
        try:
            global clickedNeuron
            if event.type == QUIT:
                pygame.quit()
                Positions.clearJSON(self)
                sys.exit()
            if event.type == MOUSEMOTION:
                Events.drawNeuronNumber(self, inputPositionsDict, self.graphPositionsDict, outputPositionsDict,
                                        pygame.mouse.get_pos())
            if event.type == MOUSEBUTTONDOWN:
                clickedNeuron = Events.getNeuronOnClick(self, self.graphPositionsDict, pygame.mouse.get_pos())
            if event.type == MOUSEBUTTONUP and clickedNeuron != None:
                Events.changeNeuronPos(self, clickedNeuron, pygame.mouse.get_pos(), self.graphPositionsDict)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    Positions.clearJSON(self)
                    sys.exit()
                if event.key == pygame.K_e:
                    self.weightVal = self.weightVal - 1
                if event.key == pygame.K_r:
                    self.weightVal = self.weightVal + 1
                if event.key == pygame.K_d:
                    if self.neuronRadius > 5:
                        self.neuronRadius = self.neuronRadius - 5
                        print(self.neuronRadius)
                if event.key == pygame.K_f:
                    self.neuronRadius = self.neuronRadius + 5
                if event.key == pygame.K_s:
                    if self.neuronText:
                        self.neuronText = False
                    else:
                        self.neuronText = True
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
                if event.key == pygame.K_g:
                    if self.weightsDirection:
                        self.weightsDirection = False
                    else:
                        self.weightsDirection = True
                if event.key == pygame.K_q:
                    if self.inputWeights:
                        self.inputWeights = False
                    else:
                        self.inputWeights = True
                if event.key == pygame.K_z:
                    if self.outputWeights:
                        self.outputWeights = False
                    else:
                        self.outputWeights = True
                if event.key == pygame.K_SPACE:
                    pause = True
                    pygame.event.clear(KEYDOWN)
                    while pause:
                        for event in pygame.event.get():
                            if event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_c:
                                    pause = False

        except AttributeError:
            print("Failure on Pygame-Event")



    ######### Method to get the Number of the Neuron back when hover
    def drawNeuronNumber(self, obPositionsDict, graphPositionsDict, outputPositionsDict, mousePose):
        mouseFont = pygame.font.SysFont("Helvetica", 32)

        hoveredInputNeuron = Events.getNeuronWhenHover(self, obPositionsDict, mousePose)
        if hoveredInputNeuron != None:
            mouseLabel = mouseFont.render("Ob-Neuron:" + str(hoveredInputNeuron), 1, self.brightGrey)
            self.screen.blit(mouseLabel, (mousePose[0] + 20, mousePose[1] - 32))

        hoveredGraphNeuron = Events.getNeuronWhenHover(self, graphPositionsDict, mousePose)
        if hoveredGraphNeuron != None:
            mouseLabel = mouseFont.render("Neuron:" + str(hoveredGraphNeuron), 1, self.brightGrey)
            self.screen.blit(mouseLabel, (mousePose[0] + 20, mousePose[1] - 32))

        hoveredOutputNeuron = Events.getNeuronWhenHover(self, outputPositionsDict, mousePose)
        if hoveredOutputNeuron != None:
            mouseLabel = mouseFont.render("Output-Neuron:" + str(hoveredOutputNeuron), 1, self.brightGrey)
            self.screen.blit(mouseLabel, (mousePose[0] - 250, mousePose[1] - 32))


    def getNeuronWhenHover(self, positionsDict, mousePose):
        maxDistance = 30
        for i in range(len(positionsDict)):
            hoveredNeuron = None
            neuronPose = positionsDict[i]
            distance = math.sqrt(((mousePose[0] - neuronPose[0]) ** 2) + ((mousePose[1] - neuronPose[1]) ** 2))
            if distance < maxDistance:
                hoveredNeuron = i
            if hoveredNeuron != None:
                return hoveredNeuron


    ######### Method to get the Number of the Neuron back if Mousclick on it
    def getNeuronOnClick(self, graphPositionsDict, mousePose):
        for i in range(len(graphPositionsDict)):
            neuronPose = graphPositionsDict[i]
            distance = math.sqrt(((mousePose[0] - neuronPose[0]) ** 2) + ((mousePose[1] - neuronPose[1]) ** 2))
            if distance < 30:
                neuron = i
                return neuron

    def changeNeuronPos(self, neuron, mousePose, graphPositionsDict):
        graphPositionsDict[neuron] = (mousePose[0], mousePose[1])



