import pygame
import math

class Events():
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



