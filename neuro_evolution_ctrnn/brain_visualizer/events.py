import pygame
import math

class Events():
    ######### Method to get the Number of the Neuron back if Mousclick on it
    def drawNeuronNumber(self, numberInputNeurons, numberNeurons, numberOutputNeurons, obPositionsDict, graphPositionsDict, outputPositionsDict, mousePose):
        mouseFont = pygame.font.SysFont("Helvetica", 32)
        for i in range(numberInputNeurons):
            neuronPose = obPositionsDict[i]
            distance = math.sqrt(((mousePose[0] - neuronPose[0]) ** 2) + ((mousePose[1] - neuronPose[1]) ** 2))
            if distance < 30:
                mouseLabel = mouseFont.render("Ob-Neuron:" + str(i), 1, self.brightGrey)
                self.screen.blit(mouseLabel, (mousePose[0] + 20, mousePose[1] - 32))

        for i in range(numberNeurons):
            neuronPose = graphPositionsDict[i]
            distance = math.sqrt(((mousePose[0] - neuronPose[0]) ** 2) + ((mousePose[1] - neuronPose[1]) ** 2))
            if distance < 30:
                mouseLabel = mouseFont.render("Neuron:" + str(i), 1, self.brightGrey)
                self.screen.blit(mouseLabel, (mousePose[0] + 20, mousePose[1] - 32))

        for i in range(numberOutputNeurons):
            neuronPose = outputPositionsDict[i]
            distance = math.sqrt(((mousePose[0] - neuronPose[0]) ** 2) + ((mousePose[1] - neuronPose[1]) ** 2))
            if distance < 30:
                mouseLabel = mouseFont.render("Output-Neuron:" + str(i), 1, self.brightGrey)
                self.screen.blit(mouseLabel, (mousePose[0] - 250, mousePose[1] - 32))

    ######### Method to get the Number of the Neuron back if Mousclick on it
    def getNeuronOnClick(self, numberNeurons, graphPositionsDict, mousePose):
        for i in range(numberNeurons):
            neuronPose = graphPositionsDict[i]
            distance = math.sqrt(((mousePose[0] - neuronPose[0]) ** 2) + ((mousePose[1] - neuronPose[1]) ** 2))
            if distance < 30:
                neuron = i
                return neuron

    def changeNeuronPos(self, neuron, mousePose, graphPositionsDict):
        graphPositionsDict[neuron] = (mousePose[0], mousePose[1])



