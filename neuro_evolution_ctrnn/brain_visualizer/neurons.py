import pygame
from brain_visualizer.color import Colour

class Neurons():
    def drawNeurons(self, positions, valueDict, minMax, hell, grau, grell, radius, matrix=False, weightNeruon=False):
        for neuron in range(len(positions)):
            position = positions[neuron]
            pos_x = int(position[0])
            pos_y = int(position[1])

            if matrix == True:
                val = valueDict[neuron, neuron]
                colorVal = valueDict[neuron, neuron] / minMax
            else:
                val = valueDict[neuron]
                colorVal = valueDict[neuron] / minMax

            if weightNeruon:
                radius = radius + int(abs(val))

            # Damit das Programm nicht abbricht wenn klipping range nicht passt
            # TODO: Das könnte man loggen
            if colorVal > 1:
                colorVal = 1
            if colorVal < -1:
                colorVal = -1

            if colorVal <= 0:
                # grau zu hell
                interpolierteFarbe = Colour.interpolateColor(grau, hell, abs(colorVal))
                textSurface = self.myfont.render(('%.5s' % val), False, self.black)
            else:
                # grau zu grün
                interpolierteFarbe = Colour.interpolateColor(grau, grell, colorVal)
                textSurface = self.myfont.render(('%.5s' % val), False, self.white)

            # Draw Circle and Text
            pygame.draw.circle(self.screen, interpolierteFarbe, (pos_x, pos_y), radius)
            if self.neuronText == True:
                self.screen.blit(textSurface, ((pos_x - 16), (pos_y - 7)))
