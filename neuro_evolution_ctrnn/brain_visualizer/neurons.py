import pygame
import logging
from brain_visualizer.color import Colors


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
            if colorVal > 1:
                colorVal = 1
                Neurons.color_logging(self, minMax)
            if colorVal < -1:
                colorVal = 1
                Neurons.color_logging(self, minMax)

            if colorVal <= 0:
                # grau zu hell
                interpolierteFarbe = Colors.interpolate_color(grau, hell, abs(colorVal))
                textSurface = self.myfont.render(('%.5s' % val), False, self.black)
            else:
                # grau zu grün
                interpolierteFarbe = Colors.interpolate_color(grau, grell, colorVal)
                textSurface = self.myfont.render(('%.5s' % val), False, self.white)

            # Draw Circle and Text
            pygame.draw.circle(self.screen, interpolierteFarbe, (pos_x, pos_y), radius)
            if self.neuronText == True:
                self.screen.blit(textSurface, ((pos_x - 16), (pos_y - 7)))

    def color_logging(self, minMax):
        if minMax == self.colorClippingRange[0]:
            var = "Input"
        if minMax == self.colorClippingRange[1]:
            var = "Graph"
        if minMax == self.colorClippingRange[2]:
            var = "Output"
        logging.warning("Please Increase Clipping Range for: " + var)
