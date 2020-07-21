import pygame
import numpy
import math

class Weights():
    def draw(self, startPosDict, endPosDict, weightMatrix, positiveWeights, negativeWeights, direction):
        for (startNeruon, endNeuron), weight in numpy.ndenumerate(weightMatrix):
            if weight != 0:
                if (weight > 0.0 and positiveWeights) or (weight < 0.0 and negativeWeights):
                    startPos = startPosDict[startNeruon]
                    startPosX = int(startPos[0])
                    startPosY = int(startPos[1])

                    endPos = endPosDict[endNeuron]
                    endPosX = int(endPos[0])
                    endPosY = int(endPos[1])

                    if weight > 0.0:
                        weightColor = self.colorPositiveWeight
                    else:
                        weightColor = self.colorNegativeWeight

                    if direction:
                        # Winkel der Linien zwischen den beiden Punkten zur x-Achse
                        rotation = math.atan2((endPos[1] - startPos[1]), (endPos[0] - startPos[0]))
                        # Punkt, Winkel und Länge der Linie für Endpunkt des Pfeils
                        arrowLength = (-1 * (self.neuronRadius + 15))
                        arrowEnd = (endPos[0] + arrowLength * math.cos(rotation), endPos[1] + arrowLength * math.sin(rotation))
                        if rotation != 0:
                            Weights.arrow(self, self.screen, weightColor, weightColor, startPos, arrowEnd, 10, int(abs(weight)) + self.weightVal)
                    elif not direction:
                        pygame.draw.line(self.screen, weightColor, (startPosX, startPosY), (endPosX, endPosY), int(abs(weight)) + self.weightVal)

    def arrow(self, screen, color, tricolor, start, end, trirad, width):
        pygame.draw.line(screen, color, start, end, width)
        rotation = math.degrees(math.atan2(start[1] - end[1], end[0] - start[0])) + 90
        pygame.draw.polygon(screen, tricolor, (
        (end[0] + trirad * math.sin(math.radians(rotation)), end[1] + trirad * math.cos(math.radians(rotation))), (
        end[0] + trirad * math.sin(math.radians(rotation - 120)),
        end[1] + trirad * math.cos(math.radians(rotation - 120))), (
        end[0] + trirad * math.sin(math.radians(rotation + 120)),
        end[1] + trirad * math.cos(math.radians(rotation + 120)))))
