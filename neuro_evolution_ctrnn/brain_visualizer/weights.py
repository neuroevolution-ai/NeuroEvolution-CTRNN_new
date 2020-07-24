import pygame
import numpy
import math

class Weights():
    def drawWeights(self, startPosDict, endPosDict, weightMatrix, positiveWeights, negativeWeights, direction):
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

                    width = int(abs(weight)/0.5) + self.weightVal - 6
                    #width = int(abs(weight)) + self.weightVal
                    if self.weightVal == 0 and width < 1:
                        width = 1

                    if direction:
                        # Winkel der Linien zwischen den beiden Punkten zur x-Achse
                        rotation = math.atan2((endPos[1] - startPos[1]), (endPos[0] - startPos[0]))
                        # Punkt, Winkel und Länge der Linie für Endpunkt des Pfeils
                        trirad = 5 + width
                        arrowLength = (-1 * (self.neuronRadius + trirad + 5))

                        arrowEnd = (endPos[0] + arrowLength * math.cos(rotation), endPos[1] + arrowLength * math.sin(rotation))
                        if rotation != 0:
                            Weights.arrow(self, self.screen, weightColor, weightColor, startPos, arrowEnd, trirad, width)
                    elif not direction:
                        pygame.draw.line(self.screen, weightColor, (startPosX, startPosY), (endPosX, endPosY), width)

    def arrow(self, screen, color, tricolor, start, end, trirad, width):
        if width >= 1:
            #trirad += width
            pygame.draw.line(screen, color, start, end, width)
            rotation = math.degrees(math.atan2(start[1] - end[1], end[0] - start[0])) + 90
            pygame.draw.polygon(screen, tricolor, (
            (end[0] + trirad * math.sin(math.radians(rotation)), end[1] + trirad * math.cos(math.radians(rotation))), (
            end[0] + trirad * math.sin(math.radians(rotation - 120)),
            end[1] + trirad * math.cos(math.radians(rotation - 120))), (
            end[0] + trirad * math.sin(math.radians(rotation + 120)),
            end[1] + trirad * math.cos(math.radians(rotation + 120)))))
