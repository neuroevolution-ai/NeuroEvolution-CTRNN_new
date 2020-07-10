import pygame

class Weights():
    def draw(self, positions0, positions1, weightMatrix, positiveWeights, negativeWeights, direction, output):
        for zeile in range(len(positions0)):
            for spalte in range(len(positions1)):
                weight = weightMatrix[zeile, spalte]
                if weight > 0.0 and positiveWeights:
                    position0 = positions0[zeile]
                    pos_x0 = int(position0[0])
                    pos_y0 = int(position0[1])

                    position1 = positions1[spalte]
                    pos_x1 = int(position1[0])
                    pos_y1 = int(position1[1])

                    if direction:
                        pygame.draw.line(self.screen, self.grey, (pos_x0, pos_y0), (pos_x1 - 7, pos_y1 + 7), int(weight) + self.weightVal)
                        pygame.draw.line(self.screen, self.grey, (pos_x0, pos_y0), (pos_x1 + 7, pos_y1 - 7), int(weight) + self.weightVal)
                    elif direction and output:
                        pygame.draw.line(self.screen, self.grey, (pos_x0 - 7, pos_y0 + 7), (pos_x1, pos_y1), int(weight) + self.weightVal)
                        pygame.draw.line(self.screen, self.grey, (pos_x0 + 7, pos_y0 - 7), (pos_x1, pos_y1), int(weight) + self.weightVal)
                    else:
                        pygame.draw.line(self.screen, self.grey, (pos_x0, pos_y0), (pos_x1, pos_y1), int(weight) + self.weightVal)


                if weight < 0.0 and negativeWeights:
                    position0 = positions0[zeile]
                    pos_x0 = int(position0[0])
                    pos_y0 = int(position0[1])

                    position1 = positions1[spalte]
                    pos_x1 = int(position1[0])
                    pos_y1 = int(position1[1])

                    weight = abs(weight)

                    if direction:
                        pygame.draw.line(self.screen, self.white, (pos_x0, pos_y0), (pos_x1 - 7, pos_y1 + 7), int(weight) + self.weightVal)
                        pygame.draw.line(self.screen, self.white, (pos_x0, pos_y0), (pos_x1 + 7, pos_y1 - 7), int(weight) + self.weightVal)
                    elif direction and output:
                        pygame.draw.line(self.screen, self.grey, (pos_x0 - 7, pos_y0 + 7), (pos_x1, pos_y1), int(weight) + self.weightVal)
                        pygame.draw.line(self.screen, self.grey, (pos_x0 + 7, pos_y0 - 7), (pos_x1, pos_y1), int(weight) + self.weightVal)
                    else:
                        pygame.draw.line(self.screen, self.white, (pos_x0, pos_y0), (pos_x1, pos_y1), int(weight) + self.weightVal)
