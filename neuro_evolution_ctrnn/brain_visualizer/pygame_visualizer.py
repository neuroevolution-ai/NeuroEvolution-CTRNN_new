import pygame
import os
import networkx as nx
import json
from json import JSONEncoder
import numpy
import matplotlib.pyplot as plt
import numpy as np
from neuro_evolution_ctrnn.brain_visualizer.matplotlib import Matplotlib

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

class Pygame():
    def pygame(self, in_values, out_values):

        #TODO: Schließen über das Kreuz im Fenster

        # Initial pygame module
        #successes, failures = pygame.init()
        #print("{0} successes and {1} failures".format(successes, failures))
        pygame.init()

        # Set position of screen (x, y) & create screen (length, width)
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (115, 510)
        screen = pygame.display.set_mode([1500, 500])

        w, h = pygame.display.get_surface().get_size()

        # Display Name
        pygame.display.set_caption('CTRNN visualizer')

        # colors
        black = (0, 0, 0)
        white = (255, 255, 255)

        # Set display color
        displayColor = 0, 0, 0

        screen.fill((displayColor))

        # Set Color for Numbers
        numColor = 255, 255, 255

        # initialize & set font
        pygame.font.init()
        myfont = pygame.font.SysFont("Arial", 15)

        #imort position
        #pos = Matplotlib.matplotlib(self)
        def Convert(lst):
            res_dct = {i : lst[i] for i in range(0, len(lst), 1)}
            return res_dct

        brainSate = self.brain.y
        brainWeight = self.brain.W
        brainStatedict = Convert(brainSate)

        #g = nx.Graph()
        #g.add_nodes_from(brainStatedict)

        #print(brainWeight[0,1])
        #G.add_edge(1, 2, weight=3)

        # [(0, 0, {'weight': -1.9011296238884632}), (0, 1, {'weight': 3.220957626055494}), (0, 4, {'weight': -0.6631949403544992}), (0, 14, {'weight': 4.642733924984142}),
        G = nx.from_numpy_array(brainWeight, parallel_edges=True)
        #print(G.edges(data=True))

        arrray = {0: (-0.2728846, 0.74711038), 1: (-0.31891225, 0.66547815), 2: (0.17179, 0.92973358),
                  3: (1., -0.2231467), 4: (0.38143834, -0.8415738), 5: (-0.60921009, 0.50518079),
                  6: (-0.37992527, -0.65073305), 7: (0.04851295, -0.24817101),
                  8: (-0.08354904, -0.74049539), 9: (0.67633978, 0.42055284), 10: (0.5041357, -0.49892226),
                  11: (0.34567744, 0.64230415), 12: (-0.5994108, 0.56297205),
                  13: (-0.13769502, -0.12910191), 14: (0.49246168, 0.73320535), 15: (0.7950964, 0.49450273),
                  16: (0.63158082, -0.84052595), 17: (0.62863304, 0.49440121),
                  18: (-0.73856454, -0.41477315), 19: (-0.69630721, 0.65250366),
                  20: (-0.11833097, -0.41475844), 21: (-0.36507268, -0.23317927),
                  22: (0.46646422, -0.7360597), 23: (-0.67845279, -0.42695827),
                  24: (-0.78669895, -0.09019501), 25: (-0.24630579, -0.73455162),
                  26: (0.97569744, 0.39554758), 27: (0.24129693, -0.72899607),
                  28: (-0.67049509, 0.41776016), 29: (-0.65730965, 0.290889)}

        fpath = "position.json"
        if os.path.isfile(fpath) and os.path.getsize(fpath) == 0:
            numpyData = nx.spring_layout(G)
            with open("position.json", "w") as write_file:
                json.dump(numpyData, write_file, cls=NumpyArrayEncoder)
                write_file.close()

        with open("position.json", "r") as read_file:
            decodedArray = json.load(read_file)

        print(decodedArray)
        pos = nx.spring_layout(G, pos=arrray)
        #nx.draw(G, pos)
        #nx.draw_networkx_labels(G, pos)
        #nx.draw_networkx(G, pos)
        #print(pos)
        #plt.show()

        #time.sleep(0.5)

        # numberNeurons = len(brainSate)
        # pos = {}
        # for zeile in range(int(numberNeurons/10)):
        #     for spalte in range(10):
        #         val = str(self.brain.y[((zeile*10) + spalte)])
        #         color = abs((self.brain.y[((zeile*10) + spalte)]) * 255) * 10
        #         textSurface = myfont.render(('%.5s' % val), False, numColor)
        #         pos_y = (zeile) * 200 + 50
        #         pos_x = int((spalte+1) * (1000/10)) + 200
        #
        #         pos[((zeile*10) + spalte)] = [pos_x, pos_y]


        # draw split lines
        pygame.draw.line(screen, (255, 255, 255), (250,0), (250,500), 1)
        pygame.draw.line(screen, (255, 255, 255), (1250, 0), (1250, 500), 1)

        ########### n-1 Linien pro Neuron ; Input zu Neuron
        numberNeurons = len(brainSate)
        numberObValues = len(in_values)
        textSurface = myfont.render("Input Neurons: " + str(numberNeurons), False, numColor)
        screen.blit(textSurface, (20, 10))

        obPositionsDict2 = {}
        for i in range(numberObValues):
            if i % 2:  # ungerade
                x_pos = 75
                y_pos = 50 + i * (400 / (numberObValues-1))
                obPositionsDict2[i] = [x_pos, y_pos]
            else:  # gerade
                x_pos = 125
                y_pos = 50 + i * (400 / (numberObValues-1))
                obPositionsDict2[i] = [x_pos, y_pos]

        for zeile in range(numberNeurons):
            for spalte in range(numberObValues):
                kantenGewicht = self.brain.V[zeile, spalte]
                if kantenGewicht > 0.0:
                    position0 = obPositionsDict2[spalte]
                    pos_x0 = int(position0[0])
                    pos_y0 = int(position0[1])
                    position1 = pos[zeile]
                    pos_x1 = int(position1[0] * 500) + 500 + 200
                    pos_y1 = int(position1[1] * (-250)) + 250
                    pygame.draw.line(screen, (255, 255, 255), (pos_x0, pos_y0), (pos_x1, pos_y1), int(kantenGewicht))


        ########### n-1 Linien pro Neuron ; Neuron zu Neuron
        textSurface = myfont.render("Neurons in Hidden Layer: " + str(numberObValues), False, numColor)
        screen.blit(textSurface, (260, 10))

        numberNeurons = len(brainSate) - 1
        for zeile in range(numberNeurons):
            for spalte in range(numberNeurons):
                kantenGewicht = self.brain.W[zeile, spalte]
                if kantenGewicht > 0.0:
                    position0 = pos[zeile]
                    pos_x0 = int(position0[0] * 500) + 500 + 200
                    pos_y0 = int(position0[1] * (-250)) + 250
                    position1 = pos[spalte]
                    pos_x1 = int(position1[0] * 500) + 500 + 200
                    pos_y1 = int(position1[1] * (-250)) + 250
                    pygame.draw.line(screen, (255, 255, 255), (pos_x0, pos_y0), (pos_x1, pos_y1), int(kantenGewicht))

        ########### n-1 Linien pro Neuron ; Neuron zu Output
        numberOutputValues = len(out_values)
        textSurface = myfont.render("Output Neurons: " + str(numberOutputValues), False, numColor)
        screen.blit(textSurface, (1260, 10))

        outputPositionsDict = {}
        for i in range(numberOutputValues):
            if i % 2:  # ungerade
                x_pos = 1250 + 75
                y_pos = 50 + i * (400 / (numberOutputValues-1))
                outputPositionsDict[i] = [x_pos, y_pos]
            else:  # gerade
                x_pos = 1250 + 125
                y_pos = 50 + i * (400 / (numberOutputValues-1))
                outputPositionsDict[i] = [x_pos, y_pos]

        for zeile in range(numberNeurons):
            for spalte in range(numberOutputValues):
                kantenGewicht = self.brain.T[zeile, spalte]
                if kantenGewicht > 0.0:
                    position0 = outputPositionsDict[spalte]
                    pos_x0 = int(position0[0])
                    pos_y0 = int(position0[1])
                    position1 = pos[zeile]
                    pos_x1 = int(position1[0] * 500) + 500 + 200
                    pos_y1 = int(position1[1] * (-250)) + 250
                    pygame.draw.line(screen, (255, 255, 255), (pos_x0, pos_y0), (pos_x1, pos_y1), int(kantenGewicht))

        ########### Draw neurons
        #TODO: Neuronen nicht übereinander zeichnen
        for neuron in range(numberNeurons + 1):
            position = pos[neuron]
            pos_x = int(position[0] * 500) + 500 + 200
            pos_y = int(position[1] * (-250)) + 250

            val = str(self.brain.y[neuron])
            color = abs((self.brain.y[neuron]) * 255) * 10
            #color = 255
            blue = (22,44,66)
            textSurface = myfont.render(('%.5s' % val), False, numColor)

            # Draw Circle and Text
            pygame.draw.circle(screen, (0, 0, color), (pos_x, pos_y), 30)
            #pygame.draw.circle(screen, blue, (pos_x, pos_y), 20)
            screen.blit(textSurface, ((pos_x - 16), (pos_y - 7)))

        ########### draw ob-values (in)
        for ob in range(numberObValues):
            #ob_values = in_values

            position = obPositionsDict2[ob]
            pos_x = int(position[0])
            pos_y = int(position[1])

            val = in_values[ob]
            #color = abs((self.brain.y[x]) * 255) * 10
            color = 255
            blue = (22,44,66)
            textSurface = myfont.render(('%.5s' % val), False, numColor)

            # Draw Circle and Text
            pygame.draw.circle(screen, (0, 0, color), (pos_x, pos_y), 20)
            #pygame.draw.circle(screen, blue, (pos_x, pos_y), 20)
            screen.blit(textSurface, ((pos_x - 16), (pos_y - 7)))

        ########### draw action-values (out)
        action_values = out_values
        for output in range(numberOutputValues):
            #ob_values = in_values

            position = outputPositionsDict[output]
            pos_x = int(position[0])
            pos_y = int(position[1])

            val = in_values[output]
            #color = abs((self.brain.y[x]) * 255) * 10
            color = 255
            blue = (22,44,66)
            textSurface = myfont.render(('%.5s' % val), False, numColor)

            # Draw Circle and Text
            pygame.draw.circle(screen, (0, 0, color), (pos_x, pos_y), 20)
            #pygame.draw.circle(screen, blue, (pos_x, pos_y), 20)
            screen.blit(textSurface, ((pos_x - 16), (pos_y - 7)))



        # Update the Screen
        pygame.display.flip()

        # Updates the content of the window
        pygame.display.flip()
