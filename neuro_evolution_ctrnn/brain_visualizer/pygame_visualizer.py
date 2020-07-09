import pygame, sys
from pygame.locals import *
import os
import networkx as nx
import json
from json import JSONEncoder
import numpy
import math
import matplotlib.pyplot as plt
import numpy as np
import time
from networkx.drawing.nx_agraph import graphviz_layout
from neuro_evolution_ctrnn.brain_visualizer.position import Position


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

class Pygame():

    def pygame(self, in_values, out_values):
        # Initial pygame module
        #successes, failures = pygame.init()
        #print("{0} successes and {1} failures".format(successes, failures))
        pygame.init()

        # Set position of screen (x, y) & create screen (length, width)
        #os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (115, 510)
        screen = pygame.display.set_mode([1500, 800])
        w, h = pygame.display.get_surface().get_size()

        # Display Name
        pygame.display.set_caption('CTRNN visualizer')

        # colors
        black = (0, 0, 0)
        white = (255, 255, 255)
        grey = (169,169,169)
        brightGrey = (220,220,220)
        darkGrey = (79,79,79)
        # Set Color for Display
        displayColor = 0, 0, 0
        # Set Color for Numbers
        numColor = (220, 220, 220)
        numColor = darkGrey

        # Fill screen with color
        screen.fill((displayColor))

        # initialize & set font
        pygame.font.init()
        myfont = pygame.font.SysFont("Helvetica", 15)


        def Convert(lst):
            res_dct = {i : "{my_weight: " + str(lst[i]) + "}" for i in range(0, len(lst), 1)}
            return res_dct

        # variables
        brainSate = self.brain.y
        brainWeight = self.brain.W
        # brainStatedict = Convert(brainWeight)

        ######### imort position
        #g = nx.Graph()
        #g.add_nodes_from(brainStatedict)

        # [(0, 0, {'weight': -1.9011296238884632}), (0, 1, {'weight': 3.220957626055494}), (0, 4, {'weight': -0.6631949403544992}), (0, 14, {'weight': 4.642733924984142}),
        #print(numpy.info(brainWeight))

        #testList = [(0, 0, {'my_weight': -1.9011296238884632}), (0, 1, {'my_weight': 3.220957626055494}), (0, 4, {'my_weight': -0.6631949403544992}), (0, 14, {'my_weight': 4.642733924984142}), (0, 16, {'my_weight': -5.6578376702341515}), (0, 26, {'my_weight': 1.293560810386954}), (0, 29, {'my_weight': -0.45573146626343664}), (0, 15, {'my_weight': 0.3027749587764458}), (0, 17, {'my_weight': -1.8913914615413283}), (0, 27, {'my_weight': -2.593604385364791}), (1, 1, {'my_weight': -0.3269634938395962}), (1, 2, {'my_weight': 3.4920562338129724}), (1, 5, {'my_weight': 4.310887877104206}), (1, 15, {'my_weight': 0.4967038254356908}), (1, 17, {'my_weight': 4.410908103092465}), (1, 27, {'my_weight': 1.900508570109873}), (1, 16, {'my_weight': -5.5831046197012855}), (1, 18, {'my_weight': 1.9511178193431877}), (1, 28, {'my_weight': 4.364636408579826}), (2, 2, {'my_weight': -0.2701620867747317}), (2, 3, {'my_weight': 4.169719537546574}), (2, 6, {'my_weight': -0.334598059551278}), (2, 16, {'my_weight': -2.89125635749759}), (2, 18, {'my_weight': -2.9394808347638373}), (2, 28, {'my_weight': 1.8841214881853943}), (2, 17, {'my_weight': -1.090403712454153}), (2, 19, {'my_weight': 2.085909540317346}), (2, 29, {'my_weight': 1.6521967332985596}), (3, 3, {'my_weight': -1.3857269468490805}), (3, 4, {'my_weight': 3.467843906896431}), (3, 7, {'my_weight': 1.3829201641760127}), (3, 17, {'my_weight': 6.301079300216028}), (3, 19, {'my_weight': -4.106403795112888}), (3, 29, {'my_weight': -5.741830150425138}), (3, 18, {'my_weight': -1.408201512779737}), (3, 20, {'my_weight': -2.5231848129878833}), (4, 4, {'my_weight': -3.2114452544082686}), (4, 5, {'my_weight': 0.602809246348201}), (4, 8, {'my_weight': 1.120394062245133}), (4, 18, {'my_weight': 2.871181150916248}), (4, 20, {'my_weight': 6.22413025378874}), (4, 19, {'my_weight': -2.7962373293687723}), (4, 21, {'my_weight': 0.035086364705746886}), (5, 5, {'my_weight': -0.7620648654275473}), (5, 6, {'my_weight': 0.24477103353686186}), (5, 9, {'my_weight': -0.3323866804877831}), (5, 19, {'my_weight': 1.439899502530342}), (5, 21, {'my_weight': 1.6229084183693334}), (5, 20, {'my_weight': -0.4010278103968084}), (5, 22, {'my_weight': -2.0118033653760805}), (6, 6, {'my_weight': -0.18008875697994342}), (6, 7, {'my_weight': -1.4109345435797696}), (6, 10, {'my_weight': -1.4737146014822502}), (6, 20, {'my_weight': 1.7386318179992024}), (6, 22, {'my_weight': 1.763152974306688}), (6, 21, {'my_weight': 1.1564780809247195}), (6, 23, {'my_weight': 2.1757807939942007}), (7, 7, {'my_weight': -4.109591306828882}), (7, 8, {'my_weight': -1.316924757039319}), (7, 11, {'my_weight': 0.27213000208894306}), (7, 21, {'my_weight': 2.8809219083896918}), (7, 23, {'my_weight': -0.6294336430156857}), (7, 22, {'my_weight': 5.093538258668716}), (7, 24, {'my_weight': 4.2193739507722965}), (8, 8, {'my_weight': -5.805194784565136}), (8, 9, {'my_weight': -5.389133451099294}), (8, 12, {'my_weight': -1.3126988271686133}), (8, 22, {'my_weight': 1.8864666970433175}), (8, 24, {'my_weight': 1.1687200800188318}), (8, 23, {'my_weight': 2.08579399084509}), (8, 25, {'my_weight': -0.8449372723760589}), (9, 9, {'my_weight': -6.9515261844656955}), (9, 10, {'my_weight': 2.182804434744901}), (9, 13, {'my_weight': 3.153390048762577}), (9, 23, {'my_weight': -4.220267943962885}), (9, 25, {'my_weight': 5.206301505308167}), (9, 24, {'my_weight': -1.2772887424047517}), (9, 26, {'my_weight': 4.128759670468028}), (10, 10, {'my_weight': -5.608918417074133}), (10, 11, {'my_weight': -1.4781972716612983}), (10, 14, {'my_weight': 0.5290001214002217}), (10, 24, {'my_weight': -1.26131905524565}), (10, 26, {'my_weight': -0.06784729363520087}), (10, 15, {'my_weight': -0.22569757495179782}), (10, 25, {'my_weight': 4.182084002162871}), (10, 27, {'my_weight': 2.670595999092118}), (11, 11, {'my_weight': -7.225470549729787}), (11, 12, {'my_weight': 2.3706973634840356}), (11, 15, {'my_weight': -0.6083856053206074}), (11, 25, {'my_weight': -6.914328526697285}), (11, 27, {'my_weight': -0.3429122144534084}), (11, 16, {'my_weight': 0.8299282482137915}), (11, 26, {'my_weight': 1.3100793110791091}), (11, 28, {'my_weight': 1.6268632468659532}), (12, 12, {'my_weight': -0.13210525903514703}), (12, 13, {'my_weight': 2.8694065191512257}), (12, 16, {'my_weight': -2.6889166176523}), (12, 26, {'my_weight': -3.505597871789142}), (12, 28, {'my_weight': -0.20596051401330331}), (12, 17, {'my_weight': 1.4895684822521285}), (12, 27, {'my_weight': -4.640729608129235}), (12, 29, {'my_weight': -6.592320123790266}), (13, 13, {'my_weight': -3.0444421420841543}), (13, 14, {'my_weight': -2.3161593922200585}), (13, 17, {'my_weight': 0.017662716944124468}), (13, 27, {'my_weight': 5.443859490947134}), (13, 29, {'my_weight': 3.061071420038591}), (13, 15, {'my_weight': 0.8413461957439806}), (13, 18, {'my_weight': 2.5961544886052916}), (13, 28, {'my_weight': 0.30848520749914765}), (14, 14, {'my_weight': -2.8375807525264793}), (14, 15, {'my_weight': 1.4687925914715292}), (14, 18, {'my_weight': -0.6339492769075543}), (14, 28, {'my_weight': -1.240648805854599}), (14, 16, {'my_weight': -1.9735210434358739}), (14, 19, {'my_weight': -1.7475205599132373}), (14, 29, {'my_weight': 1.4538721891046373}), (15, 15, {'my_weight': -0.49711209388004474}), (15, 18, {'my_weight': -3.2976921559697563}), (15, 28, {'my_weight': -0.0424077498086014}), (15, 16, {'my_weight': -0.4556554402082836}), (15, 17, {'my_weight': 0.6058364145512951}), (15, 20, {'my_weight': -0.20028977299704087}), (16, 16, {'my_weight': -2.6843303545500823}), (16, 19, {'my_weight': -4.67559962472941}), (16, 29, {'my_weight': -0.5125708183858373}), (16, 17, {'my_weight': -3.80218422577037}), (16, 18, {'my_weight': -0.37146067575430397}), (16, 21, {'my_weight': -0.2729709795093107}), (17, 17, {'my_weight': -2.3412674105194444}), (17, 20, {'my_weight': 2.626663567881043}), (17, 18, {'my_weight': -2.7719200462135745}), (17, 19, {'my_weight': -0.3096903118710015}), (17, 22, {'my_weight': 0.46599969789059004}), (18, 18, {'my_weight': -1.2516090775138682}), (18, 21, {'my_weight': -1.0920871771171319}), (18, 19, {'my_weight': 0.5698193025998213}), (18, 20, {'my_weight': 1.6691745201156614}), (18, 23, {'my_weight': 1.3805035923255708}), (19, 19, {'my_weight': -2.6478714099072955}), (19, 22, {'my_weight': -1.8154871093174842}), (19, 20, {'my_weight': 0.6962186248374396}), (19, 21, {'my_weight': 0.8981145347637491}), (19, 24, {'my_weight': -0.19957557898541073}), (20, 20, {'my_weight': -0.5711770860778239}), (20, 23, {'my_weight': 1.0570953375346486}), (20, 21, {'my_weight': -1.9348655226722864}), (20, 22, {'my_weight': 1.2118205942715283}), (20, 25, {'my_weight': 2.5627651167835372}), (21, 21, {'my_weight': -2.8388786130333434}), (21, 24, {'my_weight': 2.6331384215457407}), (21, 22, {'my_weight': 2.7833095911912498}), (21, 23, {'my_weight': -0.47294738538690895}), (21, 26, {'my_weight': -0.7236835279582017}), (22, 22, {'my_weight': -2.5444798122723937}), (22, 25, {'my_weight': 4.1783369392201}), (22, 23, {'my_weight': 0.7640505058405782}), (22, 24, {'my_weight': -2.9622259225629426}), (22, 27, {'my_weight': 1.6000161466639906}), (23, 23, {'my_weight': -0.31603444797041574}), (23, 26, {'my_weight': -2.242392679310452}), (23, 24, {'my_weight': 2.419029412509442}), (23, 25, {'my_weight': -0.6485648566595601}), (23, 28, {'my_weight': 4.019928461121245}), (24, 24, {'my_weight': -1.0570491399281623}), (24, 27, {'my_weight': 1.3646431133197177}), (24, 25, {'my_weight': 0.6758865639369812}), (24, 26, {'my_weight': -1.5118472085721253}), (24, 29, {'my_weight': 1.9980070434740482}), (25, 25, {'my_weight': -3.7921531684188983}), (25, 28, {'my_weight': 0.789727866207521}), (25, 26, {'my_weight': -3.2175462512045563}), (25, 27, {'my_weight': 3.8784366372555787}), (26, 26, {'my_weight': -0.6414628951713206}), (26, 29, {'my_weight': -2.074175292195851}), (26, 27, {'my_weight': 0.5409021843102414}), (26, 28, {'my_weight': -0.15331312704993705}), (27, 27, {'my_weight': -1.995689900107273}), (27, 28, {'my_weight': -1.678094345428447}), (27, 29, {'my_weight': 3.711883112794214}), (28, 28, {'my_weight': -2.876201783218588}), (28, 29, {'my_weight': -0.19818270516965844}), (29, 29, {'my_weight': -1.2900470478603419})]
        #testRay = numpy.array(testList)

        ########### Create Graph from numpy Array
        # G = nx.from_numpy_array(brainWeight)
        #
        # print(nx.info(G))
        # print(G.edges(data=True))
        # z = nx.get_edge_attributes(G, "weight")
        # print("Attributes: ")

        ##### Create Graph by adding Nodes and Edges seperate
        G = nx.Graph(brain="CTRNN")

        for i in range(len(brainSate)):
            G.add_node(i)

        for zeile in range(len(brainSate)):
            for spalte in range(len(brainSate)):
                value = brainWeight[zeile, spalte]
                G.add_edges_from([(zeile, spalte, {'myweight': value})])
                #G.add_weighted_edges_from([(zeile, spalte, value)])

        # print(nx.info(G))
        # z = nx.get_edge_attributes(G, "myweight")
        # print("Attributes: ")
        # print(z)
        # print(G.edges(data=True))

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

        ######### Create initial Pose and write to Json file; If pose already in file, just read it
        fpath = "position.json"
        if os.path.isfile(fpath) and os.path.getsize(fpath) == 0:
            numpyData = nx.spring_layout(G, k=3, weight="myweight", iterations=100)
            with open("position.json", "w") as write_file:
                json.dump(numpyData, write_file, cls=NumpyArrayEncoder)
                write_file.close()

        with open("position.json", "r") as read_file:
            decodedArray = json.load(read_file)
            read_file.close()
        #print(decodedArray)
        initialPosDict = {}
        for i in range(30):
            value = decodedArray[str(i)]
            initialPosDict[i] = value

        # pos ist ein Dictionary mit Werten in einem Koordintensystem mit Ursprung in der Mitte und Achsenlängen von -1 bis 1
        pos = nx.spring_layout(G, pos=initialPosDict, weight="myweight")
        # pos2 ist ein Dictionary mit Werten in einem Koordintensystem mit Ursprung in der Mitte und Achsenlängen von 0 bis h/2-100; h = höhe des Bildschirms
        pos2 = nx.spring_layout(G, pos=initialPosDict, weight="myweight", scale=h/2-100)

        pos2 = Position.getGraphPositions(self, w, h)

        ########### Plot Graph
        #pos = nx.nx_pydot.graphviz_layout(G)
        # nx.draw(G, pos)
        # print(pos)
        # plt.show()
        # time.sleep(0.5)


        # draw split lines
        # pygame.draw.line(screen, (255, 255, 255), ((w/6), 0), ((w/6), h), 1)
        # pygame.draw.line(screen, (255, 255, 255), (((5*w)/6), 0), (((5*w)/6), h), 1)
        # pygame.draw.line(screen, (255, 255, 255), (0, 40), (w, 40), 1)


        ########### Methods to convert from classic coordinate system to pygame window
        def changeCoordinateSystemX(x_coordinate):
            x = int((x_coordinate * ((2 * w) / 6)) + ((w/2)-15))
            return x

        def changeCoordinateSystemY(y_coordinate):
            y = int(60 + (y_coordinate * ((-1*(h-100))/2)) + ((h-100)/2))
            return y

        ########### n-1 Linien pro Neuron ; Input zu Neuron
        numberNeurons = G.number_of_nodes()
        numberObValues = len(in_values)
        textSurface = myfont.render("Input Neurons: " + str(numberObValues), False, numColor)
        screen.blit(textSurface, ((((1*w)/12)-55), 10))

        # obPositionsDict = {}
        # for i in range(numberObValues):
        #     if i % 2:  # ungerade
        #         #x_pos = 75
        #         x_pos = (w/18)
        #         #y_pos = 50 + i * (400 / (numberObValues - 1))
        #         y_pos = 50 + i * ((h - 90) / (numberObValues))
        #         obPositionsDict[i] = [x_pos, y_pos]
        #     else:  # gerade
        #         #x_pos = 125
        #         x_pos = ((2*w) / 18)
        #         #y_pos = 50 + i * (400 / (numberObValues-1))
        #         y_pos = 50 + ((2*(h - 90)) / (numberObValues)) + i * ((h - 90) / (numberObValues))
        #         obPositionsDict[i] = [x_pos, y_pos]

        obPositionsDict = {}
        for i in range(numberObValues):

            if (h - 80) / (numberObValues * 30) > 1:
                x_pos = ((w) / 18)
                y_pos = (80 + (h / 2)) - ((numberObValues * 50)) + (i * 65)
                obPositionsDict[i] = [x_pos, y_pos]
            else:
                if i % 2:  # ungerade
                    x_pos = ((w) / 18)
                    y_pos = (30 + (h / 2)) - ((numberObValues * 30) / 2) + (i * 30)
                    obPositionsDict[i] = [x_pos, y_pos]
                else:  # gerade
                    x_pos = ((2 * w) / 18)
                    y_pos = (30 + (h / 2)) - ((numberObValues * 30) / 2) + (i * 30)
                    obPositionsDict[i] = [x_pos, y_pos]

        for zeile in range(numberNeurons):
            for spalte in range(numberObValues):
                kantenGewicht = self.brain.V[zeile, spalte]
                if kantenGewicht > 0.0:
                    position0 = obPositionsDict[spalte]
                    pos_x0 = int(position0[0])
                    pos_y0 = int(position0[1])
                    # position1 = pos[zeile]
                    # #pos_x1 = int(position1[0] * 500) + 500 + 200
                    # #pos_x1 = int(position1[0] * ((2*w)/6)) + ((5 * w) / 12)
                    # pos_x1 = changeCoordinateSystemX(position1[0])
                    # #pos_y1 = int(position1[1] * (-250)) + 250
                    # #pos_y1 = 50 + int(position1[1] * ((-1*(h-50))/2)) + ((h-50)/2)
                    # pos_y1 = changeCoordinateSystemY(position1[1])
                    position1 = pos2[zeile]
                    pos_x1 = int(position1[0] + (w / 2))
                    pos_y1 = int(position1[1] + (h / 2))
                    pygame.draw.line(screen, (169,169,169), (pos_x0, pos_y0), (pos_x1, pos_y1), int(kantenGewicht)-2)


        ########### n-1 Linien pro Neuron ; Neuron zu Neuron
        textSurface = myfont.render("Neurons in Hidden Layer: " + str(numberNeurons), False, numColor)
        screen.blit(textSurface, ((((6*w)/12)-90), 10))

        #TODO: checken ob die iteration (erst Zeile dann Spalte) stimmt bei self.brain.W[zeile, spalte]
        for zeile in range(numberNeurons):
            for spalte in range(numberNeurons):
                kantenGewicht = self.brain.W[zeile, spalte]
                if kantenGewicht > 0.0:
                    # position0 = pos[zeile]
                    # #pos_x0 = int(position0[0] * 500) + 500 + 200
                    # #pos_y0 = int(position0[1] * (-250)) + 250
                    # #pos_x0 = int(position0[0] * ((2 * w) / 6)) + ((5 * w) / 12)
                    # #pos_y0 = 50 + int(position0[1] * ((-1 * (h - 50)) / 2)) + ((h - 50) / 2)
                    # pos_x0 = changeCoordinateSystemX(position0[0])
                    # pos_y0 = changeCoordinateSystemY(position0[1])
                    # position1 = pos[spalte]
                    # #pos_x1 = int(position1[0] * ((2 * w) / 6)) + ((5 * w) / 12)
                    # #pos_y1 = 50 + int(position1[1] * ((-1 * (h - 50)) / 2)) + ((h - 50) / 2)
                    # pos_x1 = changeCoordinateSystemX(position1[0])
                    # pos_y1 = changeCoordinateSystemY(position1[1])

                    position0 = pos2[zeile]
                    pos_x0 = int(position0[0] + (w / 2))
                    pos_y0 = int(position0[1] + (h / 2))

                    position1 = pos2[spalte]
                    pos_x1 = int(position1[0] + (w / 2))
                    pos_y1 = int(position1[1] + (h / 2))

                    pygame.draw.line(screen, (169,169,169), (pos_x0, pos_y0), (pos_x1, pos_y1), int(kantenGewicht)-2)

        ########### n-1 Linien pro Neuron ; Neuron zu Output
        numberOutputValues = len(out_values)
        textSurface = myfont.render("Output Neurons: " + str(numberOutputValues), False, numColor)
        screen.blit(textSurface, ((((11*w)/12)-55), 10))

        # outputPositionsDict = {}
        # for i in range(numberOutputValues):
        #     if i % 2:  # ungerade
        #         #x_pos = 1250 + 75
        #         x_pos = ((16*w) / 18)
        #         #y_pos = 50 + i * (400 / (numberOutputValues-1))
        #         abstand = ((h - 120) / (numberOutputValues-1))
        #         y_pos = 80 + i * abstand
        #         outputPositionsDict[i] = [x_pos, y_pos]
        #     else:  # gerade
        #         #x_pos = 1250 + 125
        #         x_pos = ((17 * w) / 18)
        #         #y_pos = 50 + i * (400 / (numberOutputValues-1))
        #         abstand = ((h - 120) / (numberOutputValues-1))
        #         y_pos = 80 + i * abstand
        #         outputPositionsDict[i] = [x_pos, y_pos]

        outputPositionsDict = {}
        for i in range(numberOutputValues):

            if (h-80)/(numberOutputValues*40) > 1:
                x_pos = ((11 * w) / 12)
                y_pos = (80 + (h/2)) - ((numberOutputValues*50)) + (i*65)
                outputPositionsDict[i] = [x_pos, y_pos]
            else:
                if i % 2:  # ungerade
                    x_pos = ((16 * w) / 18)
                    y_pos = (80 + (h/2)) - ((numberOutputValues*50)/2) + (i*50)
                    outputPositionsDict[i] = [x_pos, y_pos]
                else:  # gerade
                    x_pos = ((17 * w) / 18)
                    y_pos = (80 + (h/2)) - ((numberOutputValues*50)/2) + (i*50)
                    outputPositionsDict[i] = [x_pos, y_pos]

        for zeile in range(numberNeurons):
            for spalte in range(numberOutputValues):
                kantenGewicht = self.brain.T[zeile, spalte]
                if kantenGewicht > 0.0:
                    position0 = outputPositionsDict[spalte]
                    pos_x0 = int(position0[0])
                    pos_y0 = int(position0[1])
                    # position1 = pos[zeile]
                    # # pos_x1 = int(position1[0] * 500) + 500 + 200
                    # #pos_x1 = int(position1[0] * ((2 * w) / 6)) + ((5 * w) / 12)
                    # pos_x1 = changeCoordinateSystemX(position1[0])
                    # # pos_y1 = int(position1[1] * (-250)) + 250
                    # #pos_y1 = 50 + int(position1[1] * ((-1 * (h - 50)) / 2)) + ((h - 50) / 2)
                    # pos_y1 = changeCoordinateSystemY(position1[1])
                    position1 = pos2[zeile]
                    pos_x1 = int(position1[0] + (w / 2))
                    pos_y1 = int(position1[1] + (h / 2))
                    pygame.draw.line(screen, (169,169,169), (pos_x0, pos_y0), (pos_x1, pos_y1), int(kantenGewicht)-2)

        ########### Interpolate between two colors and value between 0 and 1
        def interpolateColor(color1, color2, val):
            R = int((color2[0] - color1[0]) * abs(val) + color1[0])
            G = int((color2[1] - color1[1]) * abs(val) + color1[1])
            B = int((color2[2] - color1[2]) * abs(val) + color1[2])
            color = (R, G, B)
            return color


        ########### Draw neurons
        #TODO: Neuronen nicht übereinander zeichnen
        positionsDict = {}
        for neuron in range(G.number_of_nodes()):
            position = pos2[neuron]
            # #pos_x = int(position[0] * 500) + 500 + 200
            # #pos_y = int(position[1] * (-250)) + 250
            # #pos_x = int((position[0] * ((2 * w) / 6)) + ((5 * w) / 12))
            # #pos_y = int(50 + (position[1] * ((-1 * (h - 50)) / 2)) + ((h - 50) / 2))
            # pos_x = changeCoordinateSystemX(position[0])
            # pos_y = changeCoordinateSystemY(position[1])
            pos_x = int(position[0] + (w/2))
            pos_y = int(position[1] + (h/2))
            positionsDict[neuron] = [pos_x, pos_y]

            val = self.brain.y[neuron]
            #color = abs((self.brain.y[neuron]) * 255) * 10
            textSurface = myfont.render(('%.5s' % str(val)), False, numColor)

            interpolierteFarbe = interpolateColor((189,198,222), (0,24,75), val*10)

            # werte gehen von (-0,1 - 0,1)(189,198,222)
            farbe0 = 0,24,75

            kantenGewicht = self.brain.W[neuron, neuron]
            interpolierteFarbeRand = interpolateColor((255, 198, 165), (255, 0, 0), kantenGewicht/10)

            # Draw Circle and Text
            #TODO: Interpolieren der Randfarbe mit Verbindung zu sich selbst aus W Matrix
            pygame.draw.circle(screen, interpolierteFarbeRand, (pos_x, pos_y), 30)
            pygame.draw.circle(screen, interpolierteFarbe, (pos_x, pos_y), 27)
            #pygame.draw.circle(screen, blue, (pos_x, pos_y), 20)
            screen.blit(textSurface, ((pos_x - 16), (pos_y - 7)))

        ########### draw ob-values (in)
        for ob in range(numberObValues):
            position = obPositionsDict[ob]
            pos_x = int(position[0])
            pos_y = int(position[1])

            val = in_values[ob]
            color = abs((self.brain.y[ob]) * 255) * 10
            #color = 255
            # TODO: wird hier gerade noch gekappt
            if val > 1.4 or val < -1.4:
                val = 1
            interpolierteFarbe = interpolateColor((165, 222, 148), (16, 57, 16), int(val))
            textSurface = myfont.render(('%.5s' % val), False, numColor)

            # Draw Circle and Text
            pygame.draw.circle(screen, interpolierteFarbe, (pos_x, pos_y), 30)
            #pygame.draw.circle(screen, (0, color, 0), (pos_x, pos_y), 30)
            screen.blit(textSurface, ((pos_x - 16), (pos_y - 7)))


        ########### draw action-values (out)
        for output in range(numberOutputValues):
            #ob_values = in_values

            position = outputPositionsDict[output]
            pos_x = int(position[0])
            pos_y = int(position[1])

            val = out_values[output]
            color = abs((self.brain.y[output]) * 255) * 10
            #color = 255
            interpolierteFarbe = interpolateColor((255, 231, 198), (173, 74, 24), val)
            textSurface = myfont.render(('%.5s' % val), False, numColor)

            # Draw Circle and Text
            pygame.draw.circle(screen, interpolierteFarbe, (pos_x, pos_y), 30)
            #pygame.draw.circle(screen, (color, 0, 0), (pos_x, pos_y), 30)
            screen.blit(textSurface, ((pos_x - 16), (pos_y - 7)))


        ######### Method to get the Number of the Neuron back if Mousclick on it
        def getNeuronsOnClick(self, mousePose):
            for i in range(numberObValues):
                neuronPose = obPositionsDict[i]
                distance = math.sqrt(((mousePose[0] - neuronPose[0]) ** 2) + ((mousePose[1] - neuronPose[1]) ** 2))
                if distance < 30:
                    mouseFont = pygame.font.SysFont("Helvetica", 32)
                    mouseLabel = mouseFont.render("Ob-Neuron:" + str(i), 1, brightGrey)
                    screen.blit(mouseLabel, (mousePose[0], mousePose[1]-32))

            for i in range(numberNeurons):
                neuronPose = positionsDict[i]
                distance = math.sqrt(((mousePose[0] - neuronPose[0]) ** 2) + ((mousePose[1] - neuronPose[1]) ** 2))
                if distance < 30:
                    mouseFont = pygame.font.SysFont("Helvetica", 32)
                    mouseLabel = mouseFont.render("Neuron:" + str(i), 1, brightGrey)
                    screen.blit(mouseLabel, (mousePose[0], mousePose[1]-32))

            for i in range(numberOutputValues):
                neuronPose = outputPositionsDict[i]
                distance = math.sqrt(((mousePose[0] - neuronPose[0]) ** 2) + ((mousePose[1] - neuronPose[1]) ** 2))
                if distance < 30:
                    mouseFont = pygame.font.SysFont("Helvetica", 32)
                    mouseLabel = mouseFont.render("Output-Neuron:" + str(i), 1, brightGrey)
                    screen.blit(mouseLabel, (mousePose[0]-200, mousePose[1]-32))



        ######### Events: Close when x-Button, Show Number of Neuron when click on it
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            # if event.type == MOUSEBUTTONDOWN:
            #     getNeuronsOnClick(self, pygame.mouse.get_pos())
            if event.type == MOUSEMOTION:
                getNeuronsOnClick(self, pygame.mouse.get_pos())


        # Update the Screen
        #pygame.display.flip()

        # Updates the content of the window
        pygame.display.flip()
