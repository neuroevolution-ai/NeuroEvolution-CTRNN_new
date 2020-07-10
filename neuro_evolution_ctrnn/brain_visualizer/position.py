import networkx as nx
import json
from json import JSONEncoder
import os
import numpy


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

class Positions():
    def getGraphPositions(self, w, h):
        brainState = self.brain.y
        brainWeight = self.brain.W

        ######### imort position
        # g = nx.Graph()
        # g.add_nodes_from(brainStatedict)

        # [(0, 0, {'weight': -1.9011296238884632}), (0, 1, {'weight': 3.220957626055494}), (0, 4, {'weight': -0.6631949403544992}), (0, 14, {'weight': 4.642733924984142}),
        # print(numpy.info(brainWeight))

        # testList = [(0, 0, {'my_weight': -1.9011296238884632}), (0, 1, {'my_weight': 3.220957626055494}), (0, 4, {'my_weight': -0.6631949403544992}), (0, 14, {'my_weight': 4.642733924984142}), (0, 16, {'my_weight': -5.6578376702341515}), (0, 26, {'my_weight': 1.293560810386954}), (0, 29, {'my_weight': -0.45573146626343664}), (0, 15, {'my_weight': 0.3027749587764458}), (0, 17, {'my_weight': -1.8913914615413283}), (0, 27, {'my_weight': -2.593604385364791}), (1, 1, {'my_weight': -0.3269634938395962}), (1, 2, {'my_weight': 3.4920562338129724}), (1, 5, {'my_weight': 4.310887877104206}), (1, 15, {'my_weight': 0.4967038254356908}), (1, 17, {'my_weight': 4.410908103092465}), (1, 27, {'my_weight': 1.900508570109873}), (1, 16, {'my_weight': -5.5831046197012855}), (1, 18, {'my_weight': 1.9511178193431877}), (1, 28, {'my_weight': 4.364636408579826}), (2, 2, {'my_weight': -0.2701620867747317}), (2, 3, {'my_weight': 4.169719537546574}), (2, 6, {'my_weight': -0.334598059551278}), (2, 16, {'my_weight': -2.89125635749759}), (2, 18, {'my_weight': -2.9394808347638373}), (2, 28, {'my_weight': 1.8841214881853943}), (2, 17, {'my_weight': -1.090403712454153}), (2, 19, {'my_weight': 2.085909540317346}), (2, 29, {'my_weight': 1.6521967332985596}), (3, 3, {'my_weight': -1.3857269468490805}), (3, 4, {'my_weight': 3.467843906896431}), (3, 7, {'my_weight': 1.3829201641760127}), (3, 17, {'my_weight': 6.301079300216028}), (3, 19, {'my_weight': -4.106403795112888}), (3, 29, {'my_weight': -5.741830150425138}), (3, 18, {'my_weight': -1.408201512779737}), (3, 20, {'my_weight': -2.5231848129878833}), (4, 4, {'my_weight': -3.2114452544082686}), (4, 5, {'my_weight': 0.602809246348201}), (4, 8, {'my_weight': 1.120394062245133}), (4, 18, {'my_weight': 2.871181150916248}), (4, 20, {'my_weight': 6.22413025378874}), (4, 19, {'my_weight': -2.7962373293687723}), (4, 21, {'my_weight': 0.035086364705746886}), (5, 5, {'my_weight': -0.7620648654275473}), (5, 6, {'my_weight': 0.24477103353686186}), (5, 9, {'my_weight': -0.3323866804877831}), (5, 19, {'my_weight': 1.439899502530342}), (5, 21, {'my_weight': 1.6229084183693334}), (5, 20, {'my_weight': -0.4010278103968084}), (5, 22, {'my_weight': -2.0118033653760805}), (6, 6, {'my_weight': -0.18008875697994342}), (6, 7, {'my_weight': -1.4109345435797696}), (6, 10, {'my_weight': -1.4737146014822502}), (6, 20, {'my_weight': 1.7386318179992024}), (6, 22, {'my_weight': 1.763152974306688}), (6, 21, {'my_weight': 1.1564780809247195}), (6, 23, {'my_weight': 2.1757807939942007}), (7, 7, {'my_weight': -4.109591306828882}), (7, 8, {'my_weight': -1.316924757039319}), (7, 11, {'my_weight': 0.27213000208894306}), (7, 21, {'my_weight': 2.8809219083896918}), (7, 23, {'my_weight': -0.6294336430156857}), (7, 22, {'my_weight': 5.093538258668716}), (7, 24, {'my_weight': 4.2193739507722965}), (8, 8, {'my_weight': -5.805194784565136}), (8, 9, {'my_weight': -5.389133451099294}), (8, 12, {'my_weight': -1.3126988271686133}), (8, 22, {'my_weight': 1.8864666970433175}), (8, 24, {'my_weight': 1.1687200800188318}), (8, 23, {'my_weight': 2.08579399084509}), (8, 25, {'my_weight': -0.8449372723760589}), (9, 9, {'my_weight': -6.9515261844656955}), (9, 10, {'my_weight': 2.182804434744901}), (9, 13, {'my_weight': 3.153390048762577}), (9, 23, {'my_weight': -4.220267943962885}), (9, 25, {'my_weight': 5.206301505308167}), (9, 24, {'my_weight': -1.2772887424047517}), (9, 26, {'my_weight': 4.128759670468028}), (10, 10, {'my_weight': -5.608918417074133}), (10, 11, {'my_weight': -1.4781972716612983}), (10, 14, {'my_weight': 0.5290001214002217}), (10, 24, {'my_weight': -1.26131905524565}), (10, 26, {'my_weight': -0.06784729363520087}), (10, 15, {'my_weight': -0.22569757495179782}), (10, 25, {'my_weight': 4.182084002162871}), (10, 27, {'my_weight': 2.670595999092118}), (11, 11, {'my_weight': -7.225470549729787}), (11, 12, {'my_weight': 2.3706973634840356}), (11, 15, {'my_weight': -0.6083856053206074}), (11, 25, {'my_weight': -6.914328526697285}), (11, 27, {'my_weight': -0.3429122144534084}), (11, 16, {'my_weight': 0.8299282482137915}), (11, 26, {'my_weight': 1.3100793110791091}), (11, 28, {'my_weight': 1.6268632468659532}), (12, 12, {'my_weight': -0.13210525903514703}), (12, 13, {'my_weight': 2.8694065191512257}), (12, 16, {'my_weight': -2.6889166176523}), (12, 26, {'my_weight': -3.505597871789142}), (12, 28, {'my_weight': -0.20596051401330331}), (12, 17, {'my_weight': 1.4895684822521285}), (12, 27, {'my_weight': -4.640729608129235}), (12, 29, {'my_weight': -6.592320123790266}), (13, 13, {'my_weight': -3.0444421420841543}), (13, 14, {'my_weight': -2.3161593922200585}), (13, 17, {'my_weight': 0.017662716944124468}), (13, 27, {'my_weight': 5.443859490947134}), (13, 29, {'my_weight': 3.061071420038591}), (13, 15, {'my_weight': 0.8413461957439806}), (13, 18, {'my_weight': 2.5961544886052916}), (13, 28, {'my_weight': 0.30848520749914765}), (14, 14, {'my_weight': -2.8375807525264793}), (14, 15, {'my_weight': 1.4687925914715292}), (14, 18, {'my_weight': -0.6339492769075543}), (14, 28, {'my_weight': -1.240648805854599}), (14, 16, {'my_weight': -1.9735210434358739}), (14, 19, {'my_weight': -1.7475205599132373}), (14, 29, {'my_weight': 1.4538721891046373}), (15, 15, {'my_weight': -0.49711209388004474}), (15, 18, {'my_weight': -3.2976921559697563}), (15, 28, {'my_weight': -0.0424077498086014}), (15, 16, {'my_weight': -0.4556554402082836}), (15, 17, {'my_weight': 0.6058364145512951}), (15, 20, {'my_weight': -0.20028977299704087}), (16, 16, {'my_weight': -2.6843303545500823}), (16, 19, {'my_weight': -4.67559962472941}), (16, 29, {'my_weight': -0.5125708183858373}), (16, 17, {'my_weight': -3.80218422577037}), (16, 18, {'my_weight': -0.37146067575430397}), (16, 21, {'my_weight': -0.2729709795093107}), (17, 17, {'my_weight': -2.3412674105194444}), (17, 20, {'my_weight': 2.626663567881043}), (17, 18, {'my_weight': -2.7719200462135745}), (17, 19, {'my_weight': -0.3096903118710015}), (17, 22, {'my_weight': 0.46599969789059004}), (18, 18, {'my_weight': -1.2516090775138682}), (18, 21, {'my_weight': -1.0920871771171319}), (18, 19, {'my_weight': 0.5698193025998213}), (18, 20, {'my_weight': 1.6691745201156614}), (18, 23, {'my_weight': 1.3805035923255708}), (19, 19, {'my_weight': -2.6478714099072955}), (19, 22, {'my_weight': -1.8154871093174842}), (19, 20, {'my_weight': 0.6962186248374396}), (19, 21, {'my_weight': 0.8981145347637491}), (19, 24, {'my_weight': -0.19957557898541073}), (20, 20, {'my_weight': -0.5711770860778239}), (20, 23, {'my_weight': 1.0570953375346486}), (20, 21, {'my_weight': -1.9348655226722864}), (20, 22, {'my_weight': 1.2118205942715283}), (20, 25, {'my_weight': 2.5627651167835372}), (21, 21, {'my_weight': -2.8388786130333434}), (21, 24, {'my_weight': 2.6331384215457407}), (21, 22, {'my_weight': 2.7833095911912498}), (21, 23, {'my_weight': -0.47294738538690895}), (21, 26, {'my_weight': -0.7236835279582017}), (22, 22, {'my_weight': -2.5444798122723937}), (22, 25, {'my_weight': 4.1783369392201}), (22, 23, {'my_weight': 0.7640505058405782}), (22, 24, {'my_weight': -2.9622259225629426}), (22, 27, {'my_weight': 1.6000161466639906}), (23, 23, {'my_weight': -0.31603444797041574}), (23, 26, {'my_weight': -2.242392679310452}), (23, 24, {'my_weight': 2.419029412509442}), (23, 25, {'my_weight': -0.6485648566595601}), (23, 28, {'my_weight': 4.019928461121245}), (24, 24, {'my_weight': -1.0570491399281623}), (24, 27, {'my_weight': 1.3646431133197177}), (24, 25, {'my_weight': 0.6758865639369812}), (24, 26, {'my_weight': -1.5118472085721253}), (24, 29, {'my_weight': 1.9980070434740482}), (25, 25, {'my_weight': -3.7921531684188983}), (25, 28, {'my_weight': 0.789727866207521}), (25, 26, {'my_weight': -3.2175462512045563}), (25, 27, {'my_weight': 3.8784366372555787}), (26, 26, {'my_weight': -0.6414628951713206}), (26, 29, {'my_weight': -2.074175292195851}), (26, 27, {'my_weight': 0.5409021843102414}), (26, 28, {'my_weight': -0.15331312704993705}), (27, 27, {'my_weight': -1.995689900107273}), (27, 28, {'my_weight': -1.678094345428447}), (27, 29, {'my_weight': 3.711883112794214}), (28, 28, {'my_weight': -2.876201783218588}), (28, 29, {'my_weight': -0.19818270516965844}), (29, 29, {'my_weight': -1.2900470478603419})]
        # testRay = numpy.array(testList)

        ##### Create Graph from numpy Array
        # G = nx.from_numpy_array(brainWeight)
        #
        # print(nx.info(G))
        # print(G.edges(data=True))
        # z = nx.get_edge_attributes(G, "weight")
        # print("Attributes: ")

        ##### Create Graph by adding Nodes and Edges seperate
        G = nx.Graph(brain="CTRNN")

        for i in range(len(brainState)):
            G.add_node(i)

        for zeile in range(len(brainState)):
            for spalte in range(len(brainState)):
                value = brainWeight[zeile, spalte]
                G.add_edges_from([(zeile, spalte, {'myweight': value})])
                # G.add_weighted_edges_from([(zeile, spalte, value)])

        # print(nx.info(G))
        # z = nx.get_edge_attributes(G, "myweight")
        # print("Attributes: ")
        # print(z)
        # print(G.edges(data=True))

        ######### Create initial Pose and write to Json file; If pose already in file, just read it
        fpath = "position.json"
        if os.path.isfile(fpath) and os.path.getsize(fpath) == 0:
            numpyData = nx.spring_layout(G, k=5, weight="myweight", iterations=100)
            with open("position.json", "w") as write_file:
                json.dump(numpyData, write_file, cls=NumpyArrayEncoder)
                write_file.close()

        with open("position.json", "r") as read_file:
            decodedArray = json.load(read_file)
            read_file.close()
        # print(decodedArray)
        initialPosDict = {}
        for i in range(30):
            value = decodedArray[str(i)]
            initialPosDict[i] = value

        # pos ist ein Dictionary mit Werten in einem Koordintensystem mit Ursprung in der Mitte und Achsenlängen von -1 bis 1
        #pos = nx.spring_layout(G, pos=initialPosDict, weight="myweight")
        # pos2 ist ein Dictionary mit Werten in einem Koordintensystem mit Ursprung in der Mitte und Achsenlängen von 0 bis h/2-100; h = höhe des Bildschirms
        pos2 = nx.spring_layout(G, k=2, pos=initialPosDict, weight="myweight", scale=h / 2 - 50)


        graphPositionsDict = {}
        for each in pos2:
            position0 = pos2[each]
            pos_x0 = int(position0[0] + (w / 2))
            if pos_x0 > (w/2):
                pos_x0 = pos_x0 + 50
            if pos_x0 < (w/2):
                pos_x0 = pos_x0 - 50
            pos_y0 = int(position0[1] + (h / 2)) + 60
            graphPositionsDict[each] = [pos_x0, pos_y0]

        return graphPositionsDict


    def clearJSON(self):
        open("position.json", "w").close()
        print("exiting")


    def getInputOutputPositions(self, numberNeurons, inputOrOutput):
        PositionsDict = {}
        if inputOrOutput == "input":
            x = ((1 * self.w) / 12)
            x2 = ((1 * self.w) / 18)
            x3 = ((2 * self.w) / 18)
        elif inputOrOutput == "output":
            x = ((11 * self.w) / 12)
            x2 = ((16 * self.w) / 18)
            x3 = ((17 * self.w) / 18)

        for i in range(numberNeurons):
            if ((self.h - 100) / (numberNeurons * (self.neuronRadius*2))) > 1:
                x_pos = x
                y_pos = (50 + (self.h / 2)) - ((numberNeurons * (self.neuronRadius))) + (i * (self.neuronRadius*2))
                PositionsDict[i] = [x_pos, y_pos]
            else:
                if i % 2:  # ungerade
                    x_pos = x2
                    y_pos = (50 + (self.h / 2)) - ((numberNeurons * self.neuronRadius) / 2) + (i * self.neuronRadius)
                    PositionsDict[i] = [x_pos, y_pos]
                else:  # gerade
                    x_pos = x3
                    y_pos = (50 + (self.h / 2)) - ((numberNeurons * self.neuronRadius) / 2) + (i * self.neuronRadius)
                    PositionsDict[i] = [x_pos, y_pos]
        return PositionsDict

