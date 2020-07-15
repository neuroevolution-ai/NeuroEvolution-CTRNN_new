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

        ##### Create Graph by adding Nodes and Edges seperate
        G = nx.Graph(brain="CTRNN")

        for i in range(len(brainState)):
            G.add_node(i)

        for zeile in range(len(brainState)):
            for spalte in range(len(brainState)):
                value = brainWeight[zeile, spalte]
                G.add_edges_from([(zeile, spalte, {'myweight': value})])


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
        for i in range(len(self.brain.W)):
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

