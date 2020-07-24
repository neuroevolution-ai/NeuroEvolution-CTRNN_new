import networkx as nx
import numpy as np

class Positions():
    def getGraphPositions(self, w, h):
        brainState = self.brain.y
        brainWeight = self.brain.W.todense()

        ##### Create Graph by adding Nodes and Edges seperate
        G = nx.Graph(brain="CTRNN")

        for i in range(len(brainState)):
            G.add_node(i)

        for index, value in np.ndenumerate(brainWeight):
            G.add_edges_from([(index[0], index[1], {'myweight': value})])

        pos = nx.spring_layout(G, k=1, weight="myweight", iterations=50, scale=h / 2 - 100)

        # Adapt positions from spring-layout Method to pygame windows
        graphPositionsDict = {}
        for each in pos:
            position = pos[each]
            pos_x = int(position[0] + (w / 2))
            if pos_x > (w/2):
                pos_x = pos_x + 50
            if pos_x < (w/2):
                pos_x = pos_x - 50
            pos_y = int(position[1] + (h / 2)) + 60
            graphPositionsDict[each] = [pos_x, pos_y]

        return graphPositionsDict


    ##### Calcuale Input or Output Positions based on number of Neurons and radius of Neurons
    def getInputOutputPositions(self, numberNeurons, is_input):
        PositionsDict = {}
        if is_input:
            x = ((1 * self.w) / 12)
            x2 = ((1 * self.w) / 18)
            x3 = ((2 * self.w) / 18)
        elif not is_input:
            x = ((11 * self.w) / 12)
            x2 = ((16 * self.w) / 18)
            x3 = ((17 * self.w) / 18)

        # Place Neurons in one row if there is enough place, else take two rows
        for i in range(numberNeurons):
            if ((self.h - 100) / (numberNeurons * (self.neuronRadius*2))) > 1:
                x_pos = x
                y_pos = ((self.neuronRadius*2) + (self.h / 2)) - ((numberNeurons * (self.neuronRadius))) + (i * (self.neuronRadius*2))
                PositionsDict[i] = [x_pos, y_pos]
            else:
                if i % 2:  # ungerade
                    x_pos = x2
                    y_pos = ((self.neuronRadius*2) + (self.h / 2)) - ((numberNeurons * self.neuronRadius) / 2) + (i * self.neuronRadius)
                    PositionsDict[i] = [x_pos, y_pos]
                else:  # gerade
                    x_pos = x3
                    y_pos = ((self.neuronRadius*2) + (self.h / 2)) - ((numberNeurons * self.neuronRadius) / 2) + (i * self.neuronRadius)
                    PositionsDict[i] = [x_pos, y_pos]
        return PositionsDict

