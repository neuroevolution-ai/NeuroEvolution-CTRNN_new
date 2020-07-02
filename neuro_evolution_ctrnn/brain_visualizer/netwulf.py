import networkx as nx
import netwulf as nw
import matplotlib.pyplot as plt
import time

class Netwulf():
    def netwulf(self):
        #G = nx.barabasi_albert_graph(100,m=1)

        g = nx.Graph()

        global brainSate
        brainSate = self.brain.y

        arr = brainSate
        for i in range(0, 29):
            x = arr[i]
            g.add_node(i, value=x)

        nw.visualize(g)
