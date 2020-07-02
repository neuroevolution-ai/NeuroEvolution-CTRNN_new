import networkx as nx
import matplotlib.pyplot as plt
import time

class Matplotlib():
    def matplotlib(self):
        #G = nx.barabasi_albert_graph(100,m=1)

        g = nx.Graph()

        global brainSate
        brainSate = self.brain.y
        brainWeight = self.brain.W
        print(brainWeight)


        arr = brainSate
        for i in range(0, 29):
            x = arr[i]
            g.add_node(i, value=x)

        pos2 = nx.spring_layout(g)
        #print(pos2)

        # G = nx.Graph()
        # G.add_node(1, value=0.5)
        # G.add_node(2, value=0.8)
        # G.add_edge(1, 2, weight=3)
        #
        # G.add_node(3, value=0.1)
        # G.add_node(4, value=0.2)
        # G.add_edge(1, 2, weight=1)


        fig = plt.figure()
        #nx.draw(g, with_labels=True, node_color="skyblue", node_size=300, edge_color="white")
        nx.draw(g, pos2)
        #nx.draw_networkx_nodes(g, pos2)
        fig.set_facecolor("#eaeaea")
        plt.show()

        time.sleep(10)
        return pos2
