from pyvis.network import Network
import networkx as nx
import time


class Pyvis():
    def pyvis(self):
        from pyvis.network import Network

        global brainSate
        brainSate = self.brain.y

        g = Network()
        arr = brainSate
        for i in range(0, 29):
            x = arr[i]
            g.add_node(i, value=x)

        g.show_buttons()
        g.show("basic.html")
        print(arr)

        time.sleep(5)
