from graph_tool.all import *

class GraphTool():
    def graphTool(self):

        # neuer Graph der Klasse Graph
        g = Graph()

        v1 = g.add_vertex()
        v2 = g.add_vertex()

        e = g.add_edge(v1, v2)

        # Exportiert ne PDF in /home/benny/NeuroEvolution-CTRNN_new
        graph_draw(g, vertex_text=g.vertex_index, output="two-nodes.pdf")

