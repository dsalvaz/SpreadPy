import networkx as nx
from glob import glob


class IOUtils(object):
    """
    Class that defines utilities for input-output processing
    """

    def __init__(self):
        """
        Model Constructor

        """

    def read_example_mental_lexicon(self, folder):
        """
        A function for a toy example in the ReadME of the package.
        """

        def find_layer_names(layer_files):
            name_layers = []
            for file in layer_files:
                name_layer = file.split("\\")[-1]
                name_layers.append(name_layer)
            return [name_layer.split('.')[0] for name_layer in name_layers]

        def build_graph(layer_files, name_layers):
            g = nx.Graph()
            for i, file in enumerate(layer_files):
                with open(file) as f:
                    for l in f:
                        l = l.rsplit()
                        g.add_edge(l[0], l[1])

                        if 'layer' not in g.edges[l[0], l[1]].keys():
                            g.edges[l[0], l[1]]['layer'] = [name_layers[i]]
                        else:
                            if name_layers[i] not in g.edges[l[0], l[1]]['layer']:
                                g.edges[l[0], l[1]]['layer'].append(name_layers[i])
            return g

        layer_files = glob(folder + '*.txt')
        layer_names = find_layer_names(layer_files)

        g_multiplex = build_graph(layer_files, layer_names)
        g_multiplex.remove_edges_from(nx.selfloop_edges(g_multiplex))

        return g_multiplex, layer_names

