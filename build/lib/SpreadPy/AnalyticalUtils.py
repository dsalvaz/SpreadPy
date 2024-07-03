import networkx as nx
import numpy as np
import pandas as pd
import json
from nltk.corpus import wordnet as wn

from conformity import attribute_conformity

from scipy.sparse.linalg import eigs


class AnalyticalUtils(object):
    """
    Class that defines utilities for:
    1) feature-rich properties of a graph, i.e., node attributes (length, polysemy, etc);
    2) extract useful patterns, subgraphs, and graph representations, e.g., laplacians, LVC, etc;
    3) analyzing superdiffusion modes of a graph;
    """

    def __init__(self, graph):
        """
        Model Constructor

        :param graph: A networkx graph object
        """

        self.graph = graph


    def ___add_additional_attributes(self, attribute_dict):
        """
        Add attributes to nodes based on the provided dictionary
        """
        for attr_name, attr_values in attribute_dict.items():
            for node, value in attr_values.items():
                self.graph.nodes[node][attr_name] = value


    def ___add_word_length(self, binned=False):
        """
        Set word length as a node attribute
        """

        length_values = []
        for n in self.graph.nodes():
            length_values.append(len(n))

        if binned==True:
            polysemy_series = pd.Series(length_values)
            labels = [0, 1, 2, 3]
            bins = pd.qcut(polysemy_series, q=4, labels=labels)

        for i, n in enumerate(self.graph.nodes()):
            self.graph.nodes[n]['length'] = length_values[i]
            if binned==True:
                self.graph.nodes[n]['length_bin'] = bins[i]


    def ___add_word_polysemy(self, binned=False):
        """
        Set word polysemy based on wordnet as a node attribute
        """

        polysemy_values = []
        for n in self.graph.nodes():
            syn = wn.synsets(n)
            polysemy_values.append(len(syn))

        if binned==True:
            polysemy_series = pd.Series(polysemy_values)
            labels = [0, 1, 2, 3]
            bins = pd.qcut(polysemy_series, q=4, labels=labels)

        for i, n in enumerate(self.graph.nodes()):
            self.graph.nodes[n]['polysemy'] = polysemy_values[i]
            if binned==True:
                self.graph.nodes[n]['polysemy_bin'] = bins[i]


    def word_features_pipeline(self, binned=False, conformity=[], additional_attributes=None):
        """
        Perform a pipeline to add word features as attributes on nodes
        """

        self.___add_word_length(binned=binned)
        self.___add_word_polysemy(binned=binned)

        if additional_attributes:
            self.___add_additional_attributes(additional_attributes)

        if conformity != []:
            for file in conformity:
                format_conformity = {}
                attr_name = ''
                with open(file) as f:
                    data = json.load(f)
                    for alpha, dct in data.items():
                        for attr, dct_conf in dct.items():
                            if attr_name == '':
                                attr_name = attr
                            for node, conf in dct_conf.items():
                                format_conformity[node] = conf

                for n in self.graph.nodes():
                    self.graph.nodes[n][attr_name+'_conformity'] = format_conformity[n]


    def node_conformity(self, alphas=[1,2], attribute=['length_bin'], profile_size=1, savefile=''):

        node_to_conformity = attribute_conformity(self.graph, alphas, attribute, profile_size=profile_size)

        if savefile != '':
            with open(savefile, 'w') as outfile:
                json.dump(node_to_conformity, outfile)

        return node_to_conformity


    def graph_for_layers(self, layer_names, multigraph=False):
        """
        Create a subgraph for specific layers

        :param layer_names: list of layer names as strings
        :return: a networkx graph object
        """

        layer_edges = [(u, v) for u, v, layers in self.graph.edges(data='layer') if
                       any(layer in layers for layer in layer_names)]

        if multigraph == True:
            layer_graph = nx.MultiGraph(layer_edges)
        else:
            layer_graph = nx.Graph(layer_edges)


        return layer_graph


    def get_largest_connected_component(self, layer_names):
        """
        Create a subgraph of the largest connected component of a graph

        :param layer_names: list of layer names as strings
        :return: a networkx graph object
        """

        g_layer = self.graph_for_layers(layer_names=[layer_names])
        largest_component = max(nx.connected_components(g_layer), key=len)

        return g_layer.subgraph(largest_component)


    def ___recursive_lvc(self, layers, lccs):
        """
        Recursively finds the largest viable component (LVC) among specified layers in a multiplex graph.

        :param layers: List of layer names to conduct the search on.
        :param lccs: List of largest connected components for each layer.
        :return: the LVC as a list of nodes
        """

        common_nodes = set(lccs[0]).intersection(*lccs[1:])
        subgraph = self.graph.subgraph(common_nodes)
        largest_component = max(nx.connected_components(subgraph), key=len)
        subg_lcc = subgraph.subgraph(largest_component)
        self.graph = subg_lcc

        new_lccs = [self.get_largest_connected_component(layer) for layer in layers]
        new_common_nodes = set(new_lccs[0]).intersection(*new_lccs[1:])

        if new_common_nodes == common_nodes:
            return list(new_common_nodes)
        else:
            return self.___recursive_lvc(layers, new_lccs)


    def lvc(self, layer_names):
        """
        Finds the largest viable component (LVC), namely the largest common connected set of nodes
        among specified layers in a multiplex graph.

        :param layer_names: List of layer names to conduct the search on.
        :return: the LVC as a list of nodes
        """

        lccs = [self.get_largest_connected_component(layer) for layer in layer_names]

        return self.___recursive_lvc(layer_names, lccs)


    def adjacency_for_layers(self, layer_names):
        """
        Retrieve the adjacency matrix for the specific layers of a networkx graph object

        :param layer_names: layer names as a list of strings
        :return: a numpy array representing the adjacency matrix
        """

        layer_graph = self.graph_for_layers(layer_names)
        nodes_order = sorted(layer_graph.nodes())
        adjacency_matrix = nx.linalg.graphmatrix.adjacency_matrix(layer_graph, nodelist=nodes_order).toarray()

        return adjacency_matrix


    def laplacian_for_layers(self, layer_names):
        """
        Retrieve the laplacian matrix for the specific layers of a networkx graph object

        :param layer_names: layer name as a list of string
        :return: a numpy array representing the laplacian matrix
        """

        layer_graph = self.graph_for_layers(layer_names)
        nodes_order = sorted(layer_graph.nodes())
        laplacian = nx.linalg.laplacianmatrix.laplacian_matrix(layer_graph, nodelist=nodes_order).toarray()

        return laplacian


    def average_laplacian(self, laplacian_1, laplacian_2):
        """
        Compute the average of two Laplacian matrices element-wise.

        :param laplacian_1: Laplacian matrix 1 (numpy array)
        :param laplacian_2: Laplacian matrix 2 (numpy array)
        :return: Average Laplacian matrix (numpy array)
        """

        # Check if matrices have the same shape
        if laplacian_1.shape != laplacian_2.shape:
            raise ValueError("Laplacian matrices must have the same shape.")

        # Compute the average element-wise
        L_l1_l2 = (laplacian_1 + laplacian_2) / 2

        return L_l1_l2


    def supralaplacian(self, laplacian_1, laplacian_2, dx=1):
        """
        Construct the supralaplacian matrix from two Laplacian matrices.

        :param laplacian_1: Laplacian matrix 1 (numpy array)
        :param laplacian_2: Laplacian matrix 2 (numpy array)
        :param dx: inter-layer diffusion parameter
        :return: Supralaplacian matrix (numpy array)
        """

        n = laplacian_1.shape[0]  # Assuming laplacian_1 and laplacian_2 have the same shape
        # Create the blocks
        block1 = np.block([
            [laplacian_1, np.zeros((n, n))],
            [np.zeros((n, n)), laplacian_2]
        ])

        block2 = np.block([
            [np.eye(n), -np.eye(n)],
            [-np.eye(n), np.eye(n)]
        ])

        supralaplacian_matrix = block1 + (dx * block2)

        return supralaplacian_matrix


    def compute_second_smallest_eigenvalue(self, matrix):
        """
        Compute the second smallest eigenvalue of a matrix using the 'SM' method.

        :param matrix: Input matrix (numpy array)
        :return: Second smallest eigenvalue (float)
        """

        second_smallest = eigs(matrix.astype(np.float32), k=2, which='SM', return_eigenvectors=False).real[0]

        return second_smallest


