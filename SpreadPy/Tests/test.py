from SpreadPy.Models.models import *
from SpreadPy.AnalyticalUtils import *

import unittest
import networkx as nx
import random

class TestIC(unittest.TestCase):

    def test_graph_multigraph(self):

        g = nx.barabasi_albert_graph(100, 3)
        for u, v in g.edges():
            if random.uniform(0, 1) < 0.5:
                g.edges[(u, v)]['layer'] = ['l1']
            else:
                g.edges[(u, v)]['layer'] = ['l1', 'l2']

        mg = nx.MultiGraph()
        for u, v, d in g.edges(data=True):
            for k, ll in d.items():
                for l in ll:
                    mg.add_edge(u, v, layer=l)

        utils_g = AnalyticalUtils(g)
        utils_mg = AnalyticalUtils(mg)

        l1_g = utils_g.graph_for_layers(['l1'])
        l1_mg = utils_mg.graph_for_layers(['l1'], multigraph=True)

        l2_g = utils_g.graph_for_layers(['l2'])
        l2_mg = utils_mg.graph_for_layers(['l2'], multigraph=True)

        ge_l1 = l1_g.number_of_edges()
        gme_l1 = l1_mg.number_of_edges()

        ge_l2 = l2_g.number_of_edges()
        gme_l2 = l2_mg.number_of_edges()

        self.assertEqual(ge_l1, gme_l1)
        self.assertEqual(ge_l2, gme_l2)

    def test_adjacency(self):

        g = nx.barabasi_albert_graph(100, 3)
        for u, v in g.edges():
            if random.uniform(0, 1) < 0.5:
                g.edges[(u, v)]['layer'] = ['l1']
            else:
                g.edges[(u, v)]['layer'] = ['l1', 'l2']

        utils = AnalyticalUtils(g)
        adjacency_l1 = utils.adjacency_for_layers(['l1'])
        adjacency_l2 = utils.adjacency_for_layers(['l2'])

        self.assertNotEqual(adjacency_l1, adjacency_l2)

    def test_laplacian(self):

        g = nx.barabasi_albert_graph(100, 3)
        for u, v in g.edges():
            if random.uniform(0, 1) < 0.5:
                g.edges[(u, v)]['layer'] = ['l1']
            else:
                g.edges[(u, v)]['layer'] = ['l1', 'l2']

        utils = AnalyticalUtils(g)
        laplacian_l1 = utils.laplacian_for_layers(['l1'])
        laplacian_l2 = utils.laplacian_for_layers(['l2'])

        self.assertNotEqual(laplacian_l1, laplacian_l2)


    def test_BaseSpreading(self):

        g = nx.barabasi_albert_graph(100, 3)

        model = BaseSpreading(g, retention=0.3, decay=0, suppress=0)
        time = 5

        initial_active_nodes = [0]
        initial_status = {node: 1 if node in initial_active_nodes else 0 for node in g.nodes}
        model.status = initial_status

        results = model.iteration_bunch(time)
        final_status = results[-1]['status']
        ranked_results = model.rank_status_activation(final_status)

        self.assertIsInstance(results, list)
        self.assertIsInstance(results[0]['iteration'], int)
        self.assertIsInstance(results[0]['status'], dict)
        self.assertIsInstance(ranked_results, dict)

        activation_series = model.node_activation_series(results)
        self.assertIsInstance(activation_series, dict)

    def test_WeightedSpreading(self):

        g = nx.barabasi_albert_graph(100, 3)
        for u, v in g.edges():
            g[u][v]['weight'] = random.uniform(0, 1)

        model = BaseSpreading(g, retention=0.3, decay=0, suppress=0, weighted=True)
        time = 5

        initial_active_nodes = [0]
        initial_status = {node: 1 if node in initial_active_nodes else 0 for node in g.nodes}
        model.status = initial_status

        results = model.iteration_bunch(time)
        final_status = results[-1]['status']
        ranked_results = model.rank_status_activation(final_status)

        self.assertIsInstance(results, list)
        self.assertIsInstance(results[0]['iteration'], int)
        self.assertIsInstance(results[0]['status'], dict)
        self.assertIsInstance(ranked_results, dict)

        activation_series = model.node_activation_series(results)
        self.assertIsInstance(activation_series, dict)

    def test_viz(self):
        g = nx.barabasi_albert_graph(100, 3)

        model = BaseSpreading(g, retention=0.5, decay=0.1, suppress=0)
        time = 10

        initial_active_nodes = [0]
        initial_status = {node: 1 if node in initial_active_nodes else 0 for node in g.nodes}
        model.status = initial_status

        results = model.iteration_bunch(time)
        activation_series = model.node_activation_series(results)

        model.plot_activated_series(activation_series, time, [0], figname='test.png')

    def test_viz_comparison(self):

        g = nx.barabasi_albert_graph(100, 3)
        for u, v in g.edges():
            g[u][v]['random_weight'] = random.uniform(0.1, 1)

        model_1 = BaseSpreading(g, retention=0.3, decay=0.1, suppress=0)
        model_2 = BaseSpreading(g, retention=0.3, decay=0.1, suppress=0,
                                weighted=True, weight='random_weight')
        time = 10

        initial_active_nodes = [0]
        initial_status = {node: 1 if node in initial_active_nodes else 0 for node in g.nodes}
        model_1.status = initial_status
        model_2.status = initial_status

        results_1 = model_1.iteration_bunch(time)
        results_2 = model_2.iteration_bunch(time)
        #activation_series_1 = model_1.node_activation_series(results_1)
        #activation_series_2 = model_2.node_activation_series(results_2)

        #model_1.plot_activated_series(activation_series_1, time, [0], figname='test.png')
        #model_2.plot_activated_series(activation_series_2, time, [0], figname='test_1.png')

    def test_multiplex_spreading(self):

        g = nx.barabasi_albert_graph(100, 3)
        for u, v in g.edges():
            if random.uniform(0, 1) < 0.5:
                g.edges[(u, v)]['layer'] = ['l1']
            else:
                g.edges[(u, v)]['layer'] = ['l1', 'l2']

        model = MultiplexSpreading(g, retention=0.5, decay=0.1, suppress=0,
                                   intra_layer_r=1, inter_layer_r=1)
        time = 10

        initial_active_nodes = [0]
        initial_layers = ['l1']
        initial_status = {layer: {node: 1 if node in initial_active_nodes and layer in initial_layers \
                          else 0 for node in g.nodes} for layer in model.layers}
        model.status = initial_status

        results = model.iteration_bunch(time)

        activation_series_l1 = model.node_activation_series(results, 'l1')
        activation_series_l2 = model.node_activation_series(results, 'l2')
        model.plot_activated_series(activation_series_l1, time, [0], figname='test.png')
        model.plot_activated_series(activation_series_l2, time, [0], figname='test_1.png')

        #print(results)

        #print(activation_series_l1[0])
        #print(activation_series_l2[0])




