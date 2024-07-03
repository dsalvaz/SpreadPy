from ..SpreadingModel import SpreadingBaseModel
from ..AnalyticalUtils import AnalyticalUtils
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["BaseSpreading", "MultiplexSpreading"]

class BaseSpreading(SpreadingBaseModel):
    """
    Class that defines a Base Spreading Model
    """

    def __init__(self, graph, retention, decay, suppress, weighted=False, weight='weight'):
        """
        Model Constructor
        """

        super(self.__class__, self).__init__(graph, retention, decay, suppress)
        self.weighted = weighted
        self.weight = weight
        if weighted:
            self.weights = self.add_edge_strength(weight)


    def ___spread_uniform(self, u, neighbors, status):
        """
        Spread activation uniformly to neighboring nodes.

        :param u: Node from which activation spreads
        :param neighbors: List of neighboring nodes
        :param status: Dictionary of node statuses
        """

        total_neighbors = len(neighbors)
        decrease_activation_u = 0

        for v in neighbors:
            excess_activation = status[u] / total_neighbors
            # Apply retention factor
            excess_activation *= (1 - self.retention)
            status[v] += excess_activation # increase neighbor
            decrease_activation_u += excess_activation

        status[u] -= decrease_activation_u # decrease current


    def ___spread_weighted(self, u, neighbors, status):
        """
        Spread activation to neighboring nodes according to edge weights.

        :param u: Node from which activation spreads
        :param neighbors: List of neighboring nodes
        :param status: Dictionary of node statuses
        """

        decrease_activation_u = 0

        for v in neighbors:
            weight_uv = self.weights[self.weight][(u,v)]
            excess_activation = status[u] * weight_uv
            # Apply retention factor
            excess_activation *= (1 - self.retention)
            status[v] += excess_activation # increase neighbor
            decrease_activation_u += excess_activation

        status[u] -= decrease_activation_u # decrease current


    def iteration(self):
        """
        Execute a single model iteration

        :return: Iteration_id, Incremental node status (dictionary node->status)
        """

        actual_status = {
            node: self.status[node] for node in self.graph.nodes
        }

        if self.actual_iteration == 0:
            self.actual_iteration += 1
            return {
                "iteration": 0,
                "status": actual_status.copy(),
            }

        for u in self.graph.nodes:

            if actual_status[u] > 0:

                neighbors = self.node_neighbors[u]
                spread_function = self.___spread_weighted if self.weighted else self.___spread_uniform
                spread_function(u, neighbors, actual_status)

        # Apply decay and suppression
        actual_status = {node: (1 - self.decay) * activation if activation >= self.suppress else 0
                         for node, activation in actual_status.items()}

        activation = self.status_activation(actual_status)
        self.status = actual_status  # update status
        self.actual_iteration += 1

        return {
            "iteration": self.actual_iteration - 1,
            "status": activation.copy(),
        }




class MultiplexSpreading(SpreadingBaseModel):
    """
    Class that defines a Multiplex Spreading Model
    """

    def __init__(self, graph, retention, decay, suppress, intra_layer_prob, inter_layer_prob, # r
                 weighted=False, weight='weight'):
        """
        Model Constructor
        """

        #super(self.__class__, self).__init__(graph, retention, decay, suppress)
        super(MultiplexSpreading, self).__init__(graph, retention, decay, suppress)

        self.layers = set([layer for _, _, data in self.graph.edges(data=True) for layer in data['layer']])
   

        utils = AnalyticalUtils(self.graph)

        self.layer_gs = {layer: utils.graph_for_layers(layer_names=[layer]) for layer in self.layers}
        self.node_neighbors = {
            layer: {node: list(self.layer_gs[layer].neighbors(node)) if node in self.layer_gs[layer].nodes else [] for node in self.graph.nodes}
            for layer in self.layers
        }

        self.weighted = weighted
        self.weight = weight
        if weighted:
            self.weights = {layer: self.add_edge_strength(self.weight, layer) for layer in self.layers}

        self.status = {layer: {node: 0 for node in self.graph.nodes} for layer in self.layers}

        #self.r = r
        #total_r = 1 + r
        #self.intra_layer_prob = 1 / total_r
        #self.inter_layer_prob = r / total_r

        self.intra_layer_prob = intra_layer_prob
        self.inter_layer_prob = inter_layer_prob



    def ___spread_intra_multiplex_uniform(self, u, neighbors, status, layer):
        """
        Spread activation uniformly to neighboring nodes.

        :param u: Node from which activation spreads
        :param neighbors: List of neighboring nodes
        :param status: Dictionary of node statuses
        """

        total_neighbors = len(neighbors)
        decrease_activation_u = 0

        for v in neighbors:
            excess_activation = status[layer][u] / total_neighbors
            # Apply retention factor
            excess_activation *= (1 - self.retention) * self.intra_layer_prob
            status[layer][v] += excess_activation # increase neighbor
            decrease_activation_u += excess_activation

        status[layer][u] -= decrease_activation_u # decrease current


    def ___spread_inter_multiplex_uniform(self, u, status, current_layer):
        """
        Spread activation uniformly to neighboring nodes.

        :param u: Node from which activation spreads
        :param neighbors: List of neighboring nodes
        :param status: Dictionary of node statuses
        :param current_layer: Layer from which starts the replication on other layers
        """

        decrease_activation_u = 0
        for layer in self.layers: # replica
            if layer != current_layer:

                excess_activation = status[current_layer][u]
                # Apply retention factor
                excess_activation *= (1 - self.retention) * self.inter_layer_prob
                status[layer][u] += excess_activation  # increase replica
                decrease_activation_u += excess_activation


                # Spread to neighbors in the replicated layer
                neighbors = self.node_neighbors[layer][u]
                total_neighbors = len(neighbors)
                decrease_activation_u_neighbors = 0

                for v in neighbors:
                    neighbor_activation = excess_activation / total_neighbors
                    neighbor_activation *= (1 - self.retention)
                    status[layer][v] += neighbor_activation  # increase neighbor in replicated layer
                    decrease_activation_u_neighbors += neighbor_activation

                status[layer][u] -= decrease_activation_u_neighbors  # decrease activation due to neighbor spread
            

        status[current_layer][u] -= decrease_activation_u  # decrease current


    def ___spread_intra_multiplex_weighted(self, u, neighbors, status, layer):
        """
        Spread activation to neighboring nodes according to edge weights.

        :param u: Node from which activation spreads
        :param neighbors: List of neighboring nodes
        :param status: Dictionary of node statuses
        """

        #total_weight = sum(self.layer_gs[layer][u][v][self.weight] for v in neighbors)
        decrease_activation_u = 0

        for v in neighbors:
            weight_uv = self.weights[layer][self.weight][(u,v)]
            excess_activation = status[layer][u] * weight_uv
            # Apply retention factor
            excess_activation *= (1 - self.retention) * self.intra_layer_prob
            status[layer][v] += excess_activation  # increase neighbor
            decrease_activation_u += excess_activation

        status[layer][u] -= decrease_activation_u  # decrease current


    def iteration(self):
        """
        Execute a single model iteration

        :return: Iteration_id, Incremental node status (dictionary node->status)
        """

        actual_status = {layer:
                              {node: self.status[layer][node] for node in self.graph.nodes}
                          for layer in self.layers
                          }

        if self.actual_iteration == 0:
            self.actual_iteration += 1
            return {
                "iteration": 0,
                "status": actual_status.copy(),
            }

        for u in self.graph.nodes:

            for layer in self.layers:

                if actual_status[layer][u] > 0:

                    #############  activate within the same layer #########
                    neighbors = self.node_neighbors[layer][u]
                    spread_function_intra = self.___spread_intra_multiplex_weighted if self.weighted \
                                      else self.___spread_intra_multiplex_uniform

                    spread_function_intra(u, neighbors, actual_status, layer)


                    #############  activate replica across layer #########
                    #initial_layer = layer 
                    # no need to distinguish between weighted and non-weighted
                    spread_function_inter = self.___spread_inter_multiplex_uniform # format coherent with above
                    #spread_function_inter(u, actual_status, initial_layer)
                    spread_function_inter(u, actual_status, layer)

        for layer in self.layers:
            actual_status[layer] = {
                node: (1 - self.decay) * activation if activation >= self.suppress else 0
                for node, activation in actual_status[layer].items()
            }

        activation = self.status_activation(actual_status)
        self.status = actual_status
        self.actual_iteration += 1

        return {
            "iteration": self.actual_iteration - 1,
            "status": activation.copy(),
            }

    def add_edge_strength(self, name_attr, layer, eps=0.001):
        """
        Calculate edge strength for specified node attributes.

        :param name_attrs: List of node attribute names as strings
        :param eps: parameter value to guarantee not zero division
        :return: a networkx graph object with weights on edges
        """

        weights = defaultdict(lambda: defaultdict(float))
        # for na in name_attrs:
        beta_nodes = {}
        for u in self.graph.nodes():
            a = self.graph.nodes[u][name_attr]

            neighbors = self.node_neighbors[layer][u]
            beta_i = 0
            for v in neighbors:
                b = self.graph.nodes[v][name_attr]
                beta_i += 1 / (abs(a - b) + eps)
            beta_nodes[u] = beta_i ** -1

        for u, v in self.layer_gs[layer].edges(): # nodes are the same but edges are not
            a = self.graph.nodes[u][name_attr]
            b = self.graph.nodes[v][name_attr]

            weight_u = beta_nodes[u] / (abs(a - b) + eps)
            weight_v = beta_nodes[v] / (abs(a - b) + eps)

            weights[name_attr][(u, v)] = weight_u
            weights[name_attr][(v, u)] = weight_v

        return weights


    def status_activation(self, actual_status):
        """
         Compute the point-to-point variations for each status w.r.t. the previous system configuration

        :param actual_status: the actual simulation status
        :return: node that have changed their statuses (dictionary status->nodes),

        """
        activation = defaultdict(lambda: defaultdict(int))
        for layer, dct in self.status.items():
            for n, v in dct.items():
                if v != actual_status[layer][n]:
                    activation[layer][n] = actual_status[layer][n]

        return activation


    def node_activation_series(self, system_status, layer):
        """
        Create a time series of activation for each node

        :param system_status: the actual simulation system
        :return: a dict whose keys are nodes and values a list of activation units
        """
        activation_values = defaultdict(list)
        for result in system_status:
            for node, activation in result["status"][layer].items():
                activation_values[node].append(activation)

        return activation_values

    def plot_top_k_rank(self, dicts_sem, dicts_phon, timelist, figname=''):

        num_plots = len(dicts_sem)
        fig, axes = plt.subplots(1, num_plots, figsize=(num_plots * 3, 6))
        fig.suptitle('Top-K Rank Activated Words', fontsize=16)

        for i, d in enumerate(dicts_sem):
            ax = axes[i] if num_plots > 1 else axes

            shared_keys = sorted(list(set(d.keys()).intersection(dicts_phon[i].keys())))
            unique_keys_d = sorted(list(set(d.keys()).difference(dicts_phon[i].keys())))
            unique_keys_d_1 = sorted(list(set(dicts_phon[i].keys()).difference(d.keys())))

            y1_keys = shared_keys + unique_keys_d
            y2_keys = shared_keys + unique_keys_d_1

            y1_keys = list(reversed(y1_keys))
            y2_keys = list(reversed(y2_keys))

            y1_values = []
            for k in y1_keys:
                y1_values.append(d[k])

            y2_values = []
            for k in y2_keys:
                y2_values.append(dicts_phon[i][k])

            ax.plot(y1_values, range(len(d)), marker='o', alpha=0.5, linewidth=2, color='blue', label='semantics')
            ax.set_yticks(range(len(d)))
            ax.set_yticklabels(y1_keys, fontsize=12, color='blue')
            # Define custom x-axis ticks
            x_ticks = np.linspace(min(y1_values), max(y1_values), num=4)  # Define ticks here
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([round(tick) for tick in x_ticks], fontsize=12)  # Convert to strings for display
            ax.grid(alpha=0.2)
            ax.set_title(f"t={timelist[i]}")

            ax.axhline(y=len(y1_keys)-len(shared_keys) - 0.5, color='black', linestyle='--', linewidth=1.5)

            ax2 = ax.twinx()
            ax2.plot(y2_values, range(len(d)), marker='o', alpha=0.5, linewidth=2, color='red', label='phonology')
            ax2.set_yticks(range(len(d)))
            ax2.set_yticklabels(y2_keys, fontsize=12, color='red')

            if i==0:
                lines_1, labels_1 = ax.get_legend_handles_labels()
                lines_2, labels_2 = ax2.get_legend_handles_labels()
                lines = lines_1 + lines_2
                labels = labels_1 + labels_2
                ax.legend(lines, labels)

        fig.text(0.5, -0.01, 'Activation', ha='center', fontsize=13)  # Common x-label
        plt.tight_layout()

        if figname != '':
            plt.savefig(figname, bbox_inches='tight')








