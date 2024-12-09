import abc
import six
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import matplotlib.pyplot as plt

@six.add_metaclass(abc.ABCMeta)
class SpreadingBaseModel(object):
    """
    Partial Abstract Class that defines Spreading Models
    """

    def __init__(self, graph, retention=0.5, decay=0.1, suppress=0):
        """
        Model Constructor

        :param graph: A networkx graph object
        :param retention: The ratio of activation units retained by a node, between 0 and 1
        :param decay: The decay ratio for propagating activation units, between 0 and 1.
        :param suppress: The node suppression ratio for propagating activation units, between 0 and 1.
        """

        self.actual_iteration = 0
        self.graph = graph
        self.node_neighbors = {node: list(self.graph.neighbors(node)) for node in self.graph.nodes}
        self.retention = retention
        self.decay = decay
        self.suppress = suppress
        self.status = {n: 0 for n in self.graph.nodes} # all nodes are inactive


    @abc.abstractmethod
    def iteration(self):
        """
        Execute a single model iteration

        :return: Iteration_id, Incremental node status (dictionary node->status)
        """
        pass


    def iteration_bunch(self, bunch_size):
        """
        Execute a bunch of model iterations

        :param bunch_size: the number of iterations to execute
        :return: a list containing for each iteration a dictionary
                {"iteration": iteration_id, "status": dictionary_node_to_status}
        """

        system_status = []
        for it in tqdm(range(0, bunch_size)):
            its = self.iteration()
            system_status.append(its)

        return system_status


    def add_edge_strength(self, name_attr, eps=0.01, similarity=False):
        """
        Calculate directional edge strengths based on node attributes.

        :param name_attr: Name of the node attribute as a string
        :param eps: Small parameter to prevent division by zero
        :return: A dictionary of weights for directed edges
        """

        weights = defaultdict(lambda: defaultdict(float)) 
    
        if similarity == True:

            beta_nodes = {}
            # Step 1: Compute beta_i for each node
            for u in self.graph.nodes():
                a = self.graph.nodes[u][name_attr]
                beta_i_sum = 0
                for v in self.node_neighbors[u]:
                    b = self.graph.nodes[v][name_attr]
                    beta_i_sum += 1 / (abs(a - b) + eps)
                beta_nodes[u] = 1 / beta_i_sum  # Normalization factor for node u

            # Step 2: Compute directional weights for edges
            for u, v in self.graph.edges():
                a = self.graph.nodes[u][name_attr]
                b = self.graph.nodes[v][name_attr]

                # Directional weights calculation
                weight_uv = beta_nodes[u] / (abs(a - b) + eps)  # Energy flowing from u to v
                weight_vu = beta_nodes[v] / (abs(b - a) + eps)  # Energy flowing from v to u

                # Assign weights to directed edges
                weights[name_attr][(u, v)] = weight_uv
                weights[name_attr][(v, u)] = weight_vu
 
        else: # CREATIVITY FOR DISSIMILARITY

            beta_nodes = {}
            for u in self.graph.nodes():
                a = self.graph.nodes[u][name_attr]
                beta_i_sum = 0
                for v in self.node_neighbors[u]:
                    b = self.graph.nodes[v][name_attr]
                    # Invertire la logica per enfatizzare differenze
                    beta_i_sum += abs(a - b) + eps
                beta_nodes[u] = 1 / beta_i_sum  # Normalizzazione per il nodo u

            # Step 2: Compute directional weights for edges
            for u, v in self.graph.edges():
                a = self.graph.nodes[u][name_attr]
                b = self.graph.nodes[v][name_attr]

                # Modifica: flussi favoriscono caratteristiche dissimili
                weight_uv = beta_nodes[u] * (abs(a - b) + eps)  # Energia da u a v
                weight_vu = beta_nodes[v] * (abs(b - a) + eps)  # Energia da v a u

                # Assegna i pesi agli archi diretti
                weights[name_attr][(u, v)] = weight_uv
                weights[name_attr][(v, u)] = weight_vu

        return weights



    def status_activation(self, actual_status):
        """
        Compute the point-to-point variations for each status w.r.t. the previous system configuration

        :param actual_status: the actual simulation status
        :return: node that have changed their statuses (dictionary status->nodes),

        """
        activation = {}
        for n, v in self.status.items():
            if v != actual_status[n]:
                activation[n] = actual_status[n]

        return activation


    def rank_status_activation(self, actual_status):
        """
        Rank an actual simulation status

        :param actual_status: the actual simulation status
        :return: the actual simulation status sorted by value
        """

        return dict(
            sorted(actual_status.items(), key=lambda x: x[1], reverse=True)
        )


    def count_status_activation(self, actual_status, th=0.01):

        count = 0
        for _, v in actual_status.items():
            if v > th:
                count += 1

        return count


    def node_activation_series(self, system_status):
        """
        Create a time series of activation for each node

        #:param system_status: the actual simulation system
        #:return: a dict whose keys are nodes and values a list of activation units
        """

        activation_values = defaultdict(list)
        for result in system_status:
            for node, activation in result["status"].items():
                activation_values[node].append(activation)

        return activation_values


    def plot_activated_series(self, activation_values, time, nodes, title='', figname=''):
        """
        Plot the time series of activation units for each activated node

        :param activation_values: a dictionary of nodes and lists of activation values
        :time: number of iterations
        :title: title of mpl plot
        :figname: name of the saved file
        """

        fig, ax = plt.subplots()

        time_points = list(range(time))
        for node in nodes:
            if len(time_points) == len(activation_values[node]):
                ax.plot(time_points, activation_values[node], marker='o', alpha=0.5, label=node)

        ax.set_xlabel('Time', fontsize=18)
        ax.set_ylabel('Activation', fontsize=18)
        ax.tick_params(axis='both', labelsize=18)
        ax.legend(fontsize=15)
        plt.grid(alpha=0.2)
        if title != '':
            plt.title(title)

        plt.tight_layout()

        if figname != '':
            plt.savefig(figname, bbox_inches='tight')


    def plot_top_k_rank(self, *dicts, timelist, figname=''):

        num_plots = len(dicts)
        fig, axes = plt.subplots(1, num_plots, figsize=(num_plots * 3, 6))
        fig.suptitle('Top-K Rank Activated Words', fontsize=16)

        for i, d in enumerate(dicts):
            ax = axes[i] if num_plots > 1 else axes
            values = list(reversed(list(d.values())))
            keys = list(reversed(list(d.keys())))
            ax.plot(values, range(len(d)), marker='o', alpha=0.5, linewidth=2)
            ax.set_yticks(range(len(d)))
            ax.set_yticklabels(keys, fontsize=12)
            # Define custom x-axis ticks
            x_ticks = np.linspace(min(values), max(values), num=5)  # Define ticks here
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([f'{tick:.1f}' for tick in x_ticks], fontsize=12)  # Convert to strings for display
            ax.grid(alpha=0.2)
            ax.set_title(f"t={timelist[i]}")

        fig.text(0.5, -0.01, 'Activation', ha='center', fontsize=13)  # Common x-label
        plt.tight_layout()

        if figname != '':
            plt.savefig(figname, bbox_inches='tight')

