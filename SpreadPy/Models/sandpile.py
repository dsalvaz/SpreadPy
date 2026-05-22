from ..SpreadingModel import SpreadingBaseModel
import numpy as np

__all__ = ["SandpileSpreading"]


class SandpileSpreading(SpreadingBaseModel):
    """
    Abelian Sandpile Model on arbitrary graphs.

    Following the generalization by Dhar (1990), the sandpile is defined on
    any undirected finite graph G = (V, E). A designated vertex s ∈ V, called
    the sink, is not allowed to topple: grains sent to it are lost (dissipation).
    Any vertex can serve as the sink — no virtual nodes are needed.

    Each iteration drops 1 grain on a random non-sink node and topples
    until stable. This is the classic BTW sandpile on arbitrary graphs.

    Optionally, a dissipation probability f can be set: every grain that
    moves from one node to another disappears with probability f instead
    of arriving. When f > 0 and no sink is specified, dissipation is
    entirely stochastic.
    """

    def __init__(self, graph, sink=None, f=0):
        """
        :param graph:   networkx graph
        :param sink:    any node s ∈ V designated as sink (absorbs grains, never topples).
                        If None, no sink is used.
        :param f:       dissipation probability per grain transfer (default 0)
        """
        # the sandpile doesn't need retention, decay or suppress
        super(SandpileSpreading, self).__init__(graph, retention=0, decay=0, suppress=0)

        if sink is not None and sink not in self.graph.nodes:
            raise ValueError("Sink node must be in the graph.")

        self.sink = sink
        self.f = f
        # all nodes except the sink — these are the ones that can accumulate and topple
        self.non_sink_nodes = [n for n in self.graph.nodes if n != self.sink]
        # we cache the degree of each node so we don't recompute it every time
        self.node_degree = {n: self.graph.degree(n) for n in self.graph.nodes}

    def ___add_grain(self, actual_status, node=None):
        """Add one grain to a single node."""
        # if no node is specified, pick one at random among non-sink nodes
        if node is None:
            node = self.non_sink_nodes[np.random.randint(len(self.non_sink_nodes))]
        # give it +1 grain
        actual_status[node] += 1

    def ___topple(self, actual_status):
        """Keep toppling unstable nodes until everything is stable again."""
        toppled_nodes = set()
        total_topplings = 0

        # a node is unstable when its grains >= its degree (and degree > 0)
        unstable = [v for v in self.non_sink_nodes
                    if self.node_degree[v] > 0 and actual_status[v] >= self.node_degree[v]]

        while unstable:
            for v in unstable:
                # normal node: it loses exactly deg(v) grains
                actual_status[v] -= self.node_degree[v]
                toppled_nodes.add(v)
                total_topplings += 1

                # now distribute: each neighbour gets +1 grain
                for u in self.node_neighbors[v]:
                    if self.sink is not None and u == self.sink:
                        continue  # grain lost to sink
                    if self.f > 0 and np.random.random() < self.f:
                        continue  # grain dissipated stochastically
                    actual_status[u] += 1

            # after this round of topples, check if new nodes became unstable
            unstable = [v for v in self.non_sink_nodes
                        if self.node_degree[v] > 0 and actual_status[v] >= self.node_degree[v]]

        return {
            "toppled_nodes": toppled_nodes,
            "total_topplings": total_topplings,
            "avalanche_size": len(toppled_nodes),
        }

    def iteration(self, node=None):
        """
        Run one sandpile step: add grain(s) then topple until stable.

        :param node: where to drop the grain (random if None)
        :return: dict with 'iteration' number, 'status' snapshot, and avalanche stats
        """
        # take a snapshot of the current grain counts
        actual_status = {n: self.status[n] for n in self.graph.nodes}

        # iteration 0 is just the initial configuration, nothing happens yet
        if self.actual_iteration == 0:
            self.actual_iteration += 1
            return {"iteration": 0, "status": actual_status.copy(),
                    "toppled_nodes": set(), "total_topplings": 0, "avalanche_size": 0}

        # --- step 1: add grains ---
        # standard sandpile: just drop one grain on a single node
        self.___add_grain(actual_status, node=node)

        # --- step 2: topple until everything is stable ---
        avalanche = self.___topple(actual_status)

        # --- step 3: save the new state ---
        self.status = actual_status
        self.actual_iteration += 1

        return {"iteration": self.actual_iteration - 1, "status": actual_status.copy(),
                **avalanche}
