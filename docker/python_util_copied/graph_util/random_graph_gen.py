import random as rn

import networkx as nx
import torch

from python_util.graph_util.graph_utils import create_random_edgelist


class RandomGraph:
    def __init__(self, nodes_list: list[int], num_nodes: int, graph: nx.Graph):
        self.num_nodes = num_nodes
        self.graph = graph
        self.nodes_list = nodes_list


class RandomGraphGenerator:

    @staticmethod
    def generate_random_graphs(max_size: int, min_size: int,
                               proba_cxn: float, num_graphs_gen: int) -> list[RandomGraph]:
        graphs = []
        random = rn.Random()
        for _ in range(num_graphs_gen):
            graph = nx.Graph()
            num_nodes = random.randint(min_size, max_size)
            for i in range(num_nodes):
                graph.add_node(i)
            for r in create_random_edgelist(num_nodes, proba_cxn):
                from_ = r[0]
                to = r[1]
                graph.add_edge(from_, to)
            graphs.append(RandomGraph([i for i in range(num_nodes)], num_nodes, graph))

        return graphs

    @staticmethod
    def generate_random_edgelist(max_size: int, min_size: int, num_gen: int) -> list[list[list[int]]]:
        random = rn.Random()
        edgelists = []
        for _ in range(num_gen):
            num_nodes = random.randint(min_size, max_size)
            proba_cxn = random.random()
            edgelists.append(create_random_edgelist(num_nodes, proba_cxn))

        return edgelists

