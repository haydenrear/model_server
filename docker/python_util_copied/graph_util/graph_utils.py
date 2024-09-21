import math
from random import random
import random as rn
from typing import Tuple, Optional

import networkx as nx
import numpy as np
import torch

from python_util.torch_utils.padding import pad_add_end_to_match


def create_random_edgelist(n_nodes: int, probability: float) -> list[list[int]]:
    out_edgelist: list[list[int]] = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if rn.random() > probability:
                out_edgelist.append([i, j])

    return out_edgelist


def transform_from_adjacency_matrix(adj_matrix: torch.Tensor) -> torch.Tensor:
    edge_list = []
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i, j] != 0:
                edge_list.append([i, j])
    return torch.tensor(edge_list).T


def extract_edgelist_from_graph(starting_graph: nx.Graph, nodes: list):
    edges = []
    graph_nodes = [n for n in starting_graph.nodes]
    for from_t, to_ in nx.edges(starting_graph, graph_nodes):
        if from_t in nodes and to_ in nodes:
            edges.append([nodes.index(from_t),
                          nodes.index(to_)])

    return torch.tensor(edges).T


def extract_edgelist_subgraph(starting_edgelist: torch.Tensor, nodes: set[int]):
    out_edges = []
    assert starting_edgelist.shape[0] == 2
    assert len(starting_edgelist.shape) == 2

    for edges in starting_edgelist.T:
        from_ = int(edges[0])
        to_ = int(edges[1])
        if from_ in nodes and to_ in nodes:
            out_edges.append([from_, to_])

    return out_edges


def transform_to_adjacency_matrix(edge_list: torch.Tensor, starting: torch.Tensor = None) -> torch.Tensor:
    # Extract edges from the edge list
    src_nodes, dest_nodes = edge_list[:, 0], edge_list[:, 1]
    num_nodes = max(int(torch.max(src_nodes)) + 1, int(torch.max(dest_nodes)) + 1)

    # Create an empty adjacency matrix
    if starting is None:
        starting = torch.zeros(num_nodes, num_nodes, dtype=torch.int)

    # Map edges to positions in the adjacency matrix and set corresponding values to 1
    starting[src_nodes, dest_nodes] = 1
    starting[dest_nodes, src_nodes] = 1  # For undirected graphs (remove for directed)

    return starting


def get_node_cxns_out(graph: nx.DiGraph, node_match, node_only: bool = True):
    return [i[1] if node_only else i for i in graph.out_edges(node_match)]


def get_node_cxns_in(graph: nx.DiGraph, node_match, node_only: bool = True):
    return [i[0] if node_only else i for i in graph.in_edges(node_match)]


def get_node_cxns(graph: nx.DiGraph, node_match, node_only: bool = True,
                  filter_duplicates: bool = False):
    seen = set([])
    for n in get_node_cxns_in(graph, node_match, node_only):
        if (filter_duplicates and n not in seen) or not filter_duplicates:
            seen.add(n)
            yield n
    for n in get_node_cxns_out(graph, node_match, node_only):
        if (filter_duplicates and n not in seen) or not filter_duplicates:
            seen.add(n)
            yield n
    for from_edge, to_edge in graph.edges(node_match):
        if node_only:
            n = from_edge if from_edge != node_match else to_edge
            if (filter_duplicates and n not in seen) or not filter_duplicates:
                seen.add(n)
                yield n
        else:
            n = from_edge, to_edge
            if (filter_duplicates and n not in seen) or not filter_duplicates:
                seen.add(n)
                yield n


def get_sorted_by_key(tensor: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    sorted_keys = sorted(tensor.keys())
    return [tensor[key] for key in sorted_keys]


def modularity_of_partitions(G, partition, M):
    """
    G: The subgraph
    partition: dictionary (node: community)
    M: Matrix representing the "average" from the main graph.
    """
    A = nx.to_numpy_array(G)
    nodes = G.nodes()
    degrees = np.array([G.degree(n) for n in nodes])
    m = G.number_of_edges()

    return do_calculate_modularity(A, M, m, nodes, partition)


def do_calculate_modularity(A, M, m, nodes, partition):
    Q = 0
    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if partition[node_i] == partition[node_j]:
                Q += A[i, j] - M[i, j]
    return Q / (2 * m)


def modularity_from_subgraph(main_graph, sub_graph, M):
    """
    main_graph: The main graph
    sub_graph: The subgraph whose modularity you want to calculate
    M: Matrix representing the "average" from the main graph.
    """
    A = nx.to_numpy_array(main_graph)
    nodes = main_graph.nodes()
    sub_nodes = sub_graph.nodes()
    m = main_graph.number_of_edges()

    # Create partition based on sub_graph
    partition = {}
    for node in nodes:
        if node in sub_nodes:
            partition[node] = 1  # Sub-graph community
        else:
            partition[node] = 0  # Rest of the main graph community

    return do_calculate_modularity(A, M, m, nodes, partition)


def create_edge_list_from_edges(edges, nodes=None):
    edge_list = []
    for from_, to_ in edges:
        if nodes is None:
            edge_list.append([from_, to_])
        else:
            assert isinstance(nodes, list)
            if from_ in nodes and to_ in nodes:
                edge_list.append([nodes.index(from_), nodes.index(to_)])

    return torch.tensor(edge_list).T


def current_edges(edgelist, node):
    edgelist = edgelist.T
    first_value = edgelist[0] != node
    second_value = edgelist[1] == node
    edgelist_node = first_value & second_value
    first_edgelist: torch.Tensor = edgelist[0, edgelist_node]
    first_value = edgelist[1] != node
    second_value = edgelist[0] == node
    edgelist_node = first_value & second_value
    second_edgelist: torch.Tensor = edgelist[1, edgelist_node]
    return torch.cat((first_edgelist, second_edgelist))


def add_connections(edgelist, min_edges,
                    make_symmetric: bool = False,
                    starting_adj_proba: Optional[torch.Tensor] = None):
    """
    Adds connections to a PyTorch edgelist until each node has at least min_edges edges.

    Args:
      edgelist: (torch.LongTensor) Edge list with shape [num_edges, 2] where each row is [source, target].
      min_edges: (int) Minimum number of edges per node.
      device: (torch.device) Device where the tensors are stored (e.g., cpu, cuda).

    Returns:
      :param make_symmetric:
      :param min_edges:
      :param edgelist: (torch.LongTensor) The updated edgelist with added connections.
      :param starting_adj_proba: If provided, this is an adjacency matrix with each item representing the likelihood
      of adding an edge there if there aren't enough to meet the min_edges requirement.
    """
    adjacency_matrix = transform_to_adjacency_matrix(edgelist)
    to_add_adj = add_edges_vectorized_unique(adjacency_matrix, min_edges, make_symmetric, starting_adj_proba)
    to_add_edges = transform_from_adjacency_matrix(to_add_adj)
    edgelist = torch.cat((edgelist, torch.tensor(to_add_edges).T))
    return edgelist


def convert_from_networkx_edgelist(network_k: nx.Graph):
    edge_list = nx.to_edgelist(network_k)
    edges = []
    for from_edge, to_edge, other in edge_list:
        edges.append([from_edge, to_edge])
    return torch.tensor(edges)


def add_edges_vectorized_unique(adj_matrix, n, make_symmetric: bool = False,
                                starting_edges: Optional[torch.Tensor] = None):
    num_nodes = adj_matrix.size(0)

    # Calculate the current number of edges for each node (excluding self-loops)
    current_edges = torch.sum(adj_matrix, dim=1) - adj_matrix.diag()

    # Calculate the edges to add for each node to meet the minimum requirement 'n'
    edges_to_add = torch.maximum(torch.zeros(num_nodes, dtype=torch.int), n - current_edges)

    # Mask nodes that already have enough edges from being selected
    mask = edges_to_add > 0
    mask = mask.expand([num_nodes, num_nodes])
    masked_diag = ~torch.eye(num_nodes, dtype=torch.bool, device=adj_matrix.device)
    valid_mask = mask & masked_diag
    valid_mask = valid_mask.to(dtype=torch.int)

    # Sample unique indices for each node without self-loops
    random_indices = torch.rand(num_nodes, num_nodes, device=adj_matrix.device)
    random_indices *= valid_mask

    if starting_edges is not None:
        starting_edges *= valid_mask
        random_indices += starting_edges

    v, indices = torch.topk(random_indices, k=edges_to_add.max(), dim=1)

    v[v == 0] = 1
    indices = indices[:, :edges_to_add.max()]

    # Create masks to update adjacency matrix with additional edges
    unsqueeze = edges_to_add.unsqueeze(1).expand(-1, edges_to_add.max())
    edges_to_add_mask = unsqueeze != 0

    # Update adjacency matrix with new edges
    v_ = v.masked_fill(~edges_to_add_mask, 0.0)
    next_scattered = torch.scatter(adj_matrix.to(dtype=torch.float), 1, indices, v_)
    next_scattered += adj_matrix.to(dtype=torch.float)
    next_scattered[next_scattered > 0] = 1
    adj_matrix = next_scattered.to(dtype=torch.int)

    # Ensure symmetry in the adjacency matrix
    if make_symmetric:
        adj_matrix = torch.maximum(adj_matrix, adj_matrix.t())

    return adj_matrix


def count_num_edges(edgelist: torch.Tensor) -> torch.Tensor:
    """
    :param edgelist: An edgelist
    :return:
    """
    to_count = edgelist[:, 0]
    to_count_two = edgelist[:, 1]
    degrees = torch.bincount(to_count)  # Count in-degree for each node
    degrees_two = torch.bincount(to_count_two)  # Count in-degree for each node
    degrees_two = pad_add_end_to_match([i for i in degrees.shape], degrees_two)
    degrees = pad_add_end_to_match([i for i in degrees_two.shape], degrees)
    degrees_total = degrees + degrees_two
    return degrees_total


def concatenate_graphs(graphs: list[torch.Tensor],
                       edgelists: list[torch.Tensor],
                       add_edge_probability: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    :param graphs:
    :param edgelists:
    :param add_edge_probability:
    :return:  Tuple where first element is graph and second element is edgelist.
    """
    concatenated_edgelist = torch.empty(0, 2)  # Empty tensor to store concatenated edgelist
    # Concatenate edgelists while adjusting node indices
    node_offset = 0
    for edgelist in edgelists:
        # Increment node indices of the current graph
        edgelist_incremented = edgelist + node_offset

        # Concatenate the adjusted edgelists along the rows
        concatenated_edgelist = torch.cat((concatenated_edgelist, edgelist_incremented), dim=0)
        if node_offset > 0 and add_edge_probability != 0.0:
            for node1 in range(node_offset, edgelist_incremented.max().item() + 1):
                for node2 in range(concatenated_edgelist.max().item() + 1 - node_offset):
                    if rn.random() < add_edge_probability:
                        new_edge = torch.tensor([[node1, node2 + node_offset]])
                        concatenated_edgelist = torch.cat((concatenated_edgelist, new_edge), dim=0)

        # Update node offset for the next graph
        node_offset += edgelist.max().item() + 1

    return torch.concat(graphs), concatenated_edgelist


def concatenate_edgelists(edgelists: list[torch.Tensor]) -> torch.Tensor:
    """
    :param graphs:
    :param edgelists:
    :param add_edge_probability:
    :return:  Tuple where first element is graph and second element is edgelist.
    """
    concatenated_edgelist = torch.empty(0, 2)  # Empty tensor to store concatenated edgelist
    # Concatenate edgelists while adjusting node indices
    node_offset = 0
    for edgelist in edgelists:
        # Increment node indices of the current graph
        edgelist_incremented = edgelist + node_offset

        # Concatenate the adjusted edgelists along the rows
        concatenated_edgelist = torch.cat((concatenated_edgelist, edgelist_incremented), dim=0)

        # Update node offset for the next graph
        node_offset += edgelist.max().item() + 1

    return concatenated_edgelist
