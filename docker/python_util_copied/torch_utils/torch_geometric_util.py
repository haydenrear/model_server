import torch
from torch_geometric.utils import add_self_loops, dense_to_sparse


def create_star_edge_index(num_graphs, add_self_loops_bool: bool = False):
    first = torch.tensor([list(range(1, num_graphs)), [0] * (num_graphs - 1)])
    second = torch.tensor([[0] * (num_graphs - 1), list(range(1, num_graphs))])
    edge = torch.concat([first, second], dim=1)
    if add_self_loops_bool:
        edge_index, _ = add_self_loops(edge)
        return edge_index
    return edge


def create_fully_connected_edge_list(num_nodes):
    adjacency_matrix = create_fully_connected_adjacency(num_nodes)
    edge_index = dense_to_sparse(adjacency_matrix)[0]
    return edge_index


def create_fully_connected_edge_list_tensor(num_nodes):
    edge_index = create_fully_connected_edge_list(num_nodes)
    return torch.tensor(edge_index).T


def create_fully_connected_adjacency(num_nodes):
    adjacency_matrix = torch.ones((num_nodes, num_nodes), dtype=torch.float) - torch.eye(num_nodes, dtype=torch.float)
    return adjacency_matrix.to(torch.int)
