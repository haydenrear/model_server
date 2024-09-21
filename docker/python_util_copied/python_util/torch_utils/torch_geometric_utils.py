import torch


def get_sorted_by_key(tensor: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    sorted_keys = sorted(tensor.keys())
    return [tensor[key] for key in sorted_keys]


def convert_to_adjacencies(adjacency: torch.Tensor) -> dict[int, list[int]]:
    clusters = {}
    for i, row in enumerate(adjacency):
        cluster_indices = torch.nonzero(row == 1.0)
        for index in cluster_indices:
            if int(index) in clusters:
                clusters[int(index)].append(i)
            else:
                clusters[int(index)] = [i]

    return clusters


def top_k_adjacency_matrix(num_delegations: int, input_data):
    # Initialize adjacency matrix
    max_nodes_per_cluster = len(input_data) // num_delegations + 1
    adjacency_matrix = torch.zeros((len(input_data), num_delegations))

    # Get the top cluster indices for each node based on probabilities
    _, top_clusters = torch.stack(get_sorted_by_key(input_data)).topk(
        min(num_delegations, max_nodes_per_cluster),
        dim=1
    )

    # Assign nodes to clusters based on top cluster indices
    for node_idx, clusters in enumerate(top_clusters):
        assigned_cluster = False
        for cluster_idx in clusters:
            # If cluster has fewer than the maximum nodes, assign node to cluster
            if adjacency_matrix[:, cluster_idx].sum() < max_nodes_per_cluster:
                adjacency_matrix[node_idx, cluster_idx] = 1
                assigned_cluster = True
                break
        if not assigned_cluster:
            # If the node couldn't be assigned to any top cluster, assign it to the cluster with the fewest nodes
            min_nodes_count = adjacency_matrix.sum(dim=0).min()
            min_nodes_cluster_indices = torch.nonzero(adjacency_matrix.sum(dim=0) == min_nodes_count)
            min_nodes_cluster_idx = min_nodes_cluster_indices[0, 0]
            adjacency_matrix[node_idx, min_nodes_cluster_idx] = 1

    return adjacency_matrix
