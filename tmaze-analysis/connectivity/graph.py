"""
Graph theory metrics for T-maze connectivity analysis.

Provides network topology measures using NetworkX.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    import bct
    HAS_BCT = True
except ImportError:
    HAS_BCT = False


@dataclass
class GraphMetrics:
    """Container for graph theory metrics."""
    # Global metrics
    density: float
    global_efficiency: float
    clustering_coefficient: float
    characteristic_path_length: float
    modularity: float
    small_worldness: Optional[float] = None

    # Node-level metrics
    degree: np.ndarray = None
    betweenness: np.ndarray = None
    eigenvector_centrality: np.ndarray = None
    participation_coefficient: np.ndarray = None
    local_efficiency: np.ndarray = None

    # Community structure
    community_assignments: np.ndarray = None
    n_communities: int = 0

    # Node names
    node_names: Optional[List[str]] = None

    metadata: Dict = field(default_factory=dict)

    def get_hubs(self, metric: str = 'degree', threshold: float = 2.0) -> List[int]:
        """Get hub nodes based on z-scored metric."""
        if metric == 'degree':
            values = self.degree
        elif metric == 'betweenness':
            values = self.betweenness
        elif metric == 'eigenvector':
            values = self.eigenvector_centrality
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if values is None:
            return []

        z = (values - np.mean(values)) / np.std(values)
        return list(np.where(z > threshold)[0])


def compute_graph_metrics(
    adjacency: np.ndarray,
    weighted: bool = True,
    node_names: Optional[List[str]] = None,
    normalize: bool = True
) -> GraphMetrics:
    """
    Compute comprehensive graph theory metrics.

    Parameters
    ----------
    adjacency : np.ndarray
        Adjacency matrix (symmetric, non-negative)
    weighted : bool
        Treat as weighted graph
    node_names : List[str], optional
        Names for nodes
    normalize : bool
        Normalize metrics to [0,1] where applicable

    Returns
    -------
    GraphMetrics
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx required for graph analysis")

    n_nodes = adjacency.shape[0]
    adj = adjacency.copy()

    # Ensure symmetric and no self-loops
    adj = (adj + adj.T) / 2
    np.fill_diagonal(adj, 0)

    # Create NetworkX graph
    if weighted:
        G = nx.from_numpy_array(adj)
    else:
        G = nx.from_numpy_array((adj > 0).astype(float))

    # Global metrics
    density = nx.density(G)

    # Efficiency and path length (need connected graph)
    if nx.is_connected(G):
        char_path = nx.average_shortest_path_length(G, weight='weight' if weighted else None)
        glob_eff = nx.global_efficiency(G)
    else:
        # Use largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        G_cc = G.subgraph(largest_cc).copy()
        char_path = nx.average_shortest_path_length(G_cc, weight='weight' if weighted else None)
        glob_eff = nx.global_efficiency(G)

    # Clustering
    if weighted:
        clustering = np.array(list(nx.clustering(G, weight='weight').values()))
    else:
        clustering = np.array(list(nx.clustering(G).values()))
    mean_clustering = np.mean(clustering)

    # Modularity (using Louvain)
    communities = nx.community.louvain_communities(G, weight='weight' if weighted else None,
                                                    seed=42)
    modularity = nx.community.modularity(G, communities, weight='weight' if weighted else None)
    n_communities = len(communities)

    # Community assignments
    community_assignments = np.zeros(n_nodes, dtype=int)
    for i, comm in enumerate(communities):
        for node in comm:
            community_assignments[node] = i

    # Node-level metrics
    degree = np.array([d for n, d in G.degree(weight='weight' if weighted else None)])

    betweenness = np.array(list(nx.betweenness_centrality(
        G, weight='weight' if weighted else None, normalized=normalize
    ).values()))

    try:
        eigenvector = np.array(list(nx.eigenvector_centrality(
            G, weight='weight' if weighted else None, max_iter=1000
        ).values()))
    except nx.PowerIterationFailedConvergence:
        eigenvector = np.zeros(n_nodes)

    # Local efficiency
    local_eff = np.array([nx.local_efficiency(G.subgraph(list(G.neighbors(n)) + [n]))
                         for n in G.nodes()])

    # Participation coefficient
    if HAS_BCT:
        participation = bct.participation_coef(adj, community_assignments)
    else:
        participation = _participation_coefficient(adj, community_assignments)

    # Small-worldness
    small_world = _compute_small_worldness(adj, mean_clustering, char_path)

    return GraphMetrics(
        density=density,
        global_efficiency=glob_eff,
        clustering_coefficient=mean_clustering,
        characteristic_path_length=char_path,
        modularity=modularity,
        small_worldness=small_world,
        degree=degree,
        betweenness=betweenness,
        eigenvector_centrality=eigenvector,
        participation_coefficient=participation,
        local_efficiency=local_eff,
        community_assignments=community_assignments,
        n_communities=n_communities,
        node_names=node_names,
        metadata={'weighted': weighted, 'n_nodes': n_nodes}
    )


def _participation_coefficient(
    adjacency: np.ndarray,
    community_assignments: np.ndarray
) -> np.ndarray:
    """Compute participation coefficient without BCT."""
    n_nodes = adjacency.shape[0]
    communities = np.unique(community_assignments)
    participation = np.zeros(n_nodes)

    for i in range(n_nodes):
        ki = np.sum(adjacency[i])
        if ki == 0:
            continue

        sum_sq = 0
        for c in communities:
            mask = community_assignments == c
            kic = np.sum(adjacency[i, mask])
            sum_sq += (kic / ki) ** 2

        participation[i] = 1 - sum_sq

    return participation


def _compute_small_worldness(
    adjacency: np.ndarray,
    clustering: float,
    path_length: float,
    n_random: int = 100
) -> float:
    """
    Compute small-world index (sigma).

    sigma = (C/C_random) / (L/L_random)
    sigma > 1 indicates small-world organization.
    """
    if not HAS_NETWORKX:
        return np.nan

    n_nodes = adjacency.shape[0]
    n_edges = int(np.sum(adjacency > 0) / 2)

    if n_edges == 0:
        return np.nan

    # Generate random graphs
    c_random = []
    l_random = []

    for _ in range(n_random):
        G_rand = nx.gnm_random_graph(n_nodes, n_edges, seed=None)
        if nx.is_connected(G_rand):
            c_random.append(nx.average_clustering(G_rand))
            l_random.append(nx.average_shortest_path_length(G_rand))

    if len(c_random) < 10:
        return np.nan

    c_rand = np.mean(c_random)
    l_rand = np.mean(l_random)

    if c_rand == 0 or l_rand == 0:
        return np.nan

    sigma = (clustering / c_rand) / (path_length / l_rand)
    return sigma


def modularity_detection(
    adjacency: np.ndarray,
    method: str = 'louvain',
    resolution: float = 1.0,
    n_iterations: int = 100
) -> Tuple[np.ndarray, float, int]:
    """
    Detect community structure in network.

    Parameters
    ----------
    adjacency : np.ndarray
        Adjacency matrix
    method : str
        'louvain', 'label_propagation', or 'spectral'
    resolution : float
        Resolution parameter for Louvain (higher = more communities)
    n_iterations : int
        Number of iterations for consensus

    Returns
    -------
    assignments : np.ndarray
        Community assignment for each node
    modularity : float
        Modularity score
    n_communities : int
        Number of communities detected
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx required")

    G = nx.from_numpy_array(adjacency)
    n_nodes = adjacency.shape[0]

    if method == 'louvain':
        # Run multiple times and take best
        best_mod = -1
        best_communities = None

        for _ in range(n_iterations):
            communities = nx.community.louvain_communities(
                G, weight='weight', resolution=resolution, seed=None
            )
            mod = nx.community.modularity(G, communities, weight='weight')
            if mod > best_mod:
                best_mod = mod
                best_communities = communities

        assignments = np.zeros(n_nodes, dtype=int)
        for i, comm in enumerate(best_communities):
            for node in comm:
                assignments[node] = i

        return assignments, best_mod, len(best_communities)

    elif method == 'label_propagation':
        communities = list(nx.community.label_propagation_communities(G))
        mod = nx.community.modularity(G, communities, weight='weight')

        assignments = np.zeros(n_nodes, dtype=int)
        for i, comm in enumerate(communities):
            for node in comm:
                assignments[node] = i

        return assignments, mod, len(communities)

    elif method == 'spectral':
        from sklearn.cluster import SpectralClustering

        # Estimate number of clusters
        n_clusters = max(2, int(np.sqrt(n_nodes / 2)))

        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42
        )
        assignments = clustering.fit_predict(adjacency)

        # Compute modularity
        communities = [set(np.where(assignments == i)[0]) for i in range(n_clusters)]
        mod = nx.community.modularity(G, communities, weight='weight')

        return assignments, mod, n_clusters

    else:
        raise ValueError(f"Unknown method: {method}")


def small_world_index(
    adjacency: np.ndarray,
    n_random: int = 100
) -> Dict:
    """
    Compute small-world metrics (sigma and omega).

    Parameters
    ----------
    adjacency : np.ndarray
        Adjacency matrix
    n_random : int
        Number of random graphs for comparison

    Returns
    -------
    Dict
        sigma, omega, and component metrics
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx required")

    G = nx.from_numpy_array(adjacency)
    n_nodes = adjacency.shape[0]
    n_edges = int(np.sum(adjacency > 0) / 2)

    # Observed metrics
    clustering = nx.average_clustering(G, weight='weight')
    if nx.is_connected(G):
        path_length = nx.average_shortest_path_length(G, weight='weight')
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        G_cc = G.subgraph(largest_cc)
        path_length = nx.average_shortest_path_length(G_cc, weight='weight')

    # Random graphs (Erdos-Renyi)
    c_random = []
    l_random = []

    for _ in range(n_random):
        G_rand = nx.gnm_random_graph(n_nodes, n_edges)
        if nx.is_connected(G_rand):
            c_random.append(nx.average_clustering(G_rand))
            l_random.append(nx.average_shortest_path_length(G_rand))

    # Lattice (ring) graph for omega
    try:
        G_lattice = nx.watts_strogatz_graph(n_nodes, 4, 0)  # Regular ring
        c_lattice = nx.average_clustering(G_lattice)
        if nx.is_connected(G_lattice):
            l_lattice = nx.average_shortest_path_length(G_lattice)
        else:
            l_lattice = np.inf
    except Exception:
        c_lattice = 1
        l_lattice = n_nodes / 4

    c_rand = np.mean(c_random) if c_random else clustering
    l_rand = np.mean(l_random) if l_random else path_length

    # Sigma (Humphries & Gurney)
    sigma = (clustering / c_rand) / (path_length / l_rand) if (c_rand > 0 and l_rand > 0) else np.nan

    # Omega (Telesford et al.)
    # omega = L_random/L - C/C_lattice
    omega = (l_rand / path_length) - (clustering / c_lattice) if (path_length > 0 and c_lattice > 0) else np.nan

    return {
        'sigma': sigma,
        'omega': omega,
        'clustering': clustering,
        'path_length': path_length,
        'c_random': c_rand,
        'l_random': l_rand,
        'c_lattice': c_lattice,
        'l_lattice': l_lattice
    }


def rich_club_coefficient(
    adjacency: np.ndarray,
    k_levels: Optional[List[int]] = None,
    normalize: bool = True,
    n_random: int = 100
) -> Dict:
    """
    Compute rich club coefficient.

    Rich club = tendency for high-degree nodes to connect to each other.

    Parameters
    ----------
    adjacency : np.ndarray
        Adjacency matrix
    k_levels : List[int], optional
        Degree levels to compute (default: all observed)
    normalize : bool
        Normalize by random network
    n_random : int
        Number of random graphs

    Returns
    -------
    Dict
        Rich club coefficients and significance
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx required")

    G = nx.from_numpy_array(adjacency)

    # Compute rich club
    rc = nx.rich_club_coefficient(G, normalized=False)

    if k_levels is None:
        k_levels = sorted(rc.keys())

    # Normalize by random networks
    if normalize:
        rc_random = {k: [] for k in k_levels if k in rc}

        for _ in range(n_random):
            G_rand = nx.configuration_model([d for n, d in G.degree()])
            G_rand = nx.Graph(G_rand)  # Remove parallel edges
            G_rand.remove_edges_from(nx.selfloop_edges(G_rand))

            try:
                rc_r = nx.rich_club_coefficient(G_rand, normalized=False)
                for k in rc_random:
                    if k in rc_r:
                        rc_random[k].append(rc_r[k])
            except Exception:
                continue

        # Normalize
        rc_norm = {}
        for k in k_levels:
            if k in rc and k in rc_random and len(rc_random[k]) > 0:
                rc_norm[k] = rc[k] / np.mean(rc_random[k])
            elif k in rc:
                rc_norm[k] = rc[k]

        return {
            'rich_club_raw': rc,
            'rich_club_normalized': rc_norm,
            'k_levels': k_levels,
            'n_random': n_random
        }

    return {
        'rich_club_raw': rc,
        'k_levels': k_levels
    }


def hub_identification(
    adjacency: np.ndarray,
    metrics: Optional[GraphMetrics] = None,
    threshold: float = 1.5,
    method: str = 'multi'
) -> Dict:
    """
    Identify hub nodes in the network.

    Parameters
    ----------
    adjacency : np.ndarray
        Adjacency matrix
    metrics : GraphMetrics, optional
        Pre-computed metrics
    threshold : float
        Z-score threshold for hub identification
    method : str
        'degree', 'betweenness', 'eigenvector', or 'multi' (consensus)

    Returns
    -------
    Dict
        Hub nodes and their scores
    """
    if metrics is None:
        metrics = compute_graph_metrics(adjacency)

    n_nodes = adjacency.shape[0]

    # Z-score each metric
    def zscore(x):
        return (x - np.mean(x)) / np.std(x) if np.std(x) > 0 else np.zeros_like(x)

    z_degree = zscore(metrics.degree)
    z_between = zscore(metrics.betweenness) if metrics.betweenness is not None else np.zeros(n_nodes)
    z_eigen = zscore(metrics.eigenvector_centrality) if metrics.eigenvector_centrality is not None else np.zeros(n_nodes)

    if method == 'degree':
        hub_mask = z_degree > threshold
    elif method == 'betweenness':
        hub_mask = z_between > threshold
    elif method == 'eigenvector':
        hub_mask = z_eigen > threshold
    elif method == 'multi':
        # Consensus: hub if above threshold on at least 2 metrics
        hub_score = (z_degree > threshold).astype(int) + \
                    (z_between > threshold).astype(int) + \
                    (z_eigen > threshold).astype(int)
        hub_mask = hub_score >= 2
    else:
        raise ValueError(f"Unknown method: {method}")

    hub_indices = np.where(hub_mask)[0]

    return {
        'hub_indices': hub_indices,
        'hub_names': [metrics.node_names[i] for i in hub_indices] if metrics.node_names else None,
        'n_hubs': len(hub_indices),
        'z_degree': z_degree,
        'z_betweenness': z_between,
        'z_eigenvector': z_eigen,
        'threshold': threshold,
        'method': method
    }
