from collections import defaultdict
import metis
import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection
import numpy as np
import random
import scipy
import scipy.cluster.vq as vq
import scipy.sparse
from scipy.sparse.linalg import eigsh
from sklearn.cluster import HDBSCAN # type: ignore
import sys
from typing import Tuple, List
sys.path.insert(1, './')
import utils as ut


# REGULARIZED SPECTRAL CLUSTERING

def regularized_laplacian_matrix(adj_matrix, tau):
    """
    The original code for regularized spectral clustering
    was written by samialabed in 2018. We modified the code for our
    purposes. Original script:
    https://github.com/samialabed/regualirsed-spectral-clustering

    Using ARPACK solver, compute the first K eigen vector.
    The laplacian is computed using the regularised formula from [2]
    [2]Kamalika Chaudhuri, Fan Chung, and Alexander Tsiatas 2018.
        Spectral clustering of graphs with general degrees in
        the extended planted partition model.

    L = I - D^-1/2 * A * D ^-1/2

    :param adj_matrix: adjacency matrix representation of
     graph where [m][n] >0 if there is edge and [m][n] = weight
    :param tau: the regularisation constant
    :return: the first K eigenvector
    """
    # Code inspired from nx.normalized_laplacian_matrix,
    # with changes to allow regularisation
    n, m = adj_matrix.shape
    # I = np.eye(n, m)
    I = scipy.sparse.identity(n, dtype='int8', format='dia')
    diags = adj_matrix.sum(axis=1).flatten()
    # add tau to the diags to produce a regularised diags
    if tau != 0:
        diags = np.add(diags, tau)
    # diags will be zero at points where there
    # is no edge and/or the node you are at
    # ignore the error and make it zero later
    diags_sqrt = 1.0 / np.sqrt(diags)
    diags_sqrt[np.isinf(diags_sqrt)] = 0
    D = scipy.sparse.spdiags(diags_sqrt, [0], m, n, format='csr')
    L = I - (D.dot(adj_matrix.dot(D)))
    return L


def eigen_solver(laplacian, n_clusters):
    """
    ARPACK eigen solver in Shift-Invert Mode based on
    http://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html
    """
    lap = laplacian * -1
    v0 = np.random.uniform(-1, 1, lap.shape[0])
    eigen_values, eigen_vectors = eigsh(lap, k=n_clusters, sigma=1.0, v0=v0)
    eigen_vectors = eigen_vectors.T[n_clusters::-1]
    return eigen_values, eigen_vectors[:n_clusters].T


def regularized_spectral_clustering(adj_matrix, 
                                    tau=0, 
                                    n_clusters: int=2):
    """
    :param adj_matrix: adjacency matrix representation
    of graph where [m][n] >0 if there is edge and [m][n] = weight
    :param n_clusters: cluster partitioning constant
    :param algo: the clustering separation algorithm,
    possible value kmeans++ or scan
    :return: labels, number of clustering iterations needed,
    smallest set of cluster found, execution time
    """
    regularized_laplacian = regularized_laplacian_matrix(adj_matrix, tau)
    eigen_values, eigen_vectors = eigen_solver(regularized_laplacian,
                                               n_clusters=n_clusters)
    labels = None
    if n_clusters == 2:  # cluster based on sign
        second_eigen_vector_index = np.argsort(eigen_values)[1]
        second_eigen_vector = eigen_vectors.T[second_eigen_vector_index]
        # use only the second eigenvector
        labels = [0 if val <= 0 else 1 for val in second_eigen_vector]
    return labels


def evaluate_graph(graph: nx.Graph, 
                   n_clusters: int, 
                   reg: bool=True):
    """
    Reconstruction of [1] Understanding Regularized Spectral Clustering
    via Graph Conductance, Yilin Zhang, Karl Rohe

    :param graph: Graph to be evaluated
    :param n_clusters: How many clusters to look at
    :param graph_name: the graph name used to create checkpoints and figures
    :return:
    """
    graph_degree = graph.degree() # type: ignore
    graph_average_degree = sum(val for (_, val) in graph_degree) / graph.number_of_nodes()

    adj_matrix = nx.to_scipy_sparse_array(graph, nodelist=sorted(graph.nodes()), format='csr')
    if reg:
        tau = graph_average_degree
    else:
        tau = 0

    labels = regularized_spectral_clustering(adj_matrix,
                                             tau, # type: ignore
                                             n_clusters)
    return labels


def partition_spectral(G: nx.Graph):
    
    labels = evaluate_graph(G, 2)
    node_ids = sorted(G.nodes())
    
    partitions = dict()
    partitions[0] = []
    partitions[1] = []
    for idx, l in enumerate(labels): # type: ignore
        partitions[l].append(node_ids[idx])
    return [partitions[0], partitions[1]]


# HDBSCAN

def partition_hdbscan(G: nx.Graph, vdim: int=10):
    deg_sum = sum(d for d,_ in G.degree()) # type: ignore
    avg_deg = deg_sum / G.number_of_nodes()
    Adj = nx.to_scipy_sparse_array(G, nodelist=sorted(G.nodes()), format='csr')
    reg_lap = regularized_laplacian_matrix(Adj, avg_deg)
    lap = reg_lap * -1
    # diag_shift = 1e-5 * scipy.sparse.eye(reg_lap.shape[0])
    # lap = reg_lap + diag_shift
    _, eig_vecs = eigsh(lap, k=vdim, sigma=1.0, which='SM')
    min_size = int(G.number_of_nodes() * .01)
    incr = int(G.number_of_nodes() * .01)
    hdb = HDBSCAN(cluster_selection_method='eom',
                min_cluster_size=min_size) # leaf, eom
    hdb.fit(eig_vecs)
    dist_lab = len(set(hdb.labels_))

    last_best = hdb.labels_

    while dist_lab > 2:
        last_best = hdb.labels_
        min_size += incr
        hdb = HDBSCAN(cluster_selection_method='eom',
                      min_cluster_size=min_size) # leaf, eom
        hdb.fit(eig_vecs)
        dist_lab = len(set(hdb.labels_))
    
    if dist_lab == 1:
        labels = last_best
    else:
        labels = hdb.labels_
    _, labels = vq.kmeans2(eig_vecs, 2, minit='++')
    clusters = defaultdict(list)
    for idx, l in enumerate(labels):
        clusters[l].append(idx)
    return list(clusters.values())


# METIS
 
def partition_metis(mG: nx.Graph, ufactor=40, seed=42):
    _, parts = metis.part_graph(mG, 2, seed=seed, ufactor=ufactor*10)
    # remap outer to inner ids
    node_ids = list(mG.nodes())
    partitions = [[], []]
    for n, part in enumerate(parts):
        partitions[part].append(node_ids[n])
    return partitions


def get_giant_component(dG: nx.Graph):
    if dG.is_directed():
        Gcc = sorted(nx.weakly_connected_components(dG), key=len, reverse=True)
    else:
        Gcc = sorted(nx.connected_components(dG), key=len, reverse=True)
    G_Giant = dG.subgraph(Gcc[0])
    return G_Giant


def get_partitioning(dG: nx.Graph, 
                     method='klin', 
                     seed=0) -> Tuple[nx.Graph, List[List[int]]]:

    random.seed(seed)
    G = get_giant_component(dG)
    if method == 'rsc':
        partitions = partition_spectral(G)
    elif method == 'klin':
        G_tmp = G
        # KLIN not implemented for directed graphs
        if G.is_directed():
            G_tmp = nx.Graph()
            G_tmp.add_edges_from(G.edges())
        p1, p2 = kernighan_lin_bisection(G_tmp)
        partitions = [list(p1), list(p2)]
    elif method == 'hdbscan':
        partitions = partition_hdbscan(G)
    elif method == 'metis':
        partitions = partition_metis(G)
    else:
        raise NotImplementedError
    GG, pp, vmap = ut.remapped_nx_graph(G, partitions)
    return GG, pp
