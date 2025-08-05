import networkx as nx
import numpy as np
import sys
sys.path.append('./')
import partition_algorithms as pa
import utils as ut


def generate_gnml_graph_2_labels(n: int,
                                 average_degree: float,
                                 seed: int,
                                 directed: bool = False,
                                 skew: float = 0.5):
    '''
    Samples a random graph from the null model G(n,p,l).
    Node labels are binary.
    '''
    np.random.seed(seed)
    p_d = average_degree / (n - 1)
    G_rand = nx.erdos_renyi_graph(n, p_d, seed=seed, directed=directed)
    G = pa.get_giant_component(G_rand)
    GG, _, _ = ut.remapped_nx_graph(G, [])
    nodes = list(GG.nodes())
    p0_size = int(len(nodes) * skew)
    np.random.shuffle(nodes)
    pp = [nodes[:p0_size], nodes[p0_size:]]
    return GG, pp

    
def generate_gnml_graph_k_labels(n: int,
                                 average_degree: float,
                                 seed: int,
                                 ratio_nodes_per_label: list[float],
                                 directed: bool = False,
                                 ):
    '''
    Samples a random graph from the null model G(n,p,l).
    Node labels are sampled from [0, ..., k - 1] 
    according to ratio_nodes_per_label.
    '''
    np.random.seed(seed)
    p_d = average_degree / (n - 1)
    G_rand = nx.erdos_renyi_graph(n, p_d, seed=seed, directed=directed)
    G = pa.get_giant_component(G_rand)
    GG, _, _ = ut.remapped_nx_graph(G, [])
    nodes = list(GG.nodes())
    actual_n = len(nodes)
    num_l = len(ratio_nodes_per_label)
    
    np.random.shuffle(nodes)
    p_sizes = [int(actual_n * ratio_nodes_per_label[i]) for i in range(num_l)]
    pp = []
    curs = 0
    for i in range(num_l):
        if i < num_l - 1:
            pp.append(nodes[curs:curs+p_sizes[i]])
        else:
            pp.append(nodes[curs:])
        curs += p_sizes[i]
    return GG, pp
