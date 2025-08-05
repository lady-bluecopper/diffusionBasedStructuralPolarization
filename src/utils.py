from collections import defaultdict
import graph_tool as gt
import networkx as nx
import numpy as np
import pandas as pd
import sys
from typing import Dict, Collection, Tuple

sys.path.append('./')
import partition_algorithms as pa


def read_edges(file_path: str, # path to input graph
               uniques: bool=True, # whether we want to keep multiple occurrences of the same edge
               sep: str=',', # separator of edge endpoints
               header: bool=False):  # whether the input file has a header
    '''
    Read a graph as a list of edges and remap vertex ids in
    the range [0, ..., n - 1].
    '''
    edges = list()
    vmap = dict()
    counter = 0
    with open(file_path) as in_f:
        if header:
            # remove header
            in_f.readline()
        for line in in_f.readlines():
            lst = line.strip().split(sep)
            u = lst[0].strip()
            v = lst[1].strip()
            if u not in vmap:
                vmap[u] = counter
                counter += 1
            if v not in vmap:
                vmap[v] = counter
                counter += 1
            try:
                w = int(lst[2].strip())
            except:
                w = 1
            if u != v:
                for _ in range(w):
                    edges.append((vmap[u], vmap[v]))
    if uniques:
        return set(edges), vmap
    return edges, vmap


def read_nx_graph(file_path: str, 
                  uniques: bool=True, 
                  sep: str=',', 
                  header: bool=False, 
                  directed: bool=False):
    '''
    Read a graph as a nx.Graph object.
    '''
    edges, vmap = read_edges(file_path, uniques, sep, header)
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    G.add_edges_from(edges)
    return G, vmap


def read_nx_gcc(file_path: str, 
                uniques: bool=True, 
                sep: str=',', 
                header: bool=False, 
                directed: bool=False):
    '''
    Read a graph as a nx.Graph object.
    '''
    G, vmap = read_nx_graph(file_path, uniques, sep, header, directed)
    GG = pa.get_giant_component(G)
    G_p, _, vmap_p = remapped_nx_graph(GG, [])
    vmap_pp = dict()
    for k, v in vmap.items():
        if v in vmap_p:
            vmap_pp[k] = vmap_p[v]
    return G_p, vmap_pp


def read_weighted_edges(file_path: str, 
                        sep: str=',', 
                        header: bool=True, 
                        directed: bool=False):
    '''
    Read a graph as a dictionary of weighted edges.
    The key is the edge and its value is the weight.
    '''
    edges = defaultdict(int)
    vmap = dict()
    counter = 0
    with open(file_path) as in_f:
        if header:
            # remove header
            in_f.readline()
        for line in in_f.readlines():
            lst = line.strip().split(sep)
            u = lst[0].strip()
            v = lst[1].strip()
            if u not in vmap:
                vmap[u] = counter
                counter += 1
            if v not in vmap:
                vmap[v] = counter
                counter += 1
            try:
                w = int(lst[2].strip())
            except:
                w = 1
            if u != v:
                if directed or vmap[u] < vmap[v]:
                    edges[(vmap[u], vmap[v])] += w
                else:
                    edges[(vmap[v], vmap[u])] += w
    return edges, vmap


def read_labels(file_path: str, 
                sep: str=',', 
                header: bool=False):
    '''
    Read the dictionary of node labels.
    '''
    labels = dict()
    with open(file_path) as in_f:
        if header:
            # remove header
            in_f.readline()
        for line in in_f.readlines():
            lst = line.strip().split(sep)
            u = lst[0].strip()
            lab = lst[1].strip()
            labels[u] = lab
    return labels


def compute_JLM_from_list(edges, node_labels: Dict[int, int]):
    '''
    Given a collection of edges,
    computes the JLM of the corresponding multigraph.
    '''
    num_labels = len(set(node_labels.values()))
    jlm = np.zeros((num_labels, num_labels), np.int32)
    for e in edges:
        l1 = node_labels[e[0]]
        l2 = node_labels[e[1]]
        jlm[l1][l2] += 1
        if l1 != l2:
            jlm[l2][l1] += 1
    return jlm


def remapped_gt_graph(G: nx.Graph, partitions):
    new_edges, pp, vmap = remap_vertices_idx(G, partitions)
    is_dir = G.is_directed()
    g = gt.Graph(directed=is_dir)
    g.add_edge_list(new_edges)
    return g, pp, vmap


def remapped_nx_graph(G: nx.Graph, partitions):
    new_edges, pp, vmap = remap_vertices_idx(G, partitions)
    if G.is_directed():
        g = nx.DiGraph()
    else:
        g = nx.Graph()
    g.add_edges_from(new_edges)
    return g, pp, vmap


def gt_to_nx_graph(G: gt.Graph):
    if G.is_directed():
        g = nx.DiGraph()
    else:
        g = nx.Graph()
    g.add_nodes_from(G.get_vertices())
    g.add_edges_from(G.get_edges())
    return g    


def remap_vertices_idx(G, partitions):
    edges = G.edges()
    vmap = dict()
    new_edges = []
    counter = 0
    for e in edges:
        if e[0] not in vmap:
            vmap[e[0]] = counter
            counter += 1
        if e[1] not in vmap:
            vmap[e[1]] = counter
            counter += 1
        new_edges.append((vmap[e[0]], vmap[e[1]]))
    pp = []
    for part in partitions:
        new_part = []
        for node in part:
            if node in vmap:
                new_part.append(vmap[node])
        pp.append(new_part)
    return new_edges, pp, vmap   


def compute_degree_sequence_from_list(edges: Collection[Tuple[int, int]]):
    '''
    Given a list of edges, computes the degree
    of each node in the corresponding multigraph.
    '''
    degs = defaultdict(int)
    for e in edges:
        degs[e[0]] += 1
        degs[e[1]] += 1
    return degs


def process_output_pol_measures(outputs):
    '''
    Generates a dataframe from the output of
    an experiment.
    '''
    dfs = []
    outs = outputs
    if not isinstance(outputs, list):
        outs = [outputs]
    for out in outs:
        rows = []
        common_attr = []
        for key, val in out.items():
            if isinstance(val, list):
                rows.append([key, val[0], val[1]])
            else:
                common_attr.append(key)
        if len(rows) > 0:
            df = pd.DataFrame(rows, columns=['Metric', 'Value', 'Time (s)'])
        else:
            df = pd.DataFrame()
        for key in common_attr:
            df[key] = out[key]
        dfs.append(df)
    return dfs


def compute_stats(file_path: str,
                  sep: str=',',
                  header: bool=False,
                  partitioner: str='klin'):
    
    nodes = set()
    degrees = defaultdict(int)

    edges, _ = read_edges(file_path, sep=sep, header=header)
    for e in edges:
        nodes.add(e[0])
        nodes.add(e[1])
        degrees[e[0]] += 1
        degrees[e[1]] += 1
    
    avg_deg = np.mean(list(degrees.values()))
   
    g = nx.Graph()
    g.add_edges_from(edges)
    gcc, partitions = pa.get_partitioning(g, method=partitioner)
    num_gcc_nodes = gcc.number_of_nodes()
    num_gcc_edges = gcc.number_of_edges()
    size_p0 = len(partitions[0])
    size_p1 = len(partitions[1]) 
    degrees_cc = [gcc.degree[v] for v in gcc.nodes()] # type: ignore
    avg_deg_cc = np.mean(degrees_cc)

    output = {
        'Nodes': len(nodes),
        'Edges': len(edges),
        'AVG Degree': avg_deg,
        'Nodes GCC': num_gcc_nodes,
        'Edges GCC': num_gcc_edges,
        'Size P0': size_p0,
        'Size P1': size_p1,
        'AVG Degree GCC': avg_deg_cc
    }
    return output