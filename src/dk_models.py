import networkx as nx
from typing import List
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import sys
sys.path.append('./')
import partition_algorithms as pa
import polarization_algorithms as pol


def zerok(R: nx.Graph,      # observed graph
          n_samples: int,   # number of samples to generate
          k: int,           # number of influencers for pol measures
          rwc_lst: List[str],   # list of RWC++ variants
          pa_method: str,   # partitioning algorithm
          size: int,        # size of the graph (num nodes)
          sk: float,        # skew
          gname: str,       # graph name
          max_workers: int  # number of workers for parallel computation
         ):
    '''
    dk-series null model: preserves number of nodes
    and number of edges
    '''
    
    n = R.number_of_nodes()
    m = R.number_of_edges()
    
    inputs = []
    for i in range(n_samples):
        dG = nx.gnm_random_graph(n, m, seed=i)
        G, p = pa.get_partitioning(dG, method=pa_method)
        args = {
            'Size': size,
            'Skew': sk,
            'Graph': gname,
            'Null Model': '0k',
            'Sample ID': i
        }
        inputs.append([G, p, k, rwc_lst, args])
    print('Input Generated')
    if max_workers > 1:
        return  process_map(parallel_null_model,
                           inputs, 
                           max_workers=max_workers)
    outputs = []
    for inp in tqdm(inputs):
        outputs.append(parallel_null_model(inp))
    return outputs


def onek(R: nx.Graph,      # observed graph
         n_samples: int,   # number of samples to generate
         k: int,           # number of influencers for pol measures
         rwc_lst: List[str],   # list of RWC++ variants
         pa_method: str,   # partitioning algorithm
         size: int,        # size of the graph (num nodes)
         sk: float,        # skew
         gname: str,       # graph name
         max_workers: int  # number of workers for parallel computation
        ):
    '''
    dk-series null model: preserves number of nodes,
    number of edges, and degree sequence
    '''
    
    degree_sequence = [p[1] for p in list(R.degree())] # type: ignore
    
    inputs = []
    for i in range(n_samples):
        dG = nx.Graph(nx.configuration_model(degree_sequence, seed=i))
        dG.remove_edges_from(nx.selfloop_edges(dG))
        G, p = pa.get_partitioning(dG, method=pa_method)
        args = {
            'Size': size,
            'Skew': sk,
            'Graph': gname,
            'Null Model': '1k',
            'Sample ID': i
        }
        inputs.append([G, p, k, rwc_lst, args])
    print('Input Generated')
    if max_workers > 1:
        return  process_map(parallel_null_model,
                            inputs, 
                            max_workers=max_workers)
    outputs = []
    for inp in tqdm(inputs):
        outputs.append(parallel_null_model(inp))
    return outputs


def twok(R: nx.Graph,      # observed graph
         n_samples: int,   # number of samples to generate
         k: int,           # number of influencers for pol measures
         rwc_lst: List[str],   # list of RWC++ variants
         pa_method: str,   # partitioning algorithm
         size: int,        # size of the graph (num nodes)
         sk: float,        # skew
         gname: str,       # graph name
         max_workers: int  # number of workers for parallel computation
        ):
    '''
    dk-series null model: preserves number of nodes,
    number of edges, degree sequence, and joint
    degree sequence
    '''
    
    R.remove_edges_from(nx.selfloop_edges(R))
    deg_dict = dict(R.degree()) # type: ignore
    nx.set_node_attributes(R, deg_dict, "degree")
    joint_degrees = nx.attribute_mixing_dict(R, "degree")
    
    inputs = []
    for i in range(n_samples):
        dG = nx.joint_degree_graph(joint_degrees, seed=i)
        G, p = pa.get_partitioning(dG, method=pa_method)
        args = {
            'Size': size,
            'Skew': sk,
            'Graph': gname,
            'Null Model': '2k',
            'Sample ID': i
        }
        inputs.append([G, p, k, rwc_lst, args])
    print('Input Generated')
    if max_workers > 1:
        return  process_map(parallel_null_model,
                           inputs, 
                           max_workers=max_workers)
    outputs = []
    for inp in tqdm(inputs):
        outputs.append(parallel_null_model(inp))
    return outputs


def get_output_nm(name: str,        # null model
                  R: nx.Graph,      # observed graph
                  n_samples: int,   # number of samples to generate
                  k: int,           # number of influencers for pol measures
                  rwc_lst: List[str],   # list of RWC++ variants
                  pa_method: str,   # partitioning algorithm
                  size: int,        # size of the graph (num nodes)
                  sk: float,        # skew
                  gname: str,       # graph name
                  max_workers: int  # number of workers for parallel computation
                  ):
    # calls the null model with name 'name'
    nm_map = {
        '0k': zerok,
        '1k': onek,
        '2k': twok
    }
    return nm_map[name](R, n_samples, k, rwc_lst, pa_method, size, sk, gname, max_workers)


def parallel_null_model(inp):
    # compute the polarization measure on graph gg
    gg = inp[0]
    pp = inp[1]
    k = inp[2]
    rwc_lst = inp[3]
    args = inp[4]
    
    out = pol.compute_pol_measures(gg, pp, k, rwc_lst=rwc_lst)
    out.update(args)
    return out
