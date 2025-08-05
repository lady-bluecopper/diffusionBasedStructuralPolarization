from datetime import date
from glob import glob
import networkx as nx
import os
import pandas as pd
import sys
import time
from tqdm.contrib.concurrent import process_map

sys.path.append('./')
import config as cfg
import partition_algorithms as pa
import polarization_algorithms as pol 
import utils as ut


def parallel_approximate_rwc(inp):
    
    graph = inp[0]
    prefix = inp[1]
    header = inp[2]
    sep = inp[3]
    is_directed = inp[4]
    pa_method = inp[5]
    rwc_lst = inp[6]
    graph_name = inp[7]
    compute_pr = inp[8]
    save_pr = inp[9]
    num_samples = inp[10]
    sample_sizes = inp[11]
    alpha = inp[12]
    
    graph_path = os.path.join(prefix, graph)
    G_obs, _ = ut.read_nx_graph(graph_path, header=header, sep=sep, directed=is_directed)
    GG, pp = pa.get_partitioning(G_obs, method=pa_method)
    g, p, _ = ut.remapped_gt_graph(GG, pp)
    node_colors = dict()
    for pid in [0, 1]:
        for v in pp[pid]:
            node_colors[v] = pid
    nx.set_node_attributes(GG, values=node_colors, name='color')
    ass = nx.attribute_assortativity_coefficient(GG, "color")
    
    file_name = f'Graph={graph_name}_D{is_directed}_A{alpha}'
    print(f'Processing {graph_name}')
    
    outputs = []
    for sample_size in sample_sizes:
        for seed in range(num_samples):
            aargs = dict()
            aargs['g'] = g
            aargs['p'] = p
            aargs['alpha'] = alpha
            aargs['sample_size'] = sample_size
            aargs['seed'] = seed
            aargs['compute_pr'] = compute_pr
            aargs['save_pr'] = save_pr
            aargs['filename'] = file_name
            
            for score in rwc_lst:
                st = time.time()
                rwc_plus = pol.get_rwc_score(score, aargs)
                rwc_plus_t = time.time() - st

                out = dict()
                out[f'DSP{score}__{alpha}'] = [rwc_plus, rwc_plus_t]
                out['Sample Size'] = sample_size
                out['Sample Id'] = seed
                out['Num Nodes'] = GG.number_of_nodes()
                out['Dataset'] = graph_name
                out['Assortativity'] = ass
                outputs.append(out)
    return pd.concat(ut.process_output_pol_measures(outputs))


if __name__ == '__main__':

    graph_names = cfg.graph_names
    data_dir = cfg.data_dir
    pa_method = cfg.pa_method
    header = cfg.header
    sep = cfg.sep
    extension = cfg.extension
    is_directed = cfg.directed
    rwc_lst = cfg.scores
    compute_pr = cfg.compute_pr
    save_pr = cfg.save_pr
    exp_name = cfg.exp_name
    num_samples = cfg.num_samples
    sample_sizes = cfg.sample_sizes
    alpha = cfg.alpha

    today = date.today()
    
    if len(graph_names) == 0:
        graph_names = glob(f'{data_dir}/*{extension}')
        prefix = ''
    else:
        prefix = data_dir
    
    inputs = []
    for graph in graph_names:
        graph_path = os.path.join(prefix, graph)
        graph_name = os.path.basename(graph)
        inputs.append([graph,
                       prefix,
                       header,
                       sep,
                       is_directed,
                       pa_method,
                       rwc_lst,
                       graph_name,
                       compute_pr,
                       save_pr,
                       num_samples,
                       sample_sizes,
                       alpha])
                
    dfs = process_map(parallel_approximate_rwc, inputs, max_workers=cfg.max_workers)
    dfs_concat = pd.concat(dfs)
    dfs_concat['Directed'] = is_directed
    dfs_concat['Partitioner'] = pa_method
    
    out_path = os.path.join(cfg.out_dir, f'exp_approx_rwc_real_graphs__DIR{exp_name}__PA{pa_method}__D{is_directed}__{today}.csv')
    dfs_concat.to_csv(out_path, sep=',', header=True, index=False)
