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


def parallel_pol_measures(inp):
    
    graph = inp[0]
    prefix = inp[1]
    header = inp[2]
    sep = inp[3]
    is_directed = inp[4]
    pa_method = inp[5]
    k = inp[6]
    rwc_lst = inp[7]
    graph_name = inp[8]
    compute_pr = inp[9]
    save_pr = inp[10]
    all_pol_measures = inp[11]
    alpha = inp[12]

    graph_path = os.path.join(prefix, graph)
    G_obs, _ = ut.read_nx_graph(graph_path, header=header, sep=sep, directed=is_directed)
    GG, pp = pa.get_partitioning(G_obs, method=pa_method)
    node_colors = dict()
    for pid in [0, 1]:
        for v in pp[pid]:
            node_colors[v] = pid
    nx.set_node_attributes(GG, values=node_colors, name='color')
    ass = nx.attribute_assortativity_coefficient(GG, "color")
    
    file_name = f'Graph={graph_name}_D{is_directed}_A{alpha}'
    print(f'Processing {graph_name}, with Size |V|={GG.number_of_nodes()}, |E|={GG.number_of_edges()}')
    
    infopack = dict()
    if all_pol_measures:
        infopack = pol.compute_pol_measures(GG, pp, k,
                                            rwc_lst=rwc_lst, 
                                            alpha=alpha,
                                            compute_pr=compute_pr,
                                            save_pr=save_pr,
                                            file_name=file_name)
    else:
        gr, par, _ = ut.remapped_gt_graph(GG, pp)
        args = dict()
        args['g'] = gr
        args['p'] = par
        args['k'] = k
        args['alpha'] = alpha
        args['filename'] = file_name
        for score in rwc_lst:
            st = time.time()
            rwc = pol.get_rwc_score(score, args)
            end = time.time() - st
            infopack[f'DSP{score}__{alpha}'] = [rwc, end]
    
    infopack['Dataset'] = graph_name
    infopack['Assortativity'] = ass
    return infopack


if __name__ == '__main__':

    graph_names = cfg.graph_names
    data_dir = cfg.data_dir
    pa_method = cfg.pa_method
    header = cfg.header
    sep = cfg.sep
    extension = cfg.extension
    k = cfg.k
    is_directed = cfg.directed
    rwc_lst = cfg.scores
    compute_pr = cfg.compute_pr
    save_pr = cfg.save_pr
    exp_name = cfg.exp_name
    all_pol_measures = cfg.all_pol_measures
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
                       k,
                       rwc_lst,
                       graph_name,
                       compute_pr,
                       save_pr,
                       all_pol_measures,
                       alpha])
                
    outputs = process_map(parallel_pol_measures, inputs, max_workers=cfg.max_workers)
    dfs = ut.process_output_pol_measures(outputs)
    dfs_concat = pd.concat(dfs)
    dfs_concat['Directed'] = is_directed
    dfs_concat['Partitioner'] = pa_method
    
    out_path = os.path.join(cfg.out_dir, f'exp_real_graphs__DIR{exp_name}__PA{pa_method}__K{k}__D{is_directed}__{today}.csv')
    dfs_concat.to_csv(out_path, sep=',', header=True, index=False)
