from datetime import date
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


def parallel_pol_measures_with_communities(inp):
    
    data_dir = inp[0]
    graph_file = inp[1]
    graph_name = inp[2]
    header = inp[3]
    sep = inp[4]
    k = inp[5]
    rwc_lst = inp[6]
    all_pol_measures = inp[7]
    alpha = inp[8]

    graph_path = os.path.join(data_dir, graph_file)
    G_obs, vmap = ut.read_nx_graph(graph_path, header=header, sep=sep, directed=False)
    GCC = pa.get_giant_component(G_obs)
    GG, _, vmap2 = ut.remapped_nx_graph(GCC, [])
    
    left = []
    with open(os.path.join(data_dir, f"community1_{graph_name}.txt")) as f1:
        for line in f1:
            line = line.strip()
            left.append(vmap2[vmap[line]])
    right = []
    with open(os.path.join(data_dir, f"community2_{graph_name}.txt")) as f2:
        for line in f2:
            line = line.strip()
            right.append(vmap2[vmap[line]])
    pp = [left, right]
    
    infopack = dict()
    if all_pol_measures:
        infopack = pol.compute_pol_measures(GG, pp, k,
                                            rwc_lst=rwc_lst,
                                            alpha=alpha,
                                            compute_pr=True,
                                            save_pr=False)
    else:
        gr, par, _ = ut.remapped_gt_graph(GG, pp)
        args = dict()
        args['g'] = gr
        args['p'] = par
        args['k'] = k
        args['alpha'] = alpha
        for score in rwc_lst:
            st = time.time()
            rwc = pol.get_rwc_score(score, args)
            end = time.time() - st
            infopack[f'DSP{score}__{alpha}'] = [rwc, end]

    infopack['Dataset'] = graph_name
    return infopack


if __name__ == '__main__':

    graph_names = cfg.graph_names
    data_dir = cfg.data_dir
    exp_name = cfg.exp_name
    header = cfg.header
    sep = cfg.sep
    k = cfg.k
    rwc_lst = cfg.scores
    all_pol_measures = cfg.all_pol_measures
    alpha = cfg.alpha
    fname = cfg.get_filename

    today = date.today()
    
    inputs = []
    for graph in graph_names:
        graph_file = fname(graph)
        inputs.append([data_dir,
                       graph_file,
                       graph,
                       header,
                       sep,
                       k,
                       rwc_lst,
                       all_pol_measures,
                       alpha])
                
    outputs = process_map(parallel_pol_measures_with_communities, inputs, max_workers=cfg.max_workers)
    dfs = ut.process_output_pol_measures(outputs)
    dfs_concat = pd.concat(dfs)
    
    out_path = os.path.join(cfg.out_dir, f'exp_real_graphs__DIR{exp_name}__K{k}__{today}.csv')
    dfs_concat.to_csv(out_path, sep=',', header=True, index=False)
