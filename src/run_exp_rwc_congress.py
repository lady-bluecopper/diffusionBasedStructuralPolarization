from collections import defaultdict
from datetime import date
import graph_tool as gt
import os
import pandas as pd
import sys
import time
from tqdm.contrib.concurrent import process_map

sys.path.append('./')
import config as cfg
import polarization_algorithms as pol
import utils as ut


def parallel_rwc_fixed_colors(inp):
    
    graph_path = inp[0]
    label_path = inp[1]
    sep = inp[2]
    graph_name = inp[3]
    score = inp[4]
    alpha = inp[5]
    
    obs_edges_dict, v_map = ut.read_weighted_edges(graph_path,
                                              sep=sep,
                                              header=False,
                                              directed=False)
    edges = [(e[0], e[1], w) for e, w in obs_edges_dict.items()]
    # inner id -> outer id
    inv_v_map = {v_map[v]: v for v in v_map}
    v_labels = ut.read_labels(label_path, sep=sep, header=False)
    
    G = gt.Graph(edges, eprops=[('weight', 'int')])
    # GG = extract_largest_component(G)
    part_dict = defaultdict(list)
    for n in G.get_vertices():
        part_dict[v_labels[inv_v_map[n]]].append(n)
    partitions = list(part_dict.values())    

    args = dict()
    args['g'] = G
    args['p'] = partitions
    args['alpha'] = alpha
    args['compute_pr'] = True
    args['weighted_pr'] = True
    args['save_pr'] = False
    st = time.time()
    rwc = pol.get_rwc_score(score, args)
    end = time.time() - st
    
    infopack = {
        'Metric': f'DSP{score}',
        'Value': rwc,
        'Time (s)': end,
        'Dataset': graph_name
    }
    return infopack


if __name__ == '__main__':

    data_dir = cfg.data_dir
    congr_nums = cfg.congr_nums
    congr_types = cfg.congr_types
    congr_gtype = cfg.congr_gtype
    scores = cfg.scores
    alpha = cfg.alpha
    compute_pr = cfg.compute_pr
    sep = cfg.sep
    
    today = date.today()
    
    inputs = []
    for congr in congr_nums:
        for typ in congr_types:
            gname = f'{congr_gtype}_{congr}_{typ}.tsv'
            graph_path = os.path.join(data_dir, gname)
            fname = f'{congr_gtype}_communities.tsv'
            label_path = os.path.join(data_dir, fname)
            for score in scores:
                inputs.append([graph_path, label_path, sep, gname[:-4], score, alpha])
    print('Num Graphs', len(inputs))
                    
    outputs = process_map(parallel_rwc_fixed_colors, inputs, max_workers=cfg.max_workers)
    
    dfs = []
    for out in outputs:
        df = pd.DataFrame.from_dict(out, orient='index').T
        dfs.append(df)
    dfs_concat = pd.concat(dfs)
    types_lst = ','.join(congr_types)
    out_path = os.path.join(cfg.out_dir, f'exp_rwc_{congr_gtype}__{types_lst}__{today}.csv')
    dfs_concat.to_csv(out_path, sep=',', header=True, index=False)
