from datetime import date
import numpy as np
import os
import pandas as pd
import sys
import time
from tqdm.contrib.concurrent import process_map

sys.path.append('./')
import config as cfg
import generators as gen
import polarization_algorithms as pol
import utils as ut


def parallel_pol_measures_gnml(inp):
    
    average_degree = inp[0]
    n = inp[1]
    idx = inp[2]
    skew = inp[3]
    directed = inp[4]
    k = inp[5]
    rwc_lst = inp[6]
    compute_pr = inp[7]
    all_pol_measures = inp[8]
    alpha = inp[9]
    
    np.random.seed(idx)
    
    GG, pp = gen.generate_gnml_graph_2_labels(n, average_degree, idx, directed, skew)
    
    file_name = f'PPR_GNML_N{n}_A{average_degree}_S{idx}_Sk{skew}_D{directed}_A0.85'
    
    infopack = dict()
    if all_pol_measures:
        infopack = pol.compute_pol_measures(GG, pp, k, 
                                            rwc_lst=rwc_lst,
                                            alpha=alpha,
                                            compute_pr=compute_pr,
                                            save_pr=False,
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

    infopack['Nodes'] = n
    infopack['Directed'] = directed
    infopack['AVG Deg'] = average_degree
    infopack['Skew'] = skew
    infopack['Seed'] = idx
    return infopack


if __name__ == '__main__':

    skews = cfg.skews
    avg_deg_list = cfg.average_degree_list
    num_samples = cfg.num_samples
    n = cfg.n
    direc = cfg.directed
    k = cfg.k
    rwc_lst = cfg.scores
    compute_pr = cfg.compute_pr
    all_pol_measures = cfg.all_pol_measures
    alpha = cfg.alpha
    today = date.today()
    
    inputs = []
    for avg_deg in avg_deg_list:
        for idx in range(num_samples):
            for skew in skews:
                inputs.append([avg_deg,
                               n,
                               idx,
                               skew,
                               direc,
                               k,
                               rwc_lst,
                               compute_pr,
                               all_pol_measures,
                               alpha])
                
    outputs = process_map(parallel_pol_measures_gnml, inputs, max_workers=cfg.max_workers)
    dfs = ut.process_output_pol_measures(outputs)
    dfs_concat = pd.concat(dfs)
    
    out_path = os.path.join(cfg.out_dir, f'exp_gnml__N{n}__D{direc}__S{num_samples}__{today}.csv')
    dfs_concat.to_csv(out_path, sep=',', header=True, index=False)
