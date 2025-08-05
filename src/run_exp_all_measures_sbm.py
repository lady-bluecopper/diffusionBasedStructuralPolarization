from datetime import date
from graph_tool.generation import random_graph
from itertools import product # type: ignore
import numpy as np
import os
import pandas as pd
import sys
import time
from tqdm.contrib.concurrent import process_map

sys.path.append('./')
import config as cfg
import polarization_algorithms as pol
import utils as ut


def parallel_pol_measures_sbm(inp):
    
    wb = inp[0]
    bb = inp[1]
    seed = inp[2]
    N = inp[3]
    num_blocks = inp[4]
    k = inp[5]
    rwc_lst = inp[6]
    compute_pr = inp[7]
    all_pol_measures = inp[8]
    alpha = inp[9]
    
    
    def prob(a, b):
            if a == b:
                return wb
            else:
                return bb
        
    np.random.seed(seed)
    gg, bm = random_graph(N, lambda: np.random.poisson(num_blocks), 
                          directed=False,
                          model="blockmodel",
                          block_membership=lambda: np.random.randint(num_blocks),
                          edge_probs=prob) # type: ignore
    pp = [[], []]
    for idx, l in enumerate(bm.get_array()): # type: ignore
        bin_l = l % 2
        pp[bin_l].append(idx)
    
    file_name = f'PPR_SBM_N{N}_WB{wb}_BB{bb}_S{seed}_B{num_blocks}_A{alpha}'

    infopack = dict()
    if all_pol_measures:
        GG = ut.gt_to_nx_graph(gg)
        infopack = pol.compute_pol_measures(GG, pp, k, 
                                            rwc_lst=rwc_lst,
                                            alpha=alpha,
                                            compute_pr=compute_pr,
                                            save_pr=False,
                                            file_name=file_name)
    else:
        args = dict()
        args['g'] = gg
        args['p'] = pp
        args['k'] = k
        args['alpha'] = alpha
        args['compute_pr'] = compute_pr
        args['save_pr'] = False
        args['filename'] = file_name
        for score in rwc_lst:
            st = time.time()
            rwc = pol.get_rwc_score(score, args)
            end = time.time() - st
            infopack[f'DSP{score}__{alpha}'] = [rwc, end]

    infopack['WB'] = wb
    infopack['BB'] = bb
    infopack['Seed'] = seed
    infopack['N'] = N
    infopack['Num Blocks'] = num_blocks
    return infopack


if __name__ == '__main__':

    num_samples = cfg.num_samples
    n = cfg.n
    wbs = cfg.wbs
    bbs = cfg.bbs
    max_blocks = cfg.max_blocks
    k = cfg.k
    scores = cfg.scores
    alpha = cfg.alpha
    compute_pr = cfg.compute_pr
    all_pol_measures = cfg.all_pol_measures
    today = date.today()
    
    combs = product(np.arange(2, max_blocks + 1), 
                    np.arange(num_samples), 
                    wbs, bbs)
    
    inputs = []
    for combo in combs:
        num_blocks = combo[0]
        N = n * num_blocks
        seed = combo[1]
        wb = combo[2]
        bb = combo[3]
        
        if wb + bb == 0:
            continue
        
        inputs.append([wb,
                       bb,
                       seed,
                       N,
                       num_blocks,
                       k,
                       scores,
                       compute_pr,
                       all_pol_measures,
                       alpha])
        
    outputs = process_map(parallel_pol_measures_sbm, inputs, max_workers=cfg.max_workers)       
    dfs = ut.process_output_pol_measures(outputs)
    dfs_concat = pd.concat(dfs)
    dfs_concat['Num Samples'] = num_samples
    
    out_path = os.path.join(cfg.out_dir, f'exp_sbm__N{n}__S{num_samples}__{today}.csv')
    dfs_concat.to_csv(out_path, sep=',', header=True, index=False)
