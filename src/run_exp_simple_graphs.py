from datetime import date
from itertools import combinations # type: ignore
import networkx as nx
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


def parallel_pol_measures_simple_graphs(inp):
    G = inp[0]
    part = inp[1]
    k = inp[2]
    scores = inp[3]
    all_pol_measures = inp[4]
    alpha = inp[5]
    args = inp[6]
    
    infopack = dict()
    if all_pol_measures:
        infopack = pol.compute_pol_measures(G, part, k, 
                                            rwc_lst=scores,
                                            alpha=alpha)
    else:
        g, pp, _ = ut.remapped_gt_graph(G, part)
        aargs = dict()
        aargs['g'] = g
        aargs['p'] = pp
        aargs['k'] = k
        aargs['alpha'] = alpha
        for score in scores:
            st = time.time()
            rwc = pol.get_rwc_score(score, args)
            end = time.time() - st
            infopack[f'DSP{score}__{alpha}'] = [rwc, end]
        
    infopack.update(args)
    return infopack


if __name__ == '__main__':
    
    sizes = cfg.sizes
    skews = cfg.skews
    plen = cfg.plen
    scores = cfg.scores
    all_pol_measures = cfg.all_pol_measures
    alpha = cfg.alpha
    k = cfg.k
    today = date.today()

    # CLIQUES
    all_dfs = []
    for cliq_size in sizes:
        inputs = []
        for skew in skews:
            num_red = int(np.ceil(cliq_size * skew))
            nodes = np.arange(cliq_size)
            part = [nodes[:num_red], nodes[num_red:]]
            
            G = nx.Graph(directed=False)
            edges = list(combinations(nodes, 2))
            G.add_edges_from(edges)
            
            args = {
                'Size': cliq_size,
                'Skew': skew,
                'Graph Type': 'Clique'
            }
            inputs.append([G, part, k, scores, all_pol_measures, alpha, args])
        outputs = process_map(parallel_pol_measures_simple_graphs, inputs, max_workers=cfg.max_workers)
        all_dfs.extend(ut.process_output_pol_measures(outputs))
    final_df = pd.concat(all_dfs)
    out_path = os.path.join(cfg.out_dir, f'exp_cliques_{today}.csv')
    final_df.to_csv(out_path, sep=',', header=True, index=False)
    
    # CYCLES
    all_dfs = []
    for cycle_size in sizes:
        inputs = []
        for alt in [True, False]:
            for skew in skews:
                if alt and skew != 0.5:
                    continue

                nodes = np.arange(cycle_size)
            
                if alt:
                    num_red = int(cycle_size * .5)
                    part = [[x for x in range(0, cycle_size, 2)], 
                            [x for x in range(1, cycle_size, 2)]]
                    gname = 'Alternating Cycle'
                else:
                    num_red = int(cycle_size * skew)
                    part = [nodes[:num_red], nodes[num_red:]]
                    gname = 'Half-split Cycle'
                
                G = nx.Graph(directed=False)
                edges = [(i, i+1) for i in range(cycle_size-1)]
                edges.append((0, cycle_size - 1))
                G.add_edges_from(edges)

                args = {
                    'Size': cycle_size,
                    'Skew': skew,
                    'Graph Type': gname
                }
                inputs.append([G, part, k, scores, all_pol_measures, alpha, args])
        outputs = process_map(parallel_pol_measures_simple_graphs, inputs, max_workers=cfg.max_workers)
        all_dfs.extend(ut.process_output_pol_measures(outputs))

    final_df = pd.concat(all_dfs)
    out_path = os.path.join(cfg.out_dir, f'exp_cycles_{today}.csv')
    final_df.to_csv(out_path, sep=',', header=True, index=False)
    
    # BARBELLS
    all_dfs = []
    skew = max(skews)
    for size in sizes:
        inputs = []
        nodes = np.arange(size)
        bell_size = int((size - plen) / 2)
        b_g = nx.barbell_graph(bell_size, m2=plen)
        g_c = b_g.copy()
        for i in range(int(bell_size * skew)):
            g_c.remove_node(size-1-i)
        nred = int(bell_size + plen / 2)

        # exp1: same size, node partitioned
        pp = [nodes[:nred], nodes[nred:]]
        args = {
            'Size': b_g.number_of_nodes(),
            'Skew': .5,
            'Graph Type': 'Node-partitioned Barbell'
        }
        inputs.append([b_g, pp, k, scores, all_pol_measures, alpha, args])
            
        # exp3: different sizes, node partitioned
        pp = [nodes[:nred], nodes[nred:g_c.number_of_nodes()]]
        args = {
            'Size': g_c.number_of_nodes(),
            'Skew': skew,
            'Graph Type': 'Node-partitioned Barbell'
        }
        inputs.append([g_c, pp, k, scores, all_pol_measures, alpha, args])

        outputs = process_map(parallel_pol_measures_simple_graphs, inputs, max_workers=cfg.max_workers)
        all_dfs.extend(ut.process_output_pol_measures(outputs))
    
    final_df = pd.concat(all_dfs)
    out_path = os.path.join(cfg.out_dir, f'exp_barbells_{today}.csv')
    final_df.to_csv(out_path, sep=',', header=True, index=False)
