from datetime import date
import networkx as nx
import numpy as np
import os
import pandas as pd
import sys

sys.path.append('./')
import config as cfg
import dk_models as dk
import utils as ut


if __name__ == '__main__':
    
    sizes = cfg.sizes
    skews = cfg.skews
    plen = cfg.plen
    k = cfg.k
    rwc_lst = cfg.scores
    num_samples = cfg.num_samples
    null_models = cfg.null_models
    pa_method = cfg.pa_method
    today = date.today()
    
    # we skip cliques, because there is
    # only one graph with that degree sequence
    # CYCLES
    all_dfs = []
    for cycle_size in sizes:
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
                    'Graph': gname,
                    'Null Model': 'Observed',
                    'Sample ID': None
                }
                outputs = dk.parallel_null_model([G, part, k, rwc_lst, args])
                all_dfs.extend(ut.process_output_pol_measures(outputs))
            
                for null_m in null_models:
                    outs = dk.get_output_nm(null_m, 
                                            G, 
                                            num_samples, 
                                            k, 
                                            rwc_lst, 
                                            pa_method, 
                                            cycle_size, 
                                            skew, 
                                            gname, 
                                            cfg.max_workers)
                    all_dfs.extend(ut.process_output_pol_measures(outs))
    final_df = pd.concat(all_dfs)
    out_path = os.path.join(cfg.out_dir, f'exp_dk_series_cycles__PA{pa_method}__K{k}_{today}.csv')
    final_df.to_csv(out_path, sep=',', header=True, index=False)

    # BARBELLS   
    all_dfs = []      
    skew = max(skews)
    graph_name = 'Node-partitioned Barbell'
    for size in sizes:
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
            'Graph': graph_name,
            'Null Model': 'Observed',
            'Sample ID': None
        }
        outputs = dk.parallel_null_model([b_g, pp, k, rwc_lst, args])
        all_dfs.extend(ut.process_output_pol_measures(outputs))
            
        for null_m in null_models:
            outs = dk.get_output_nm(null_m, 
                                    b_g, 
                                    num_samples, 
                                    k, 
                                    rwc_lst, 
                                    pa_method,
                                    b_g.number_of_nodes(), 
                                    .5,
                                    graph_name, 
                                    cfg.max_workers)
            all_dfs.extend(ut.process_output_pol_measures(outs))
        
        # exp3: different sizes, node partitioned
        pp = [nodes[:nred], nodes[nred:g_c.number_of_nodes()]]
        args = {
            'Size': g_c.number_of_nodes(),
            'Skew': skew,
            'Graph': graph_name,
            'Null Model': 'Observed',
            'Sample ID': None
        }
        outputs = dk.parallel_null_model([g_c, pp, k, rwc_lst, args])
        all_dfs.extend(ut.process_output_pol_measures(outputs))
        
        for null_m in null_models:
            outs = dk.get_output_nm(null_m, 
                                    g_c, 
                                    num_samples, 
                                    k, 
                                    rwc_lst,
                                    pa_method,
                                    g_c.number_of_nodes(), 
                                    skew,
                                    graph_name,
                                    cfg.max_workers)
            all_dfs.extend(ut.process_output_pol_measures(outs))
        print(f'Terminated Size: {size}')
    final_df = pd.concat(all_dfs)
    out_path = os.path.join(cfg.out_dir, f'exp_dk_series_barbells__PA{pa_method}__K{k}_{today}.csv')
    final_df.to_csv(out_path, sep=',', header=True, index=False)
