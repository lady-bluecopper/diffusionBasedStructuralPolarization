from datetime import date, datetime
from glob import glob
import os
import pandas as pd

import sys
sys.path.append('./')
import config as cfg
import dk_models as dk
import partition_algorithms as pa
import utils as ut


if __name__=="__main__":
    
    graph_names = cfg.graph_names
    data_dir = cfg.data_dir
    pa_method = cfg.pa_method
    header = cfg.header
    sep = cfg.sep
    extension = cfg.extension
    k = cfg.k
    rwc_lst = cfg.scores
    is_directed = cfg.directed
    num_samples = cfg.num_samples
    null_models = cfg.null_models
    today = date.today()
    
    results = []
    
    if len(graph_names) == 0:
        graph_names = glob(f'{data_dir}/*{extension}')
        prefix = ''
    else:
        prefix = data_dir
    
    for graph in graph_names:
        graph_path = os.path.join(prefix, graph)
        graph_name = os.path.basename(graph)
        outputs = []
        
        G_obs, vmap = ut.read_nx_graph(graph_path, header=header, sep=sep, directed=is_directed)
        print(graph_name, datetime.now())
        
        # REAL VALUES
        G, p = pa.get_partitioning(G_obs, method=pa_method)
        args = {
            'Size': G.number_of_nodes(),
            'Skew': None,
            'Graph': graph_name,
            'Null Model': 'Observed',
            'Sample ID': None
        }
        outputs.append(dk.parallel_null_model([G, p, k, rwc_lst, args]))
        for null_model in null_models:
            print(f'{null_model} - N={G.number_of_nodes()}, M={G.number_of_edges()}', datetime.now())
            outs = dk.get_output_nm(null_model, 
                                    G, 
                                    num_samples, 
                                    k, 
                                    rwc_lst,
                                    pa_method, 
                                    G.number_of_nodes(), 
                                    None,   # type: ignore
                                    graph_name, 
                                    cfg.max_workers)
            outputs.extend(outs)
        # Transform output into DataFrame
        results.extend(ut.process_output_pol_measures(outputs))
    
    dfs_concat = pd.concat(results)
    out_path = os.path.join(cfg.out_dir, f'exp_salloum__PA{pa_method}__K{k}__{today}.csv')
    dfs_concat.to_csv(out_path, sep=',', header=True, index=False)
