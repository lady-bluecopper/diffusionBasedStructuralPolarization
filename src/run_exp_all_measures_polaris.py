from datetime import date, datetime
from glob import glob
import networkx as nx
import numpy as np
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
import polaris.src.ConfigModel_MCMC as mcmc 
import polaris.src.MCMC_LW as lw


def parallel_pol_measures_fixed_colors(inp):
    # Node colors are given in input
    
    graph = inp[0]
    prefix = inp[1]
    header = inp[2]
    sep = inp[3]
    is_directed = inp[4]
    partitions = inp[5]
    k = inp[6]
    rwc_lst = inp[7]
    graph_name = inp[8]
    graph_id = inp[9]
    compute_pr = inp[9]
    save_pr = inp[10]
    all_pol_measures = inp[11]
    alpha = inp[12]

    graph_path = os.path.join(prefix, graph)
    G_obs, vmap = ut.read_nx_gcc(graph_path, header=header, sep=sep, directed=is_directed)
    
    nodes = set(G_obs.nodes())
    pp = [[], []]
    node_colors = dict()
    for pid in [0, 1]:
        for out_n in partitions[pid]:
            in_n = vmap.get(str(out_n), -1)
            if in_n in nodes:
                pp[pid].append(in_n)
                node_colors[in_n] = pid
    nx.set_node_attributes(G_obs, values=node_colors, name='color')
    ass = nx.attribute_assortativity_coefficient(G_obs, "color")
    file_name = f'Graph={graph_name}_ID={graph_id}_D={is_directed}_A={alpha}'

    infopack = dict()
    if all_pol_measures:
        infopack = pol.compute_pol_measures(G_obs, pp, k, 
                                            rwc_lst=rwc_lst,
                                            alpha=alpha,
                                            compute_pr=compute_pr,
                                            save_pr=save_pr,
                                            file_name=file_name)
    else:
        gr, par, _ = ut.remapped_gt_graph(G_obs, pp)
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
    infopack['Sample ID'] = graph_id
    infopack['Num Nodes'] = G_obs.number_of_nodes()
    infopack['P0 Size'] = len(pp[0])
    infopack['P1 Size'] = len(pp[1])
    infopack['Orig. P0 Size'] = len(partitions[0])
    infopack['Orig. P1 Size'] = len(partitions[1])
    return infopack


def run_sampler(edges: list[tuple[int,int]],
                degrees: dict[int, int],
                node_labels: dict[int, int],
                num_graphs: int,
                swaps: int,
                algo: str, 
                out_dir: str,
                graph_name: str,
                max_workers: int=10,
                actual_swaps: bool=False,
                seed: int=0):
    '''
    INPUT
    ======
    edges (list): list of edges in the original graph.
    degrees (dict): degree of each node.
    node_labels (dict): label of each node.
    num_graphs (int): number of random graphs to generate.
    swaps (int): number of iterations to perform before returning the current state. 
    algo (str): name of the sampler to use to move in the state space.
    out_dir (str): where the graph should be stored.
    graph_name (str): name of the observed graph.
    max_workers (int): max number of concurrent threads.
    actual_swaps (bool): if True, an iteration is counted only if the
                         transition to the next state was accepted.
    seed (int): for reproducibility.
    '''
    sampler = lw.MCMC_LW(edges, degrees, node_labels)
    print('num edges', len(edges), 'num swaps', swaps, 'num graphs', num_graphs)
    mcmc.get_graph_parallel_chains(sampler, 
                                   out_dir,
                                   graph_name,
                                   algo,
                                   num_graphs, 
                                   swaps, 
                                   max_workers, 
                                   actual_swaps, 
                                   seed) 


if __name__ == '__main__':

    graph_names = cfg.graph_names
    data_dir = cfg.data_dir
    pa_method = cfg.pa_method
    header = cfg.header
    extension = cfg.extension
    k = cfg.k
    rwc_lst = cfg.scores
    compute_pr = cfg.compute_pr
    save_pr = cfg.save_pr
    exp_name = cfg.exp_name
    all_pol_measures = cfg.all_pol_measures
    alpha = cfg.alpha
    today = date.today()
    
    samples_dir = os.path.join(cfg.out_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    
    if len(graph_names) == 0:
        graph_names = glob(f'{data_dir}/*{extension}')
        prefix = ''
    else:
        prefix = data_dir
        
    dfs = []
    for graph in graph_names:
        graph_path = os.path.join(prefix, graph)
        G_obs, vmap = ut.read_nx_graph(graph_path, header=header)
        GG, pp = pa.get_partitioning(G_obs, method=pa_method)
        swaps = int(GG.number_of_edges()) * np.log(GG.number_of_edges())
        basename = os.path.basename(graph_path)
        files = glob(f'{samples_dir}/{basename}__sampler_LW__swaps_{swaps}__runtime_*__seed_*__actualswaps_False.tsv')

        if cfg.generate and len(files) < cfg.num_samples:
            degrees = ut.compute_degree_sequence_from_list(GG.edges())
            node_labels = dict()
            for pidx, part in enumerate(pp):
                for node in part:
                    node_labels[node] = pidx
            print('Start Generation', graph, datetime.now())
            run_sampler(edges=GG.edges(),
                        degrees=degrees,
                        node_labels=node_labels,
                        num_graphs=cfg.num_samples,
                        swaps=swaps,
                        algo='LW',
                        out_dir=samples_dir,
                        graph_name=basename)
        inputs = []
        for file in files:
            graph_name = os.path.basename(file).split('__')[0]
            graph_id = os.path.basename(file).split('__')[4].split('_')[1]
            inputs.append([file, 
                           '', # prefix 
                           False, # header 
                           '\t', # sep
                           False, # directed 
                           pp, 
                           k, 
                           rwc_lst, 
                           graph_name, 
                           graph_id, 
                           compute_pr, 
                           save_pr,
                           all_pol_measures,
                           alpha])
        print('Start Computation Polarization Measures', os.path.basename(graph_path), datetime.now())
        outputs = process_map(parallel_pol_measures_fixed_colors, inputs, max_workers=cfg.max_workers)
        print('Saving', len(outputs), 'Results') 
        dfs.extend(ut.process_output_pol_measures(outputs))
    print('Terminated', len(graph_names), 'Datasets')
    dfs_concat = pd.concat(dfs)
    dfs_concat['Partitioner'] = pa_method
    out_path = os.path.join(cfg.out_dir, f'exp_POLARIS__DIR{exp_name}__PA{pa_method}__K{k}__S{cfg.num_samples}__{today}.csv')
    dfs_concat.to_csv(out_path, sep=',', header=True, index=False)
