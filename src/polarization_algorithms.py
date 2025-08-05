from collections import defaultdict
from copy import deepcopy
import graph_tool as gt
from graph_tool.centrality import pagerank as pr
import heapq
import networkx as nx
import networkx.algorithms.community as nx_comm
import numpy as np
from operator import itemgetter
import scipy.stats
import time
from typing import List
import sys
sys.path.append('./')
import utils as ut 


def get_influencer_nodes(G, left_nodes, right_nodes, n_influencers):
    """Returns the k-highest degree nodes for each partition"""

    if G.is_directed():
        left_degrees = G.in_degree(left_nodes)
        right_degrees = G.in_degree(right_nodes)
    else:
        left_degrees = G.degree(left_nodes)
        right_degrees = G.degree(right_nodes)

    if n_influencers < 1:
        k_left = max(1, int(n_influencers * len(left_nodes)))
        k_right = max(1, int(n_influencers * len(right_nodes)))
    else:
        k_left = min(n_influencers, len(left_nodes))
        k_right = min(n_influencers, len(right_nodes))

    left_influencers = heapq.nlargest(k_left, left_degrees, key=lambda x: x[1])
    right_influencers = heapq.nlargest(k_right, right_degrees, key=lambda x: x[1])
    return [node for node, _ in left_influencers], [node for node, _ in right_influencers]


def krackhardt_ratio_pol(G, partition):
    """Computes EI-Index Polarization"""
    EL = 0
    IL = 0

    left_nodes = set(partition[0])
    right_nodes = set(partition[1])

    for e in G.edges():
        s, t = e
        if (s in left_nodes and t in right_nodes) or (s in right_nodes and t in left_nodes):
            EL += 1
        elif (s in left_nodes and t in left_nodes) or (s in right_nodes and t in right_nodes):
            IL += 1
    return (EL - IL) / (EL + IL)


def extended_krackhardt_ratio_pol(G, partition):
    """Computes Extended EI-Index Polarization"""

    block_a = set(partition[0])
    block_b = set(partition[1])

    n_a = len(block_a)
    n_b = len(block_b)

    c_a = sum(1 for s, t in G.edges(block_a) if (t in block_a and s in block_a))
    c_b = sum(1 for s, t in G.edges(block_b) if (t in block_b and s in block_b))
    c_ab = sum(1 for s, t in G.edges() if (s in block_a and t in block_b) or (s in block_b and t in block_a))

    B_aa = c_a / (n_a * (n_a - 1) * 0.5) if n_a > 1 else 0
    B_bb = c_b / (n_b * (n_b - 1) * 0.5) if n_b > 1 else 0
    B_ab = c_ab / (n_a * n_b) if (n_a > 1 and n_b > 1) else 0
    B_ba = B_ab
    return (B_aa + B_bb - B_ab - B_ba) / (B_aa + B_bb + B_ab + B_ba)


def betweenness_pol(G, partition, iterations=10):
    """Computes Betweenness Centrality Controversy Polarization"""
    dict_eb = nx.edge_betweenness_centrality(G, k=int(0.75 * len(G)))

    left_nodes = set(partition[0])
    right_nodes = set(partition[1])

    cut_edges = []
    rest_edges = []

    for e in G.edges():
        s, t = e
        if (s in left_nodes and t in right_nodes) or (s in right_nodes and t in left_nodes):
            cut_edges += [e]
        elif (s in left_nodes and t in left_nodes) or (s in right_nodes and t in right_nodes):
            rest_edges += [e]

    if len(cut_edges) <= 1 or len(rest_edges) <= 1:
        # print("Not enough cut edges to compute the polarization.")
        return None

    cut_ebc = [dict_eb[e] for e in cut_edges]
    rest_ebc = [dict_eb[e] for e in rest_edges]

    if len(cut_ebc) <= 1:
        print("Error in the gap!")
        return None

    try:
        kernel_for_cut = scipy.stats.gaussian_kde(cut_ebc, "silverman")
        kernel_for_rest = scipy.stats.gaussian_kde(rest_ebc, "silverman")
    except:
        return None

    BCC = []

    for _ in range(iterations):
        cut_dist = kernel_for_cut.resample(10000)[0]
        rest_dist = kernel_for_rest.resample(10000)[0]

        cut_dist = [max(0.00001, value) for value in cut_dist]
        rest_dist = [max(0.00001, value) for value in rest_dist]

        kl_divergence = scipy.stats.entropy(rest_dist, cut_dist)

        BCCval = 1 - np.exp(-kl_divergence)
        BCC.append(BCCval)

    return sum(BCC) / len(BCC)


def gmck_pol(G, partition):
    """Computes Boundary Polarization"""
    X = set(partition[0])
    Y = set(partition[1])

    B = []

    for e in G.edges():
        s, t = e
        if (s in X and t in Y) or (s in Y and t in X):
            B.extend([s, t])

    Bset = set(B)
    Bx = Bset & X
    By = Bset & Y

    Ix = X.difference(Bx)
    Iy = Y.difference(By)
    I = Ix.union(Iy)

    Bxf = Bx.copy()
    Byf = By.copy()

    for u in Bxf:
        connections_of_u = set([n for n in G.neighbors(u)])
        if connections_of_u.issubset(B):
            Bx.remove(u)

    for v in Byf:
        connections_of_v = set([n for n in G.neighbors(v)])
        if connections_of_v.issubset(B):
            By.remove(v)

    B = Bx.union(By)

    summand = []
    for node in B:

        di = nx.cut_size(G, [node], I)

        if node in Bx:
            db = nx.cut_size(G, [node], Byf)
        else:
            db = nx.cut_size(G, [node], Bxf)

        summand.append((di / (di + db)) - 0.5)

    GMCK = (1 / (len(B) + 0.0001)) * sum(summand)
    return GMCK


def dipole_pol(G, partition, k=0.05):
    
    left_nodes = partition[0]
    right_nodes = partition[1]
    
    X_top, Y_top = get_influencer_nodes(G, left_nodes, right_nodes, k)
    X_top = set(X_top)
    Y_top = set(Y_top)

    dict_polarity = dict.fromkeys(list(G), 0)
    dict_polarity.update(zip(X_top, [-1] * len(X_top)))
    dict_polarity.update(zip(Y_top, [1] * len(Y_top)))

    listeners = set(G.nodes) - X_top - Y_top
    
    polarity = np.asarray(list(dict_polarity.values()))
    polarity_new = np.zeros(len(polarity))

    roundcount = 0
    tol = 10**-5
    notconverged = len(polarity_new)
    max_rounds = 500

    while notconverged > 0 :

        for node in listeners:
            polarity_new[node] = np.mean([polarity[n] for n in G.neighbors(node)])

        if roundcount < 1:
            polarity_new[list(X_top)] = -1
            polarity_new[list(Y_top)] = 1

        diff = np.abs(polarity - polarity_new)
        notconverged = len(diff[diff>tol])

        polarity = deepcopy(polarity_new)

        if roundcount > max_rounds:
            break
        roundcount += 1
    
    n_nodes = G.number_of_nodes()
    n_plus = len(polarity[polarity > 0]) 
    n_minus = n_nodes - n_plus

    delta_A = np.abs((n_plus - n_minus) * (1/(n_nodes)))

    gc_plus = np.mean(polarity[polarity > 0])
    gc_minus = np.mean(polarity[polarity < 0])

    pole_D = np.abs(gc_plus - gc_minus) * .5
    mblb_score = (1 - delta_A) * pole_D
    
    return mblb_score
    

def rwc_score(g, 
              partition, 
              k:float=10., 
              alpha=0.85,
              max_iter=100000,
              fair=False,
              include_infl=True,
              verbose=False):

    part_left = partition[0]
    part_right = partition[1]
    # retrieve left and right influencers
    if k < 1:
        k_left = max(1, int(k * len(part_left)))
        k_right = max(1, int(k * len(part_right)))
    else:
        k_left = int(k)
        k_right = int(k)
    # node id, degree of the top-k_left nodes in the left partition
    top_nodes_left = getNodesFromPartitionWithHighestDegree(g, k_left,
                                                            part_left)
    # node id, degree of the top-k_right nodes in the right partition
    top_nodes_right = getNodesFromPartitionWithHighestDegree(g, k_right,
                                                             part_right)
    if include_infl:
        # each node in the left partition has the same non-zero probability; others have almost 0
        uniform_left = getUniformDistribution(g, part_left, part_right)
        # each node in the right partition has the same non-zero probability; others have almost 0
        uniform_right = getUniformDistribution(g, part_right, part_left)
    else:
        real_top_left = [k for k, v in top_nodes_left]
        nodes_left = set(part_left)
        nodes_left = nodes_left.difference(real_top_left)
        nodes_right = set(part_right)
        nodes_right = nodes_right.union(real_top_left)
        # each non-influencer node in the left partition has the same non-zero probability; others have almost 0
        uniform_left = getUniformDistribution(g, nodes_left, nodes_right)

        real_top_right = [k for k, v in top_nodes_right]
        nodes_left = set(part_left)
        nodes_left = nodes_left.union(real_top_right)
        nodes_right = set(part_right)
        nodes_right = nodes_right.difference(real_top_right)
        # each non-influencer node in the right partition has the same non-zero probability; others have almost 0
        uniform_right = getUniformDistribution(g, nodes_right, nodes_left)
    
    pers = g.new_vertex_property("float")
    pers.a = uniform_left
    pagerank_left = np.array(pr(g, damping=alpha, pers=pers, max_iter=max_iter).fa) # type: ignore
    pers = g.new_vertex_property("float")
    pers.a = uniform_right
    pagerank_right = np.array(pr(g, damping=alpha, pers=pers, max_iter=max_iter).fa) # type: ignore

    start_left_end_left = sum([pagerank_left[k] for k, _ in top_nodes_left])
    start_left_end_right = sum([pagerank_left[k] for k, _ in top_nodes_right])
    start_right_end_left = sum([pagerank_right[k] for k, _ in top_nodes_left])
    start_right_end_right = sum([pagerank_right[k] for k, _ in top_nodes_right])

    if fair:
        left_b = 1.
        right_b = 1.
    else:
        left_b = float(len(part_left)) / g.num_vertices()
        right_b = float(len(part_right)) / g.num_vertices()

    den = (start_left_end_left * left_b) + (start_right_end_left * right_b)
    p_start_left_end_left = 0
    if den != 0:
        p_start_left_end_left = (start_left_end_left * left_b) / den
    den = (start_right_end_right * right_b) + (start_left_end_right * left_b)
    p_start_left_end_right = 0
    if den != 0:
        p_start_left_end_right = (start_left_end_right * left_b) / den
    den = (start_right_end_right * right_b) + (start_left_end_right * left_b)
    p_start_right_end_right = 0
    if den != 0:
        p_start_right_end_right = (start_right_end_right * right_b) / den
    den = (start_left_end_left * left_b) + (start_right_end_left * right_b)
    p_start_right_end_left = 0
    if den != 0:
        p_start_right_end_left = (start_right_end_left * right_b) / den

    rwc_score = p_start_left_end_left * p_start_right_end_right - p_start_left_end_right * p_start_right_end_left
    if verbose:
        print('fair', fair, 'include_infl', include_infl)
        print("left  -> left ", p_start_left_end_left)
        print("right -> left ", p_start_right_end_left)
        print("left  -> right", p_start_left_end_right)
        print("right -> right", p_start_right_end_right)
        print('part_left', len(part_left))
        print('part_right', len(part_right))
        print('left_b', left_b)
        print('right_b', right_b)
        print('rwc_score', rwc_score)
    return rwc_score


# returns a dict with uniform distribution to that
# particular side and close-to-zero to the other side
# for ergodicity
def getUniformDistribution(g, part_start, part_end, epsilon=1e-12):
    uniform = {k: 1.0 / len(part_start) for k in part_start}
    uniform.update({k: epsilon for k in part_end})
    return [uniform.get(v, 0) for v in g.get_vertices()]


def getNodesFromPartitionWithHighestDegree(g: gt.Graph, 
                                           k: int, 
                                           part: list[int]):
    # top-k largest nodes by in-degree in the given partition
    if g.is_directed():
        degrees = g.get_in_degrees(part)
    else:
        degrees = g.get_total_degrees(part)
    node_degrees = [(part[i], degrees[i]) for i in range(len(part))]
    return heapq.nlargest(k, node_degrees, key=itemgetter(1))


def compute_ppr(g: gt.Graph, 
                vertices: List[int],  # starting vertices for PPR 
                alpha: float,  # damping factor
                max_iter: int,  # iterations for PPR
                normalized: bool,  # normalized PPR values
                weighted_pr: bool,  # weighted PPR
                run_pr: bool,  # whether we need to compute the PPR values or load them
                save_pr: bool,  # whether we need to save the PPR values
                file_name: str):  # filename to store the PPR values
    # stationary distribution for each node
    # (RWR from each node)
    pi = dict()
    if run_pr:
        V = g.get_vertices()
        # initialize personalized vector
        eps = 1e-12
        pers = g.new_vertex_property("float")
        pers.a = [eps for _ in V]
        # get edge weights
        ew = None
        if weighted_pr:
            ew = g.edge_properties['weight']

        prev_node = vertices[0]
        for v in vertices:
            pers.a[prev_node] = eps
            pers.a[v] = 1.
            PR = np.array(pr(g, damping=alpha, pers=pers, weight=ew, max_iter=max_iter).fa) # type: ignore
            PR[v] = 0
            pi[v] = PR
            if normalized:
                pi[v] /= np.sum(PR)
                assert np.sum(pi[v]) > 0.99
            prev_node = v
        # save pagerank
        if save_pr:
            with open(file_name, 'w') as out_f:
                for v in pi:
                    pi_v = ','.join(str(pi[v][u]) for u in range(len(pi[v])))
                    out_f.write(f'{v}\t{pi_v}\n')
    else:
        with open(file_name) as in_f:
            for line in in_f:
                lst = line.strip().split('\t')
                v = int(lst[0])
                if v not in pi:
                    pi[v] = dict()
                pi_v_lst = lst[1].strip().split(',')
                for u, pi_v_u in enumerate(pi_v_lst):
                    pi[v][u] = pi_v_u 
    return pi


def rwc_plus_v9(g,  # graph
                partition,  # vector with the two partitions
                alpha=0.85,  # dumping factor
                max_iter=100000,  # max num of iterations
                verbose=False,
                weighted_pr=False,
                compute_pr=True,
                save_pr=True,
                file_name='',
                sample_size=-1,
                seed=42):

    pleft = partition[0]
    pright = partition[1]
    vset = list(pleft)
    vset.extend(pright)
    
    if sample_size > 0:
        np.random.seed(seed)
        v_to_sample = int(len(vset) * sample_size)
        sample = np.random.choice(vset, v_to_sample)
        pleftset = set(pleft)
        pleft = []
        pright = []
        for v in sample:
            if v in pleftset:
                pleft.append(v)
            else:
                pright.append(v)
        vset = list(sample)
        print('num nodes considered:', len(vset), 'P0', len(pleft), 'P1', len(pright))
        print(f'STARTING: seed={seed}, sample_size={sample_size}, time={time.time()}')
    
    lsize = len(pleft)
    rsize = len(pright)
    if verbose:
        print(f'|L|={lsize}, |R|={rsize}, alpha={alpha}')
    # stationary distribution for each node
    # (RWR from each node)
    pi = compute_ppr(g=g, 
                     alpha=alpha,
                     vertices=vset,
                     max_iter=max_iter, 
                     normalized=True, 
                     weighted_pr=weighted_pr, 
                     run_pr=compute_pr, 
                     save_pr=save_pr, 
                     file_name=file_name)
    if verbose:
        print(f'PPR COMPLETED: seed={seed}, sample_size={sample_size}, time={time.time()}')
    E_LL = 0
    E_RL = 0
    for j in pleft:
        den = sum(pi[i][j] for i in vset)
        E_LL += (sum(pi[i][j] for i in pleft) / den)
        E_RL += (sum(pi[i][j] for i in pright) / den)
    E_LL *= (rsize / (lsize + rsize - 1))
    if lsize > 0:
        E_LL /= (2 * lsize)
    E_RL *= ((lsize - 1) / (lsize + rsize - 1))
    if lsize > 0:
        E_RL /= (2 * lsize)
    if verbose:
        print('E_LL', E_LL, 'E_RL', E_RL)
    E_RR = 0
    E_LR = 0
    for j in pright:
        den = sum(pi[i][j] for i in vset)
        E_RR += (sum(pi[i][j] for i in pright) / den)
        E_LR += (sum(pi[i][j] for i in pleft) / den)
    E_RR *= (lsize / (lsize + rsize - 1))
    if rsize > 0:
        E_RR /= (2 * rsize)
    E_LR *= ((rsize - 1) / (lsize + rsize - 1))
    if rsize > 0:
        E_LR /= (2 * rsize)
    if verbose:
        print('E_RR', E_RR, 'E_LR', E_LR)
    rwc = (E_LL - E_RL) + (E_RR - E_LR)
    if verbose:
        print('RWC=', rwc)
    if verbose:
        print(f'Measure Completed for seed={seed}, sample_size={sample_size}, time={time.time()}')
    return rwc


def rwc_plus_v9_mc(g,  # graph
                   partitions,  # vector with the partitions
                   alpha=0.85,  # dumping factor
                   max_iter=100000,  # max num of iterations
                   verbose=False,
                   weighted_pr=False,
                   compute_pr=True,
                   save_pr=True,
                   file_name='',
                   sample_size=-1,
                   seed=42):

    vset = list()
    for p in partitions:
        vset.extend(p)
    
    if sample_size > 0:
        v_to_sample = int(len(vset) * sample_size)
        np.random.seed(seed)
        sample = np.random.choice(vset, v_to_sample)
        sample_set = set(sample)
        new_partitions = defaultdict(list)
        for pid, lst in enumerate(partitions):
            for v in lst:
                if v in sample_set:
                    new_partitions[pid].append(v)
        for pid in new_partitions:
            partitions[pid] = new_partitions[pid]        
        vset = list(sample)
    
    n = len(vset)
    num_c = len(partitions)
    sizes = [len(p) for p in partitions]
    # stationary distribution for each node
    # (RWR from each node)
    pi = compute_ppr(g=g, 
                     alpha=alpha,
                     vertices=vset,
                     max_iter=max_iter, 
                     normalized=True, 
                     weighted_pr=weighted_pr, 
                     run_pr=compute_pr, 
                     save_pr=save_pr, 
                     file_name=file_name)
    RWC_W = 0
    RWC_I = 0
    for c, part in enumerate(partitions):
        multiplier = 1 / sizes[c]
        sum_c = 0
        sum_c_bar = 0
        for y in part:
            den = sum(pi[i][y] for i in vset)
            for q, part2 in enumerate(partitions):
                if q == c:
                    sum_c += ((sum(pi[i][y] for i in part2) / den) * (n - sizes[c]))
                else:
                    sum_c_bar += ((sum(pi[i][y] for i in part2) / den) * (n - sizes[q] - 1))
        sum_c *= multiplier
        sum_c /= ((num_c - 1) * (n - 1))
        sum_c_bar *= multiplier
        sum_c_bar /= ((num_c - 1) * (n - 1))
        RWC_W += sum_c
        RWC_I += sum_c_bar
    if verbose:
        print('RWC_W', RWC_W, 'RWC_I', RWC_I)
    rwc = 1 / num_c * (RWC_W - RWC_I)
    if verbose:
        print('RWC=', rwc)
    return rwc


def get_rwc_score(name, args):
    g = args['g']
    p = args['p']
    k = args.get('k', .1)
    alpha = args.get('alpha', 0.85)  # dumping factor
    verbose = args.get('verbose', False)
    compute_pr = args.get('compute_pr', True)
    weighted_pr = args.get('weighted_pr', False)
    save_pr = args.get('save_pr', False)
    sample_size = args.get('sample_size', -1)
    seed = args.get('seed', 42)
    file_name = args.get('filename', 'ppr.tsv')
    
    if name == 'rwc':
        return rwc_score(g, p, k=k, alpha=alpha, verbose=verbose)

    score_map = {
        'v9': rwc_plus_v9,
        'v9_mc': rwc_plus_v9_mc
    }
    return score_map[name](g, p,
                           alpha=alpha,
                           verbose=verbose,
                           weighted_pr=weighted_pr,
                           compute_pr=compute_pr,
                           save_pr=save_pr,
                           sample_size=sample_size,
                           seed=seed,
                           file_name=file_name)


def compute_pol_measures(G: nx.Graph,
                         partition: List,
                         k=10,
                         rwc_lst: List[str]=[],
                         alpha: float=0.85,
                         verbose: bool=False,
                         compute_pr: bool=True,
                         save_pr: bool=False,
                         file_name: str='ppr.tsv'):
    
    g, pp, _ = ut.remapped_gt_graph(G, partition)

    infopack = dict()
    
    empty_part = False
    for p in pp:
        if len(p) == 0:
            empty_part = True
            break
    if empty_part:
        for score in rwc_lst:
            infopack[f'DSP{score}__{alpha}'] = [None, None]
        for meas in ['RWC', 'ARWC', 'ARWC No-Infl', 
                     'Q', 'EI', 'AEI', 'BCC', 'BP', 'DM']:
            infopack[meas] = [None, None]
        return infopack
    args = dict()
    args['g'] = g
    args['p'] = pp
    args['alpha'] = alpha
    args['compute_pr'] = compute_pr
    args['save_pr'] = save_pr
    args['filename'] = file_name
    
    st = time.time()
    k_ = k
    if k < 1:
        k_ = k * 100
    rwc = rwc_score(g, pp, include_infl=True, k=k_)
    rwc_t = time.time() - st
    if verbose:
        print("RWC completed.")
    infopack['RWC'] = [rwc, rwc_t]
    
    st = time.time()
    k_ = k
    if k < 1:
        k_ = k * 100
    no_infl_rwc = rwc_score(g, pp, k=k_, include_infl=False, verbose=verbose)
    no_infl_rwc_t = time.time() - st
    if verbose:
        print('RWC NO-INFL completed.')
    infopack['RWC No-Infl'] = [no_infl_rwc, no_infl_rwc_t]

    st = time.time()
    k_ = k
    if k > 1:
        k_ = k / 100
    arwc = rwc_score(g, pp, include_infl=True, k=k_)
    arwc_t = time.time() - st
    if verbose:
        print("ARWC completed.")
    infopack['ARWC'] = [arwc, arwc_t]

    st = time.time()
    k_ = k
    if k > 1:
        k_ = k / 100
    no_infl_arwc = rwc_score(g, pp, k=k_, include_infl=False, verbose=verbose)
    no_infl_arwc_t = time.time() - st
    if verbose:
        print('ARWC NO-INFL completed.')
    infopack['ARWC No-Infl'] = [no_infl_arwc, no_infl_arwc_t]

    st = time.time()
    mod = nx_comm.modularity(G, partition)
    mod_t = time.time() - st
    if verbose:
        print("Modularity completed.")
    infopack['Q'] = [mod, mod_t]

    st = time.time()
    ei = -1 * krackhardt_ratio_pol(G, partition)
    ei_t = time.time() - st
    if verbose:
        print("EI completed.")
    infopack['EI'] = [ei, ei_t]

    st = time.time()
    extei = extended_krackhardt_ratio_pol(G, partition)
    extei_t = time.time() - st
    if verbose:
        print("AEI completed.")
    infopack['AEI'] = [extei, extei_t]

    st = time.time()
    ebc = betweenness_pol(G, partition)
    ebc_t = time.time() - st
    if verbose:
        print("BCC completed.")
    infopack['BCC'] = [ebc, ebc_t]

    st = time.time()
    gmck = gmck_pol(G, partition)
    gmck_t = time.time() - st
    if verbose:
        print("BP completed.")
    infopack['BP'] = [gmck, gmck_t]

    st = time.time()
    k_ = k
    if k > 1:
        k_ = k / 100
    mblb = dipole_pol(G, partition, k=k_)
    mblb_t = time.time() - st
    if verbose:
        print("DM completed.")
    infopack['DM'] = [mblb, mblb_t]
    
    for score in rwc_lst:
        st = time.time()
        rwc_plus = get_rwc_score(score, args)
        rwc_plus_t = time.time() - st
        infopack[f'DSP{score}__{alpha}'] = [rwc_plus, rwc_plus_t]
        if verbose:
            print('DSP completed.')

    return infopack
