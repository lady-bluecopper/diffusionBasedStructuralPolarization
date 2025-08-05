import numpy as np
from collections import defaultdict
import math
from scipy import stats
from tqdm.contrib.concurrent import process_map


def check_degree_sequences(deg1, deg2):
    '''
    Checks if two degree sequences are equal.
    '''
    deg1_a = sorted(np.array(list(deg1.values())))
    deg2_a = sorted(np.array(list(deg2.values())))
    return np.array_equal(deg1_a, deg2_a)


def check_JLM(jlm1, jlm2):
    '''
    Checks if two JLMs are equal.
    '''
    for i in range(jlm1.shape[0]):
        for j in range(jlm2.shape[1]):
            if jlm1[i][j] != jlm2[i][j]:
                return False
    return True


def compute_JLM_from_list(edges: list[tuple[int, int]], 
                          node_labels: dict[int, int]):
    '''
    Given a list of edges,
    computes the JLM of the corresponding multigraph.
    '''
    num_labels = len(set(node_labels.values()))
    jlm = np.zeros((num_labels, num_labels), np.int32)
    for e in edges:
        l1 = node_labels[e[0]]
        l2 = node_labels[e[1]]
        jlm[l1][l2] += 1
        if l1 != l2:
            jlm[l2][l1] += 1
    return jlm


def compute_JLM_from_A(A: dict[tuple[int, int], int], 
                       node_labels: dict[int, int]):
    '''
    Given a dictionary of edge weights,
    computes the JLM of the corresponding multigraph.
    '''
    num_labels = len(set(node_labels.values()))
    jlm = np.zeros((num_labels, num_labels), np.int32)
    for e in A:
        if e[0] <= e[1]:
            l1 = node_labels[e[0]]
            l2 = node_labels[e[1]]
            if e[0] != e[1]:
                jlm[l1][l2] += A[e]
            if e[0] == e[1]:
                jlm[l1][l2] += (A[e] / 2)
            if l1 != l2:
                jlm[l2][l1] += A[e]
    return jlm


def compute_degree_sequence_from_A(A: dict[tuple[int, int], int]):
    '''
    Given a dictionary of edge weights,
    computes the degree of each node.
    '''
    degs = defaultdict(int)
    for e in A:
        if e[0] <= e[1]:
            degs[e[0]] += A[e]
            if e[0] != e[1]:
                degs[e[1]] += A[e]
    return degs


def compute_degree_sequence_from_list(edges: list[tuple[int, int]]):
    '''
    Given a list of edges, computes the degree
    of each node in the corresponding multigraph.
    '''
    degs = defaultdict(int)
    for e in edges:
        degs[e[0]] += 1
        degs[e[1]] += 1
    return degs


def compute_perturbation_score(A1, A2):
    '''
    Computes the Manhattan distance between A1 and A2.
    '''
    diffs = 0
    for k in A1:
        diffs += np.abs(A1[k] - A2.get(k, 0))
    for k in A2:
        if k not in A1:
            diffs += A2[k]
    return int(diffs)


def are_equals(M1, M2):
    '''
    Checks if two lil_matrices are equal.
    '''
    for r in range(M1.shape[0]):
        r1 = M1.getrow(r).toarray()[0]
        r2 = M2.getrow(r).toarray()[0]
        if not np.array_equal(r1, r2):
            return False
    return True


def dump_edge_list(file_path, obj):
    '''
    Save graph to disk.
    '''
    with open(file_path, 'w') as out_f:
        for e in obj:
            out_f.write(f'{e[0]}\t{e[1]}\n')
            
            
def copy_edge_list(edge_list):
    '''
    Creates a deep copy of edge_list.
    '''
    return [(u, v) for u, v in edge_list]


def copy_weight_dict(W: dict[tuple[int, int], int]):
    '''
    Creates a deep copy of W.
    '''
    # edge -> edge count
    W_prime = dict()
    for e, c in W.items():
        W_prime[(e[0], e[1])] = c
    return W_prime


def copy_label_dicts(lab_match_edges: dict[int, np.ndarray],
                     lab_match_emap: dict[int, dict[tuple[int, int], int]],
                     lab_match_m: dict[int, int]):
    '''
    Creates a deep copy of lab_match_edges, 
    lab_match_emap, and lab_match_m.
    '''
    new_lab_match_edges = dict()
    new_lab_match_emap = dict()
    new_lab_match_m = dict()
    
    for k, ar in lab_match_edges.items():
        new_lab_match_edges[k] = np.array([(x[0], x[1]) for x in ar])
    for k, dt in lab_match_emap.items():
        new_dt = dict()
        for tup, v in dt.items():
            new_dt[(tup[0], tup[1])] = v
        new_lab_match_emap[k] = new_dt
    for k, v in lab_match_m.items():
        new_lab_match_m[k] = v
    return new_lab_match_edges, new_lab_match_emap, new_lab_match_m


# def copy_edge_skip_lists(edge_sl_dict):
#     '''
#     Creates a deep copy of edge_sl_dict.
#     '''
#     copy_e_sl_dict = dict()
#     for k in edge_sl_dict:
#         copy_e_sl_dict[k] = SkipList(object)
#         for i in range(edge_sl_dict[k].size()):
#             u, v = edge_sl_dict[k].at(i)
#             copy_e_sl_dict[k].insert((u, v))
#     return copy_e_sl_dict


def convert_edgelist_to_dictionary(edgeList, A):
    '''
    Given a list of edges, populate the dictionary of
    corresponding edge weights.
    '''
    for edge in edgeList:
        A[edge[0], edge[1]] = A.get((edge[0], edge[1]), 0) + 1
        A[edge[1], edge[0]] = A.get((edge[1], edge[0]), 0) + 1
    

def check_autocorrelation_lag1(inp):
    '''
    series (list): list of degree assortativity values
    alpha (float): significance level
    Check lag-1 autocorrelation of series at significance level alpha.
    It corresponds to Algorithm 2 Dutta et al.
    '''
    series = inp[0]
    alpha = inp[1]
    n = len(series)
    data = np.asarray(series, dtype=np.float64)
    xbar = np.mean(data)
    c0 = np.sum((data - xbar) ** 2)
    
    def standard_autocorrelations(h):
        # line 2 Algorithm 2 Dutta et al.
        corr = ((data[: n - h] - xbar) * (data[h:] - xbar)).sum() / c0
        # Eq. 4 Dutta et al.
        mean = -(n - h) / (n * (n - 1))
        # numerator Eq. 5 Dutta et al.
        var = n**4 - (h + 3) * n**3 + 3 * h * n**2
        var += 2 * h * (h + 1) * n - 4 * h**2
        # denominator Eq. 5 Dutta et al.
        var /= ((n + 1) * n**2 * (n - 1)**2)
        # sigma
        SE = math.sqrt(var)
        # line 5 Algorithm 2 Dutta et al.
        standard_corr = (corr - mean) / SE
        return standard_corr

    # h = lag = 1
    y = standard_autocorrelations(1)
    # One-sided test
    z_critical = stats.norm.ppf(1 - alpha)
    # print('y', y, 'z_critical', z_critical)
    if y > z_critical:
        return 1
    return 0


def get_num_sig_autocorrelations(r_datapoints, alpha):
    sig = 0
    for data in r_datapoints:
        sig += check_autocorrelation_lag1([data, alpha])
    return sig


def get_num_sig_autocorrelations_parallel(T: int,
                                          r_datapoints: list[list[float]],
                                          gap: int,
                                          increment: int,
                                          alpha: float,
                                          max_workers: int=4) -> int:
    '''
    T (int): number of independent draws from each Markov chain.
    
    r_datapoints (array of size D x A): for each Markov chain, degree assortativity values 
                                        stored every increment steps. A is a multiple of increment.
    gap (int): current candidate sampling gap.
    increment (int): the candidate sampling gap is incremented by increment until the test for 
                     significant lag-1 autocorrelation succeeds.
    alpha (float): significance level.
    max_workers (int): maximum number of parallel computations.
    '''
    inputs = []
    step = gap // increment
    print('gap', gap, 'increment', increment)
    for data in r_datapoints:
        S_T = [data[i * step] for i in range(T)]
        inputs.append([S_T, alpha])
    output = process_map(check_autocorrelation_lag1, inputs, max_workers=max_workers)
    sig = np.sum(output)
    return sig
