import argparse

def read_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=0, help='Seed to initialize the pseudo-random number generation.')
    parser.add_argument('--num_samples', type=int, default=33, help='Number of random graphs to generate.')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of concurrent workers.')
    parser.add_argument('--base_path', type=str, default='.', help='Directory with the data and out directory.')
    parser.add_argument('--data_dir', type=str, default='data', help='Name of the directory with the datasets.')
    parser.add_argument('--graph_name', type=str, help='Name of the graph to load (without extension).')
    parser.add_argument('--algorithm', type=str, default='LA', choices=['LA', 'LW', 'CM'], help='Sampler name.')
    parser.add_argument('--D', type=int, default=10, help='Number of parallel chain to test convergence.')
    parser.add_argument('--mul_fact', default=100, type=int, help='Multiplying factor M to get the number of steps: M * num_edges.')
    parser.add_argument('--num_swaps', type=int, default=-1, help='Number of steps before returning the current state.')
    parser.add_argument('--actual_swaps', type=str, default='False', choices=['True', 'False'], help='If the number of steps to perform indicates the number of actual moves in the Markov chain.')
    parser.add_argument('--label_list', type=str, help='List of number of labels for the scalability experiment. Comma separated list.')
    parser.add_argument('--perc', type=float, default=0.05, help='Multiplying factor P to get the interval between two consecutive measurements of time elapsed: P * num_edges.')

    args = parser.parse_args()
    args = vars(args)

    return args


def read_tsv_graph(file_path):
    '''
    Read multi-graph stored as a list of edges,
    where source and destination are tab-separated.
    '''
    edges = []
    with open(file_path) as in_f:
        for line in in_f.readlines():
            lst = line.split('\t')
            edges.append((int(lst[0]), int(lst[1])))
    return edges


def read_node_labels(filepath, nodes):
    '''
    Read node labels from disk.
    Each line contains a node id and its label, separated by a tab.
    Node labels are replaced with integers starting from 0.
    '''
    inner_outer_labels = dict()
    node_labels = dict()
    lab_id = 0
    with open(filepath) as in_f:
        for line in in_f.readlines():
            lst = line.strip().split('\t')
            inn_lab = lst[1].strip()
            if inn_lab not in inner_outer_labels:
                inner_outer_labels[inn_lab] = lab_id
                lab_id += 1
            node_labels[int(lst[0])] = inner_outer_labels[inn_lab]
    for node in nodes:
        if node not in node_labels:
            node_labels[node] = lab_id
    return node_labels, inner_outer_labels