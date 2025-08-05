from collections import defaultdict
import random
import sys
sys.path.insert(1,'../')
from . import utils as ut # type: ignore


def get_compatible_edges_per_label(edges, node_labels: dict[int, int]):
    '''
    For each combination of node labels, the 
    list of compatible edges.
    
    OUTPUT
    ======
    lab_match_dict (dict): for each label, the list of ids of the 
                           compatible edges.
    lab_match_m (dict): for each label, the number of compatible edges. 
    '''
    lab_match_eids = defaultdict(list)
    lab_match_m = dict()
    for i, e in enumerate(edges):
        u, v = e[0], e[1]
        lab_match_eids[node_labels[u]].append(i)
        if node_labels[v] != node_labels[u]:
            lab_match_eids[node_labels[v]].append(i)
    for k in lab_match_eids:
        lab_match_m[k] = len(lab_match_eids[k])
    return lab_match_eids, lab_match_m


class MCMC_LW:

    def __init__(self, 
                 edges: list[tuple[int, int]], 
                 degrees: dict[int, int], 
                 node_labels: dict[int, int]):

        self.has_converged = False
        self.spacing = -1
        if len(set(degrees.values())) == 1:
            raise ValueError("Regular graph! Degree assortativity undefined.")
        self.degrees = degrees
        self.node_labels = node_labels
        self.initialize(edges)

    # @profile
    def MCMC_step(self,
                  A: dict[tuple[int, int], int], 
                  edge_list: list[tuple[int, int]],
                  swapped: list[int]) -> float:
        '''
        Performs a LSO.
        
        INPUT
        ======
        A (dict): The weight of each edge. Changed inplace.
        edge_list (list): List of edges in A. Changed inplace.
        labels (list): list of node labels.
        lab_match_eids (dict): for each node label, list of matching edge ids.
        lab_match_m (dict): for each label, number of matching edges.
        node_labels (dict): For each node, its label.
        
        OUTPUT
        ======
        swaps (list): Four nodes swapped if a swap is accepted. Empty otherwise.
        '''
        # flip fair coin
        l = random.choice(self.labels)
        m = self.lab_match_m[l]
        # sample two edges
        p1 = random.randrange(m)
        p2 = random.randrange(m - 1)
        if p1 == p2:
            p2 = m - 1
        eid1 = self.lab_match_eids[l][p1]
        eid2 = self.lab_match_eids[l][p2]
        u, w = edge_list[eid1]
        v, z = edge_list[eid2]
        # print(f'Label {l}, p1={p1}, p2={p2}, u,w={u,w}, v,z={v,z}')
        # num loops and uniques 
        self_loops = 0
        unique = 4
        if u == w:
            self_loops += 1
            unique -= 1
        if v == z:
            self_loops += 1
            unique -= 1
        if u == v:
            unique -= 1
        if u == z:
            unique -= 1
        if w == v:
            unique -= 1
        if w == z:
            unique -= 1
        # Case 1: all nodes are equal
        if unique <= 1:
            swapped[0] = -1
            return -1
        if unique == 2:
            # Case 2A
            if self_loops == 2:
                # print(f'edges are two self loops {u,w} = {u,u} and {v,z}={v,v}')
                # u and v must have the same label
                swapped[0] = u
                swapped[1] = u
                swapped[2] = v
                swapped[3] = v
                xi = (A.get((u, v), 0) + 2) * (A.get((u, v), 0) + 1) / (A[u, u] * A[v, v])
            # Case 2B
            elif self_loops == 0 and self.node_labels[u] == self.node_labels[w]:
                # print(f'edges are the same multiedge {u,w}={v,z}')
                # u and w must have the same label
                swapped[0] = u
                swapped[1] = w
                swapped[2] = u
                swapped[3] = w
                xi = (A.get((u, u), 0) + 1) * (A.get((w, w), 0) + 1) / (A[u, w] * (A[u, w] - 1))
            # Cases 2C and 2D     
            else:
                swapped[0] = -1
                return -1
        elif unique == 3:
            # Case 3A
            if self_loops > 0:
                if u == w:
                    # print(f'self loop {u,w} and separate edge {v,z}')
                    # we need this to ensure label consistency
                    if self.node_labels[v] == self.node_labels[z] or self.node_labels[u] == self.node_labels[v]:
                        swapped[0] = u
                        swapped[1] = u
                        swapped[2] = v
                        swapped[3] = z
                    else:
                        swapped[0] = u
                        swapped[1] = u
                        swapped[2] = z
                        swapped[3] = v
                    xi = (A.get((u, v), 0) + 1) * (A.get((u, z), 0) + 1) / (A[u, u] * A[v, z])
                else:
                    # print(f'self loop {v,z} and separate edge {u,w}')
                    # we need this to ensure label consistency
                    if self.node_labels[u] == self.node_labels[w] or self.node_labels[v] == self.node_labels[w]:
                        swapped[0] = u
                        swapped[1] = w
                        swapped[2] = v
                        swapped[3] = v
                    else:
                        swapped[0] = w
                        swapped[1] = u
                        swapped[2] = v
                        swapped[3] = v
                    xi = (A.get((u, v), 0) + 1) * (A.get((v, w), 0) + 1) / (A[u, w] * A[v, v])
            else: # line 19
                nl = len(set([self.node_labels[u], 
                              self.node_labels[w],
                              self.node_labels[v],
                              self.node_labels[z]]))
                if u == v:
                    # print(f'wedge centered on {u}={v}')
                    if nl == 1 or self.node_labels[u] == self.node_labels[w]:
                        swapped[0] = u
                        swapped[1] = w
                        swapped[2] = u
                        swapped[3] = z
                        xi = (A.get((u, u), 0) + 1) * (A.get((w, z), 0) + 1) / (A[u, w] * A[u, z])
                    elif self.node_labels[v] == self.node_labels[z]:
                        swapped[0] = w
                        swapped[1] = u
                        swapped[2] = z
                        swapped[3] = u
                        xi = (A.get((u, u), 0) + 1) * (A.get((w, z), 0) + 1) / (A[u, w] * A[u, z])
                    # Case 3D
                    else:
                        swapped[0] = -1
                        return -1
                elif u == z:
                    # print(f'wedge centered on {u}={z}')
                    if nl == 1 or self.node_labels[u] == self.node_labels[w]:
                        swapped[0] = u
                        swapped[1] = w
                        swapped[2] = u
                        swapped[3] = v
                        xi = (A.get((u, u), 0) + 1) * (A.get((w, v), 0) + 1) / (A[u, w] * A[u, v])
                    elif self.node_labels[v] == self.node_labels[z]:
                        swapped[0] = w
                        swapped[1] = u
                        swapped[2] = v
                        swapped[3] = u
                        xi = (A.get((u, u), 0) + 1) * (A.get((w, v), 0) + 1) / (A[u, w] * A[u, v])
                    # Case 3D
                    else:
                        swapped[0] = -1
                        return -1
                elif w == v:
                    # print(f'wedge centered on {w}={v}')
                    if nl == 1 or self.node_labels[v] == self.node_labels[z]:
                        swapped[0] = u
                        swapped[1] = w
                        swapped[2] = z
                        swapped[3] = w
                        xi = (A.get((w, w), 0) + 1) * (A.get((u, z), 0) + 1) / (A[u, w] * A[w, z])
                    elif self.node_labels[u] == self.node_labels[w]:
                        swapped[0] = w 
                        swapped[1] = u
                        swapped[2] = w
                        swapped[3] = z
                        xi = (A.get((w, w), 0) + 1) * (A.get((u, z), 0) + 1) / (A[u, w] * A[w, z])
                    # Case 3D
                    else:
                        swapped[0] = -1
                        return -1
                elif w == z:
                    # print(f'wedge centered on {w}={z}')
                    if nl == 1 or self.node_labels[v] == self.node_labels[z]:
                        swapped[0] = u
                        swapped[1] = w
                        swapped[2] = v
                        swapped[3] = w
                        xi = (A.get((w, w), 0) + 1) * (A.get((u, v), 0) + 1) / (A[u, w] * A[w, v])
                    elif self.node_labels[u] == self.node_labels[w]:
                        swapped[0] = w
                        swapped[1] = u
                        swapped[2] = w
                        swapped[3] = v
                        xi = (A.get((w, w), 0) + 1) * (A.get((u, v), 0) + 1) / (A[u, w] * A[w, v])
                    # Case 3D
                    else:
                        swapped[0] = -1
                        return -1
                # Case 3D
                else:
                    swapped[0] = -1
                    return -1
        else:
            nl = len(set([self.node_labels[u], 
                          self.node_labels[w],
                          self.node_labels[v],
                          self.node_labels[z]]))
            cond = self.node_labels[u] != self.node_labels[w] and self.node_labels[v] != self.node_labels[z]
            # Case 4A
            if nl == 3 and cond:
                if self.node_labels[u] == self.node_labels[v]:
                    swapped[0] = w
                    swapped[1] = u
                    swapped[2] = v
                    swapped[3] = z
                    xi = (A.get((u, z), 0) + 1) * (A.get((v, w), 0) + 1) / (A[u, w] * A[z, v])
                elif self.node_labels[w] == self.node_labels[z]:
                    swapped[0] = u
                    swapped[1] = w
                    swapped[2] = z
                    swapped[3] = v
                    xi = (A.get((u, z), 0) + 1) * (A.get((v, w), 0) + 1) / (A[u, w] * A[z, v])
                elif self.node_labels[u] == self.node_labels[z]:
                    swapped[0] = w
                    swapped[1] = u
                    swapped[2] = z
                    swapped[3] = v
                    xi = (A.get((u, v), 0) + 1) * (A.get((z, w), 0) + 1) / (A[u, w] * A[z, v])
                else:
                    swapped[0] = u
                    swapped[1] = w
                    swapped[2] = v
                    swapped[3] = z
                    xi = (A.get((u, v), 0) + 1) * (A.get((z, w), 0) + 1) / (A[u, w] * A[z, v])
            # Case 4B
            elif nl == 2 and cond:
                l2 = self.node_labels[u]
                if l2 == l:
                    l2 = self.node_labels[w]
                m2 = self.lab_match_m[l2]
                xi_d_1 = (A[u, w] * A[v, z]) / (m * (m - 1))  
                xi_d_2 = (A[u, w] * A[v, z]) / (m2 * (m2 - 1))
                xi_d = xi_d_1 + xi_d_2
                if self.node_labels[u] == self.node_labels[v]:
                    swapped[0] = u
                    swapped[1] = w
                    swapped[2] = z
                    swapped[3] = v
                    xi_n_1 = ((A.get((u, z), 0) + 1) * (A.get((v, w), 0) + 1) / (m * (m - 1)))
                    xi_n_2 = ((A.get((u, z), 0) + 1) * (A.get((v, w), 0) + 1) / (m2 * (m2 - 1)))
                    xi_n = xi_n_1 + xi_n_2
                    xi = xi_n / xi_d
                else:
                    # node_labels[u] == node_labels[z]
                    swapped[0] = u
                    swapped[1] = w
                    swapped[2] = v
                    swapped[3] = z
                    xi_n_1 = ((A.get((u, v), 0) + 1) * (A.get((z, w), 0) + 1) / (m * (m - 1)))
                    xi_n_2 = ((A.get((u, v), 0) + 1) * (A.get((z, w), 0) + 1) / (m2 * (m2 - 1)))
                    xi_n = xi_n_1 + xi_n_2
                    xi = xi_n / xi_d
            # Case 4C
            else:
                if random.uniform(0, 1) < .5:
                    if self.node_labels[w] == self.node_labels[z]:
                        swapped[0] = u
                        swapped[1] = w
                        swapped[2] = z
                        swapped[3] = v
                    else:
                        swapped[0] = w
                        swapped[1] = u
                        swapped[2] = v
                        swapped[3] = z
                    xi = (A.get((u, z), 0) + 1) * (A.get((v, w), 0) + 1) / (A[u, w] * A[z, v])
                elif self.node_labels[w] == self.node_labels[v]:
                    swapped[0] = u
                    swapped[1] = w
                    swapped[2] = v
                    swapped[3] = z
                    xi = (A.get((u, v), 0) + 1) * (A.get((z, w), 0) + 1) / (A[u, w] * A[z, v])
                else:
                    swapped[0] = w
                    swapped[1] = u
                    swapped[2] = z
                    swapped[3] = v
                    xi = (A.get((u, v), 0) + 1) * (A.get((z, w), 0) + 1) / (A[u, w] * A[z, v])
        P = min(xi, 1.0)
        # If we proceed with the swap we update A, swaps and edge_list
        if random.uniform(0, 1) < P:
            A[swapped[0], swapped[1]] -= 1
            A[swapped[1], swapped[0]] -= 1
            if A[swapped[0], swapped[1]] == 0:
                del A[swapped[0], swapped[1]]
                if swapped[0] != swapped[1]:
                    del A[swapped[1], swapped[0]]
            A[swapped[2], swapped[3]] -= 1
            A[swapped[3], swapped[2]] -= 1   
            if A[swapped[2], swapped[3]] == 0:
                del A[swapped[2], swapped[3]]
                if swapped[2] != swapped[3]:
                    del A[swapped[3], swapped[2]]
            A[swapped[0], swapped[2]] = A.get((swapped[0], swapped[2]), 0) + 1
            A[swapped[2], swapped[0]] = A.get((swapped[2], swapped[0]), 0) + 1
            A[swapped[1], swapped[3]] = A.get((swapped[1], swapped[3]), 0) + 1
            A[swapped[3], swapped[1]] = A.get((swapped[3], swapped[1]), 0) + 1

            edge_list[eid1] = swapped[0], swapped[2]
            edge_list[eid2] = swapped[1], swapped[3]
            
            return P
        swapped[0] = -1
        return P

    def initialize(self, 
                   edges: list[tuple[int,int]]):
        '''
        edges (list): list of edges.
        degrees (dict): for each node, its degree.
        node_labels (dict): for each node, its label.
        '''
        self.n = len(self.degrees)
        self.m = len(edges)
        self.A = dict()
        ut.convert_edgelist_to_dictionary(edges, self.A)
        self.edge_list = [(u, v) for u, v in edges]
        self.lab_match_eids, self.lab_match_m = get_compatible_edges_per_label(self.edge_list, self.node_labels)
        self.labels = []
        for l in self.lab_match_m:
            if self.lab_match_m[l] > 1:
                self.labels.append(l)

        S1 = 2 * self.m
        S2 = 0
        S3 = 0
        for i in range(self.n):
            S2 += (self.degrees[i])**2
            S3 += (self.degrees[i])**3
        self.r_denominator = S1 * S3 - (S2**2)
        self.S2 = S2
