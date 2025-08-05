import random
import sys
sys.path.insert(1,'../')
import src.utils as ut # type: ignore


class CM:

    def __init__(self, 
                 edges: list, 
                 degrees: dict[int,int], 
                 node_labels: dict[int,int]):

        self.has_converged = False
        self.spacing = -1
        if len(set(degrees.values())) == 1:
            raise ValueError("Regular graph! Degree assortativity undefined.")
        self.degrees = degrees
        self.node_labels = node_labels
        self.initialize(edges)
    
    def MCMC_step(self,
                  A: dict[tuple[int, int], int], 
                  edge_list: list[tuple[int, int]],
                  swapped: list[int]) -> float:
        '''
        Performs a step in the Markov chain using MCMC-MH (Algorithm 1).
        
        INPUT
        ======
        A (dict): The weight of each edge. Changed inplace.
        edge_list (list): List of edges in A. Changed inplace.
        
        OUTPUT
        ======
        swaps (list): Four nodes swapped if a swap is accepted. Empty otherwise.
        '''
        # Choose two edges uniformly at random
        p1 = random.randrange(self.m)
        p2 = random.randrange(self.m - 1)
        if p1 == p2:
            p2 = self.m - 1

        u, v = edge_list[p1]
        # Pick either swap orientation 50% at random
        if random.uniform(0, 1) < 0.5:
            x, y = edge_list[p2]
        else:
            y, x = edge_list[p2]

        w_uv = A[u, v]
        w_xy = A[x, y]
        
        w_ux = A.get((u,x), 0)
        w_vy = A.get((v,y), 0)

        if v == x or u == y:
            # line 7 Algorithm 1: swap would leave the multigraph unchanged
            swapped[0] = -1
            return -1
        # if we are here, it means that the swap is valid
        # and it leads to a different multigraph
        num_loops = 0
        if u == v:
            num_loops += 1
        if y == x:
            num_loops += 1
        if num_loops == 1:
            # line 9 Algorithm 1
            p = (w_ux + 1) * (w_vy + 1) / (2 * w_uv * w_xy)
        elif num_loops == 2:
            # line 11 Algorithm 1
            p = (w_ux + 2) * (w_ux + 1) / (4 * w_uv * w_xy)
        elif v == y and u == x:
            # line 13 Algorithm 1
            p = 4 * (w_ux + 1) * (w_vy + 1) / (w_uv * (w_uv - 1))
        elif v == y or u == x:
            # from a wedge to a self-loop on the middle node
            # and an edge between the other two nodes
            p = 2 * (w_ux + 1) * (w_vy + 1) / (w_uv * w_xy)
        else:
            # line 15 Algorithm 1
            p = (w_ux + 1) * (w_vy + 1) / (w_uv * w_xy)

        P = min(p, 1.0)
        # If we proceed with the swap we update A, swaps and edge_list
        if random.uniform(0, 1) < P:
            swapped[0] = u
            swapped[1] = v
            swapped[2] = x
            swapped[3] = y
            
            # if (u == v and A[u, v] == 2) or (u !) 
            
            A[u, v] -= 1
            A[v, u] -= 1
            if A[u, v] == 0:
                del A[u, v]
                if u != v:
                    del A[v, u]
            A[x, y] -= 1
            A[y, x] -= 1   
            if A[x, y] == 0:
                del A[x, y]
                if x != y:
                    del A[y, x]
            A[u, x] = A.get((u,x), 0) + 1
            A[x, u] = A.get((x,u), 0) + 1
            A[v, y] = A.get((v,y), 0) + 1
            A[y, v] = A.get((y,v), 0) + 1

            edge_list[p1] = u, x
            edge_list[p2] = v, y
            return P
        swapped[0] = -1    
        return P
    
    def initialize(self, 
                   edges: list[tuple[int,int]]):
        '''
        edges (list): list of edges.
        degrees (dict): for each node, its degree.
        '''
        n = len(self.degrees)
        
        self.A = dict()
        ut.convert_edgelist_to_dictionary(edges, self.A)
        
        self.edge_list = [(u, v) for u, v in edges]
        self.m = len(edges)

        S1 = 2 * self.m
        S2 = 0
        S3 = 0
        for i in range(n):
            S2 += (self.degrees[i])**2
            S3 += (self.degrees[i])**3

        self.r_denominator = S1 * S3 - (S2**2)
        self.S2 = S2
