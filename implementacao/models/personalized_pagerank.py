import numpy as np
import dgl
import torch
import pickle
from random import choices


class PersonalizedPageRank():
    
    def __init__(self, scipy_adj_matrix = None, dgl_network = None, visit_target = 1, rank_limit=500, reset_prob=0.0, query_target_size=10):
        assert scipy_adj_matrix != None or dgl_network != None, "Input DGL Network or Scipy Adjacency Matrix"

        self.visit_target       = visit_target
        self.rank_limit         = rank_limit
        self.reset_prob         = reset_prob
        self.query_target_size  = query_target_size

        if scipy_adj_matrix != None:
            self.from_scipy(scipy_adj_matrix)
        elif dgl_network != None:
            self.from_dgl(dgl_network)

        pass


    def from_scipy(self, scipy_adj_matrix):
        self.network = dgl.from_scipy(
            sp_mat          = scipy_adj_matrix
        )

        print("Loading Weights: This process takes a few minutes")
        init_nodes, final_nodes = self.network.edges()

        weights = []
        for i in range(len(init_nodes)):
            weight = scipy_adj_matrix[init_nodes[i], final_nodes[i]]
            weights.append( weight )

        self.network.edata['weights'] = torch.tensor(weights, dtype=float)
        
        pass
    

    def from_dgl(self, dgl_network):
        self.network = dgl_network

        self.network.edata['weights'] = self.network.edata['weights'].double()
        pass


    def predict(self, X):
        return [self.predict_one(x) for x in X]

        
    def predict_one(self, query, return_rank=False):
        rank = {}

        full_query = choices(query, k=self.query_target_size)

        paths, _ = dgl.sampling.random_walk(
            g               = self.network, 
            nodes           = full_query, 
            length          = self.visit_target, 
            prob            = 'weights',
            restart_prob    = self.reset_prob
        )

        for path in paths:
            for node in path:
                node_ = int(node)
                # Remove nodes from Query and -1 (a truncation indicator of DGL)
                if node_ == -1 or node_ in query:
                    continue

                if node in rank.keys():
                    rank[node_] += 1
                else:
                    rank[node_] = 1

        rank_order = sorted(rank.items(), key=lambda x:x[1], reverse=True)
        ranked_nodes    = [x for x, _ in rank_order]
        
        if return_rank:
            return ranked_nodes[:self.rank_limit], rank
        else:
            return ranked_nodes[:self.rank_limit]
