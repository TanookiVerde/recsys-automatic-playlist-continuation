import numpy as np
import dgl
import torch
import pickle


class PersonalizedPageRank():
    
    def __init__(self, scipy_adj_matrix = None, dgl_network = None, walk_depth=100, n_rounds = 1, rank_limit=500):
        assert scipy_adj_matrix != None or dgl_network != None, "Input DGL Network or Scipy Adjacency Matrix"

        self.walk_depth     = walk_depth
        self.n_rounds        = n_rounds
        self.rank_limit     = rank_limit

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

        self.network.edata['weights'] = torch.tensor(weights)
        
        pass
    

    def from_dgl(self, dgl_network):
        self.network = dgl_network

        pass


    def predict(self, X):
        return [self.predict_one(x) for x in X]

        
    def predict_one(self, query):
        visited_nodes = []
        for i in range(self.n_rounds):
            paths, _ = dgl.sampling.random_walk(
                g       = self.network, 
                nodes   = query, 
                length  = self.walk_depth, 
                prob    = 'weights'
            )
            visited_nodes_in_round = np.concatenate(paths.numpy()).tolist()
            visited_nodes.extend(visited_nodes_in_round)

        rank = {}
        for node in visited_nodes:
            # Remove nodes from Query and -1 (a truncation indicator of DGL)
            if node == -1 or node in query:
                continue

            if node in rank.keys():
                rank[node] += 1
            else:
                rank[node] = 1

        rank = sorted(rank.items(), key=lambda x:x[1], reverse=True)
        ranked_nodes    = [x for x, _ in rank]
        
        return ranked_nodes[:self.rank_limit]
