import numpy as np


class PersonalizedPageRank():
    
    def __init__(self, network, walk_depth=5, n_walks = 5):
        self.network = network
        self.walk_depth = walk_depth
        self.n_walks = n_walks
    
    def choice(self, options, probs):
        x = np.random.rand()
        cum = 0
        idx = 0
        for idx, p in enumerate(probs):
            cum += p
            if x < cum:
                break
        return options[idx]

    def predict(self, X):
        return [self.predict_one(x) for x in X]

        
    def predict_one(self, query):
        visited_nodes = []

        #TODO: Paralelizar
        for node in query:
            for _ in range(self.n_walks):
                path = self.randomwalk(node)
                visited_nodes.extend(path)

        rank = {}
        for node in visited_nodes:
            if node in rank.keys():
                rank[node] += 1
            else:
                rank[node] = 1

        rank = sorted(rank.items(), key=lambda x:x[1], reverse=True)
        ranked_nodes = [x for x, _ in rank]
        
        return ranked_nodes
    
    def get_neighborhood(self, node):
        neighborhood = []
        probabilities = []
        for i, j in self.network.keys():
            if i == node:
                neighborhood.append(j)
                probabilities.append(self.network[i,j])

        return neighborhood, probabilities
    
    def randomwalk(self, initial_node):
        curr_node = initial_node

        path = []
        for curr_step in range(self.walk_depth):
            neighbours, probabilities = self.get_neighborhood(curr_node)

            if len(neighbours) > 0:
                curr_node = self.choice(neighbours, probabilities)
                path.append(curr_node)
            else:
                break
        
        return path           
