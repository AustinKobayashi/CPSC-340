import numpy as np
import utils
from random_tree import RandomTree

class RandomForest():
        
    def __init__(self, max_depth, num_trees):
        self.max_depth = max_depth
        self.num_trees = num_trees



    def fit(self, X, y):
        
        self.random_trees = []
        
        for i in range(self.num_trees):
            random_tree = RandomTree(max_depth=self.max_depth)
            random_tree.fit(X, y)
            self.random_trees.append(random_tree)
            
            
            
    def predict(self, X):
        
        M, D = X.shape
        random_trees_pred = np.zeros((M, self.num_trees))
        y = np.zeros(M)
        
        for i in range(self.num_trees):
            random_trees_pred[:,i] = self.random_trees[i].predict(X)
                
        for i in range(M):
            y[i] = utils.mode(random_trees_pred[i,:])
        
        return y
        