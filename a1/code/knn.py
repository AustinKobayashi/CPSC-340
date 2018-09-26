"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
from scipy import stats
import utils

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X # just memorize the training data
        self.y = y 

    def predict(self, Xtest):
        n_train = self.X.shape[0]
        n_test = Xtest.shape[0]

        # dist_squared will be a n_test by  n_train numpy array
        # utils.euclidean_dist_squared takes args (X, Xtest)
        # but this yields an array with the size of arg X as the first dimension
        # which I don't want for the following operations
        dist_squared = utils.euclidean_dist_squared(Xtest, self.X)

        # indices of the array, sorted by the array's values
        sorted_indices = np.argsort(dist_squared)

        out = np.zeros(n_test)

        # assumes that both n_train and n_test are >= self.k
        for i in range(sorted_indices.shape[0]):
            indices = sorted_indices[i,:self.k]
            
            # maps from index in sorted indices to training y val
            values = np.fromiter((self.y[j] for j in indices), int)
            value_sum = np.sum(values)
           
            # this implementation favors 0 in the case of a tie
            if value_sum > self.k/2:
                # out is zeroes to begin with, so we only need to set in the true case
                out[i] = 1

        return out
        


class CNN(KNN):

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : an N by D numpy array
        y : an N by 1 numpy array of integers in {1,2,3,...,c}
        """

        Xcondensed = X[0:1,:]
        ycondensed = y[0:1]

        for i in range(1,len(X)):
            x_i = X[i:i+1,:]
            dist2 = utils.euclidean_dist_squared(Xcondensed, x_i)
            inds = np.argsort(dist2[:,0])
            yhat = utils.mode(ycondensed[inds[:min(self.k,len(Xcondensed))]])

            if yhat != y[i]:
                Xcondensed = np.append(Xcondensed, x_i, 0)
                ycondensed = np.append(ycondensed, y[i])

        self.X = Xcondensed
        self.y = ycondensed

        print(self.y.shape[0])
