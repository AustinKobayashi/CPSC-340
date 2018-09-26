import numpy as np
import utils

class DecisionStump:

    def __init__(self):
        pass

    def fit(self, X, y):
        """ YOUR CODE HERE FOR Q2.1 """
        N, D = X.shape


        y_mode = utils.mode(y)

        splitSat = y_mode
        splitVariable = None
        splitValue = None
        splitNot = None

        minError = np.sum(y != y_mode)

        # Check if labels are not all equal
        if np.unique(y).size > 1:
            # Loop over features looking for the best split
            for d in range(D):
                for n in range(N):
                    # Choose value to equate to
                    value = X[n, d]

                    # Find most likely class for each split
                    y_sat = utils.mode(y[X[:,d] > value])
                    y_not = utils.mode(y[X[:,d] <= value])

                    # Make predictions
                    y_pred = y_sat * np.ones(N)
                    y_pred[X[:, d] <= value] = y_not

                    # Compute error
                    errors = np.sum(y_pred != y)

                    # Compare to minimum error so far
                    if errors < minError:
                        # This is the lowest error, store this value
                        minError = errors
                        splitVariable = d
                        splitValue = value
                        splitSat = y_sat
                        splitNot = y_not

        self.splitVariable = splitVariable
        self.splitValue = splitValue
        self.splitSat = splitSat
        self.splitNot = splitNot


    def predict(self, X):
        """ YOUR CODE HERE FOR Q2.1 """
        splitVariable = self.splitVariable
        splitValue = self.splitValue
        splitSat = self.splitSat
        splitNot = self.splitNot

        M, D = X.shape

        if splitVariable is None:
            return splitSat * np.ones(M)

        yhat = np.zeros(M)

        for m in range(M):
            if X[m, splitVariable] > splitValue:
                yhat[m] = splitSat
            else:
                yhat[m] = splitNot

        return yhat



class DecisionStumpEquality:

    def __init__(self):
        pass


    def fit(self, X, y):
        N, D = X.shape

        y_mode = utils.mode(y)

        splitSat = y_mode
        splitVariable = None
        splitValue = None
        splitNot = None

        minError = np.sum(y != y_mode)

        # Check if labels are not all equal
        if np.unique(y).size > 1:
            # Loop over features looking for the best split
            X = np.round(X)

            for d in range(D):
                for n in range(N):
                    # Choose value to equate to
                    value = X[n, d]

                    # Find most likely class for each split
                    y_sat = utils.mode(y[X[:,d] == value])
                    y_not = utils.mode(y[X[:,d] != value])

                    # Make predictions
                    y_pred = y_sat * np.ones(N)
                    y_pred[X[:, d] != value] = y_not

                    # Compute error
                    errors = np.sum(y_pred != y)

                    # Compare to minimum error so far
                    if errors < minError:
                        # This is the lowest error, store this value
                        minError = errors
                        splitVariable = d
                        splitValue = value
                        splitSat = y_sat
                        splitNot = y_not

        self.splitVariable = splitVariable
        self.splitValue = splitValue
        self.splitSat = splitSat
        self.splitNot = splitNot


    def predict(self, X):

        M, D = X.shape
        X = np.round(X)

        if self.splitVariable is None:
            return self.splitSat * np.ones(M)

        yhat = np.zeros(M)

        for m in range(M):
            if X[m, self.splitVariable] == self.splitValue:
                yhat[m] = self.splitSat
            else:
                yhat[m] = self.splitNot

        return yhat



# helper function. leaves zeros as zeros.
def log0(x):
    x = x.copy()
    x[x>0] = np.log(x[x>0])
    return x
    
    
class DecisionStumpInfoGain(DecisionStump):

    def fit(self, X, y, split_features=None):
                
        N, D = X.shape

        # Address the trivial case where we do not split
        count = np.bincount(y)

        # Compute total entropy (needed for information gain)
        p = count/np.sum(count); # Convert counts to probabilities
        entropyTotal = -np.sum(p*log0(p))

        maxGain = 0
        self.splitVariable = None
        self.splitValue = None
        self.splitSat = np.argmax(count)
        self.splitNot = None

        # Check if labels are not all equal
        if np.unique(y).size <= 1:
            return

        if split_features is None:
            split_features = range(D)

        for d in split_features:
            thresholds = np.unique(X[:,d])
            for value in thresholds[:-1]:
                # Count number of class labels where the feature is greater than threshold
                y_vals = y[X[:,d] > value]
                count1 = np.bincount(y_vals)
                count1 = np.pad(count1, (0,len(count)-len(count1)), \
                                mode='constant', constant_values=0)  # pad end with zeros to ensure same length as 'count'
                count0 = count-count1
                                
                # Compute infogain
                p1 = count1/np.sum(count1)
                p0 = count0/np.sum(count0)
                H1 = -np.sum(p1*log0(p1))
                H0 = -np.sum(p0*log0(p0))
                prob1 = np.sum(X[:,d] > value)/N
                prob0 = 1-prob1

                infoGain = entropyTotal - prob1*H1 - prob0*H0
                # assert infoGain >= 0
                # Compare to minimum error so far
                if infoGain > maxGain:
                    # This is the highest information gain, store this value
                    maxGain = infoGain
                    splitVariable = d
                    splitValue = value
                    splitSat = np.argmax(count1)
                    splitNot = np.argmax(count0)
    
        # if infoGain > 0: # if there's an actual split. rather than everything going to one side. there are other ways of checking this condition...
        self.splitVariable = splitVariable
        self.splitValue = splitValue
        self.splitSat = splitSat
        self.splitNot = splitNot
