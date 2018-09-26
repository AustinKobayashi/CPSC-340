import numpy as np

class SimpleDecision:

    def __init__(self):
        pass

    def predict(self, X):

        M, D = X.shape
        y = np.zeros(M)
        
        for n in range(M):
            
            if(X[n,1] > 37.669007):
                if(X[n,0] <= -96.090109):
                    y[n] = 1
            else:
                if(X[n,0] > -115.577574):
                    y[n] = 1

        return y
