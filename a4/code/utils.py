import pickle
import os
import sys
import numpy as np
from scipy.optimize import approx_fprime

def load_dataset(dataset_name):

    # Load and standardize the data and add the bias term
    if dataset_name == "logisticData":
        with open(os.path.join('..', 'data', 'logisticData.pkl'), 'rb') as f:
            data = pickle.load(f)

        X, y = data['X'], data['y']
        Xvalid, yvalid = data['Xvalidate'], data['yvalidate']

        X, mu, sigma = standardize_cols(X)
        Xvalid, _, _ = standardize_cols(Xvalid, mu, sigma)

        X = np.hstack([np.ones((X.shape[0], 1)), X])
        Xvalid = np.hstack([np.ones((Xvalid.shape[0], 1)), Xvalid])

        return {"X":X, "y":y,
                "Xvalid":Xvalid,
                "yvalid":yvalid}

    elif dataset_name == "multiData":
        with open(os.path.join('..', 'data', 'multiData.pkl'), 'rb') as f:
            data = pickle.load(f)
        
        X, y = data['X'], data['y']
        Xvalid, yvalid = data['Xvalidate'], data['yvalidate']

        X, mu, sigma = standardize_cols(X)
        Xvalid, _, _ = standardize_cols(Xvalid, mu, sigma)

        X = np.hstack([np.ones((X.shape[0], 1)), X])
        Xvalid = np.hstack([np.ones((Xvalid.shape[0], 1)), Xvalid])

        y -= 1  # so classes are 0,..,4
        yvalid -=1

        return {"X":X, "y":y,
                "Xvalid":Xvalid,
                "yvalid":yvalid}

    else:
        with open(os.path.join('..', DATA_DIR, dataset_name+'.pkl'), 'rb') as f:
            data = pickle.load(f)
        return data

def standardize_cols(X, mu=None, sigma=None):
    # Standardize each column with mean 0 and variance 1
    n_rows, n_cols = X.shape

    if mu is None:
        mu = np.mean(X, axis=0)

    if sigma is None:
        sigma = np.std(X, axis=0)
        sigma[sigma < 1e-8] = 1.

    return (X - mu) / sigma, mu, sigma


def check_gradient(model, X, y):
    # This checks that the gradient implementation is correct
    w = np.random.rand(model.w.size)
    f, g = model.funObj(w, X, y)

    # Check the gradient
    estimated_gradient = approx_fprime(w,
                                       lambda w: model.funObj(w,X,y)[0],
                                       epsilon=1e-6)

    implemented_gradient = model.funObj(w, X, y)[1]

    if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
        raise Exception('User and numerical derivatives differ:\n%s\n%s' %
             (estimated_gradient[:5], implemented_gradient[:5]))
    else:
        print('User and numerical derivatives agree.')

def classification_error(y, yhat):
    return np.mean(y!=yhat)
