import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

import linear_model
import utils

from sklearn.model_selection import train_test_split


def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

def test_and_plot(model,X,y,Xtest=None,ytest=None,title=None,filename=None,format=None):

    # Compute training error
    yhat = model.predict(X)
    trainError = np.mean((yhat - y)**2)
    print("Training error = %.1f" % trainError)

    # Compute test error
    if Xtest is not None and ytest is not None:
        yhat = model.predict(Xtest)
        testError = np.mean((yhat - ytest)**2)
        print("Test error     = %.1f" % testError)
        print("Approx. error  = %0.3f" % (testError - trainError))
    
    # Plot model
    plt.figure()
    plt.plot(X,y,'b.')
    
    # Choose points to evaluate the function
    Xgrid = np.linspace(np.min(X),np.max(X),1000)[:,None]
    ygrid = model.predict(Xgrid)
    plt.plot(Xgrid, ygrid, 'g')
    
    if title is not None:
        plt.title(title)
    
    if filename is not None:
        filename = os.path.join("..", "figs", filename)
        print("Saving", filename)
        plt.savefig(filename, dpi=400, format=format)


def recursive_split(X, y, n, prev_x_splits=None, prev_y_splits=None):
    if n == 1:
        prev_x_splits.append(X)
        prev_y_splits.append(y)
        return np.asarray(prev_x_splits), np.asarray(prev_y_splits)
    else:
        split_proportion = 1.0 / n
        X1, X2, y1, y2 = train_test_split(X, y, test_size=split_proportion)
        
        if prev_x_splits is None:
            prev_x_splits = [X2]
        else:
            prev_x_splits.append(X2)

        if prev_y_splits is None:
            prev_y_splits = [y2]
        else:
            prev_y_splits.append(y2)

        return recursive_split(X1, y1, n-1, prev_x_splits=prev_x_splits, prev_y_splits=prev_y_splits)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)
    io_args = parser.parse_args()
    question = io_args.question


    if question == "2":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # Fit least-squares estimator
        model = linear_model.LeastSquares()
        model.fit(X,y)
        print(model.w)

        test_and_plot(model,X,y,title="Least Squares",filename="least_squares_outliers.pdf")

    elif question == "2.1":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']
        z = np.concatenate((np.full((400, 1), 1), (np.full((100, 1), 0.1))), axis=0)

        model = linear_model.WeightedLeastSquares()
        model.fit(X,y,z)
        print(model.w)

        test_and_plot(model,X,y,title="Weighted Least Squares",filename="least_squares_outliers_weighted.png")

    elif question == "2.3":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        model = linear_model.LinearModelGradient()
        model.fit(X,y)
        print(model.w)

        test_and_plot(model,X,y,title="Robust (L1) Linear Regression",filename="least_squares_robust.png")

    elif question == "3":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # Fit least-squares model
        model = linear_model.LeastSquares()
        model.fit(X,y)

        test_and_plot(model,X,y,Xtest,ytest,title="Least Squares, no bias",filename="least_squares_no_bias.pdf")

    elif question == "3.1":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        model = linear_model.LeastSquaresBias()
        model.fit(X,y)
        
        test_and_plot(model,X,y,Xtest,ytest,title="Least Squares, with bias",filename="least_squares_bias.png")

    elif question == "3.2":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']
        
        trainError = np.zeros(8)
        testError = np.zeros(8)
        approxError = np.zeros(8)
        
        for p in range(11):
            print("p = %d" % p)

            model = linear_model.LeastSquaresPoly(p)
            model.fit(X, y)
            test_and_plot(model,X,y,Xtest,ytest,title='Least Squares Polynomial p = %d'%p,filename="PolyBasis%d.png"%p)


    elif question == "4":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        n,d = X.shape
        
        # Split training data into a training and a validation set

        #Xtrain, Xvalid, ytrain, yvalid = train_test_split(X, y, train_size=0.5)

        Xtrain = X[0:n//2]
        ytrain = y[0:n//2]
        Xvalid = X[n//2: n]
        yvalid = y[n//2: n]

        # Find best value of RBF kernel parameter,
        # training on the train set and validating on the validation set

        minErr = np.inf
        for s in range(-15,16):
            sigma = 2 ** s

            # Train on the training set
            model = linear_model.LeastSquaresRBF(sigma)
            model.fit(Xtrain,ytrain)

            # Compute the error on the validation set
            yhat = model.predict(Xvalid)
            validError = np.mean((yhat - yvalid)**2)
            print("Error with sigma = 2^%-3d = %.1f" % (s ,validError))

            # Keep track of the lowest validation error
            if validError < minErr:
                minErr = validError
                bestSigma = sigma

        print("Value of sigma that achieved the lowest validation error =", bestSigma)

        # Now fit the model based on the full dataset.
        print("Refitting on full training set...\n")
        model = linear_model.LeastSquaresRBF(bestSigma)
        model.fit(X,y)

        test_and_plot(model,X,y,Xtest,ytest,
            title="Least Squares with RBF kernel and $\sigma={}$".format(bestSigma),
            filename="least_squares_rbf_bad.png",
            format="png")

            
    elif question == "4.1": 
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        n,d = X.shape
        
        # Split training data into a training and a validation set

        Xtrain, Xvalid, ytrain, yvalid = train_test_split(X, y, train_size=0.5)

        # Find best value of RBF kernel parameter,
        # training on the train set and validating on the validation set

        minErr = np.inf
        for s in range(-15,16):
            sigma = 2 ** s

            # Train on the training set
            model = linear_model.LeastSquaresRBF(sigma)
            model.fit(Xtrain,ytrain)

            # Compute the error on the validation set
            yhat = model.predict(Xvalid)
            validError = np.mean((yhat - yvalid)**2)
            print("Error with sigma = 2^%-3d = %.1f" % (s ,validError))

            # Keep track of the lowest validation error
            if validError < minErr:
                minErr = validError
                bestSigma = sigma

        print("Value of sigma that achieved the lowest validation error =", bestSigma)

        # Now fit the model based on the full dataset.
        print("Refitting on full training set...\n")
        model = linear_model.LeastSquaresRBF(bestSigma)
        model.fit(X,y)

        test_and_plot(model,X,y,Xtest,ytest,
            title="Least Squares with RBF kernel and $\sigma={}$".format(bestSigma),
            filename="least_squares_rbf_better.png",
            format="png")
    
    elif question == "4.2":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        n,d = X.shape
        
        # Split training data into a training and a validation set


        Xsets, ysets = recursive_split(X, y, 10)

        n_train = (Xsets.shape[0] - 1) * Xsets.shape[1]
        

        minErr = np.inf
        for i in range(10):
            Xvalid = Xsets[i]
            yvalid = ysets[i]

            Xtrain = None
            ytrain = None
            for j in range(10):
                if j != i:
                    if Xtrain is None:
                        Xtrain = Xsets[j]
                        ytrain = ysets[j]
                    else:
                        Xtrain = np.append(Xtrain, Xsets[j])
                        ytrain = np.append(ytrain, ysets[j])
            Xtrain = np.reshape(Xtrain, (n_train, d))
            ytrain = np.reshape(ytrain, (n_train, 1))

            print("\nFor cross-validation test set %d:" % i)
            for s in range(-15,16):
                sigma = 2 ** s

                # Train on the training set
                model = linear_model.LeastSquaresRBF(sigma)
                model.fit(Xtrain,ytrain)

                # Compute the error on the validation set
                yhat = model.predict(Xvalid)
                validError = np.mean((yhat - yvalid)**2)
                print("Error with sigma = 2^%-3d = %.1f" % (s ,validError))

                # Keep track of the lowest validation error
                if validError < minErr:
                    minErr = validError
                    bestSigma = sigma


        print("Value of sigma that achieved the lowest validation error =", bestSigma)

        # Now fit the model based on the full dataset.
        print("Refitting on full training set...\n")
        model = linear_model.LeastSquaresRBF(bestSigma)
        model.fit(X,y)

        test_and_plot(model,X,y,Xtest,ytest,
            title="Least Squares with RBF kernel and $\sigma={}$".format(bestSigma),
            filename="least_squares_rbf_best.png",
            format="png")
    