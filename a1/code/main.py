# standard Python imports
import os
import argparse
import time
import pickle

# 3rd party libraries
import numpy as np                              # this comes with Anaconda
import matplotlib.pyplot as plt                 # this comes with Anaconda
import pandas as pd                             # this comes with Anaconda
from sklearn.tree import DecisionTreeClassifier # see http://scikit-learn.org/stable/install.html
from sklearn.neighbors import KNeighborsClassifier # same as above

# CPSC 340 code
import utils
from decision_stump import DecisionStump, DecisionStumpEquality, DecisionStumpInfoGain
from decision_tree import DecisionTree
from knn import KNN, CNN
from simple_decision import SimpleDecision

# allow full print of numpy arrays
np.set_printoptions(threshold=np.inf)

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True,
        choices=["1.1", "2", "2.2", "2.3", "2.4", "3", "3.1", "3.2", "4.1", "4.2", "5"])

    io_args = parser.parse_args()
    question = io_args.question

    if question == "1.1":
        # Load the fluTrends dataset
        df = pd.read_csv(os.path.join('..','data','fluTrends.csv'))
        X = df.values
        names = df.columns.values
        
        print("         Min: %s " % np.min(X))
        print("         Max: %s " % np.max(X))
        print("        Mean: %s " % np.mean(X))
        print("      Median: %s " % np.median(X))
        print("        Mode: %s " % utils.mode(X))
        print(" 5%% quantile: %s " % np.percentile(X, 5))
        print("25%% quantile: %s " % np.percentile(X, 25))
        print("50%% quantile: %s " % np.percentile(X, 50))
        print("75%% quantile: %s " % np.percentile(X, 75))
        print("95%% quantile: %s " % np.percentile(X, 95))
        print("Region with max mean: %s " % list(df.columns.values)[np.argmax(X.mean(0))])
        print("Region with min mean: %s " % list(df.columns.values)[np.argmin(X.mean(0))])
        print("Region with max variance: %s " % list(df.columns.values)[np.argmax(np.var(X, 0))])
        print("Region with min variance: %s " % list(df.columns.values)[np.argmin(np.var(X, 0))])

    elif question == "2":

        # 1. Load citiesSmall dataset
        dataset = load_dataset("citiesSmall.pkl")
        X = dataset["X"]
        y = dataset["y"]

        # 2. Evaluate majority predictor model
        y_pred = np.zeros(y.size) + utils.mode(y)

        error = np.mean(y_pred != y)
        print("Mode predictor error: %.3f" % error)

        # 3. Evaluate decision stump
        model = DecisionStumpEquality()
        model.fit(X, y)
        y_pred = model.predict(X)

        error = np.mean(y_pred != y)
        print("Decision Stump with inequality rule error: %.3f" % error)

        # PLOT RESULT
        utils.plotClassifier(model, X, y)

        fname = os.path.join("..", "figs", "q2_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)


    elif question == "2.2":
        # 1. Load citiesSmall dataset
        dataset = load_dataset("citiesSmall.pkl")
        X = dataset["X"]
        y = dataset["y"]
        
        # 3. Evaluate decision stump
        model = DecisionStump()
        model.fit(X, y)
        y_pred = model.predict(X)

        error = np.mean(y_pred != y)
        print("Decision Stump with inequality rule error: %.3f" % error)

        # PLOT RESULT
        utils.plotClassifier(model, X, y)

        fname = os.path.join("..", "figs", "q2_2_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    elif question == "2.3":
        # 1. Load citiesSmall dataset
        dataset = load_dataset("citiesSmall.pkl")
        X = dataset["X"]
        y = dataset["y"]

        # 2. Evaluate decision tree
        model = DecisionTree(max_depth=2)
        model.fit(X, y)

        y_pred = model.predict(X)
        error = np.mean(y_pred != y)

        print("Error: %.3f" % error)

    elif question == "2.4":
        
        dataset = load_dataset("citiesSmall.pkl")
        X = dataset["X"]
        y = dataset["y"]

        depths = np.arange(1,15) # depths to try
        t = time.time()
        my_tree_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTree(max_depth=max_depth)
            model.fit(X, y)
            y_pred = model.predict(X)
            my_tree_errors[i] = np.mean(y_pred != y)
        print("Our decision tree took %f seconds" % (time.time()-t))

        plt.plot(depths, my_tree_errors, label="mine")

        t = time.time()
        sklearn_tree_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=1)
            model.fit(X, y)
            y_pred = model.predict(X)
            sklearn_tree_errors[i] = np.mean(y_pred != y)
        print("scikit-learn's decision tree took %f seconds" % (time.time()-t))


        plt.plot(depths, sklearn_tree_errors, label="sklearn")
        plt.xlabel("Depth of tree")
        plt.ylabel("Classification error")
        plt.legend()
        fname = os.path.join("..", "figs", "q2_4_tree_errors.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)


    elif question == "3":
        dataset = load_dataset("citiesSmall.pkl")
        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]
        model = DecisionTreeClassifier(max_depth=2, criterion='entropy', random_state=1)
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print("Training error: %.3f" % tr_error)
        print("Testing error: %.3f" % te_error)

    elif question == "3.1":
        dataset = load_dataset("citiesSmall.pkl")
        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]

        depths = np.arange(1,15) # depths to try
        tr_error = np.zeros(depths.size)
        te_error = np.zeros(depths.size)
        
        for i, max_depth in enumerate(depths):
            print("i: %s" % i)
            model = DecisionTree(max_depth=max_depth)
            model.fit(X, y)
            y_pred = model.predict(X)
            tr_error[i] = np.mean(y_pred != y)
            print("Training error: %.4f" % tr_error[i])

            y_pred = model.predict(X_test)
            te_error[i] = np.mean(y_pred != y_test)
            print("Testing error: %.4f" % te_error[i])

        plt.plot(depths, tr_error, label="Training Error")
        plt.plot(depths, te_error, label="Testing Error")
        plt.xlabel("Depth of tree")
        plt.ylabel("Classification error")
        plt.legend()
        fname = os.path.join("..", "figs", "q3_1_training_and_testing_errors.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
        
    elif question == "3.2":
        dataset = load_dataset("citiesSmall.pkl")
        X_full, y_full = dataset["X"], dataset["y"]

        n, d = X_full.shape

        X = X_full[0:int(n/2),:]
        X_val = X_full[int(n/2):n,:]
        
        y = y_full[0:int(n/2)]
        y_val = y_full[int(n/2):n]

        depths = np.arange(1,15) # depths to try
        
        for m in range(2):
            tr_error = np.zeros(depths.size)
            val_error = np.zeros(depths.size)
            
            for i, max_depth in enumerate(depths):
                print("depth: %s" % max_depth)
                model = DecisionTree(max_depth=max_depth)
                model.fit(X, y)
                y_pred = model.predict(X)
                tr_error[i] = np.mean(y_pred != y)
                print("Training error: %.4f" % tr_error[i])

                y_pred = model.predict(X_val)
                val_error[i] = np.mean(y_pred != y_val)
                print("Validation error: %.4f" % val_error[i])

            plt.plot(depths, tr_error, label="Training Error")
            plt.plot(depths, val_error, label="Validation Error")

            plt.xlabel("Depth of tree")
            plt.ylabel("Classification error")
            plt.legend()
            
            fname = ""
            if (m == 0):
                fname = os.path.join("..", "figs", "q3_2_validation_set_second_half.pdf")
            else:
                fname = os.path.join("..", "figs", "q3_2_validation_set_first_half.pdf")

            plt.savefig(fname)
            
            print("\nFigure saved as '%s'" % fname)
            
            plt.clf()
            
            Temp = X
            X = X_val
            X_val = Temp
            
            Temp = y
            y = y_val
            y_val = Temp


    elif question == '4.1':
        dataset = load_dataset("citiesSmall.pkl")
        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]
        
        # uses an array for the case of multiple k values
        for k in [1, 10, 50, 100, 150, 200, 250, 300, 350,400]:
            print("\n For k=%i" % k)
            # model = KNN(k)
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X, y)

            y_pred = model.predict(X)
            tr_error = np.mean(y_pred != y)
        
            y_pred = model.predict(X_test)
            te_error = np.mean(y_pred != y_test)

            print("Training error: %.3f" % tr_error)
            print("Testing error: %.3f" % te_error)
            print("Approx error: %.3f" % (te_error - tr_error))

            utils.plotClassifier(model, X, y)
            #plt.show()


    elif question == '4.2':
        dataset = load_dataset("citiesBig1.pkl")
        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]
        # to test multiple possible k values 
        for k in [1]:
            print("\n For k=%i" % k)
            model = CNN(k)
            model.fit(X, y)

            y_pred = model.predict(X)
            tr_error = np.mean(y_pred != y)
        
            y_pred = model.predict(X_test)
            te_error = np.mean(y_pred != y_test)

            print("Training error: %.3f" % tr_error)
            print("Testing error: %.3f" % te_error)
            utils.plotClassifier(model, X, y)
            plt.show()
        
        
        
        t = time.time()
        tree = DecisionTreeClassifier()
        tree.fit(X, y)
        y_pred = tree.predict(X)
        print("scikit-learn's decision tree took %f seconds" % (time.time()-t))
        
        tr_error = np.mean(y_pred != y)
        print("Training Error: %.3f" % tr_error)
        
        y_pred = tree.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print("Testing Error: %.3f" % te_error)
        
        utils.plotClassifier(tree, X, y)
        plt.xlabel("Latitude")
        plt.ylabel("Logitude")
        fname = os.path.join("..", "figs", "q4_2_tree_decision_surface.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)



            
        
         
