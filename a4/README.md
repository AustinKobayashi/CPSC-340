# l4l0b_o1j0b_a4
## 1
-----
### 1.1
![work](/a4/figs/q1_1.jpg)

### 1.2
![work](/a4/figs/q1_2.jpg)

### 1.3
![work](/a4/figs/q1_3.jpg)

### 1.4
![work](/a4/figs/q1_4.jpg)

### 1.5
![work](/a4/figs/q1_5.jpg)


## 2
-----
### 2.1
[code](/a4/code/linear_model.py) <br>
logRegL2 Training error: 0.002 <br>
logRegL2 Validation error: 0.074 <br>
num features: 101 <br>
iterations: 36


### 2.2
[code](/a4/code/linear_model.py) <br>
logRegL1 Training error: 0.000 <br>
logRegL1 Validation error: 0.052 <br>
num features: 71 <br>
iterations: 78


### 2.3
[code](/a4/code/linear_model.py) <br>
Training error: 0.000 <br>
Validation error: 0.042 <br>
num features: 24 <br>


### 2.4
From the problem description, we know that only prime numbered features are relevant to this dataset.
This means that only using the prime numbered feature will give the best results. Using more features
beyond the prime numbered features will yield worse results. This helps to describe the correlation
between the number of features used and the validation error; The validation error is proportional
to the number of features used. This explains why L0-regularization performs the best (0.042 validation
error), since it puts a penalty on the number of features used. L2-regularization performs the worst (0.074
validation error) since it puts a penalty on the L2-norm of w which causes it to uses a lot of features (101).
L1-regularization lies in between L0 and L2-regularization, at 0.052 validation error, since it performs
feature selection, but still uses more features than L0-regularization (78 vs 24)


### 2.5
[code](/a4/code/main.py)

L2: <br>
Training error: 0.002 <br>
Validation error: 0.074 <br>
num features: 101 <br>

L1: <br>
Training error: 0.000 <br>
Validation error: 0.052 <br>
num features: 71 <br>

Our training error, validation error, and the number of features used are identical for scikit-learn's L1 and
L2-regularization


## 4 
-----

### 4.1
Simply using validation error will lead to overcomplexity and in turn overfitting, while BIC (and AIC) penalize complexity.

### 4.2
Compared to an exhaustive search, forward selection will be cheaper, avoids overfitting, and can lead to fewer false positives.

### 4.3
In L2-regularization, greater λ will increase training error but decrease approximation error.

### 4.4

### 4.5
Using least-squares on binary classification will give non-zero error scores for correct predictions and can give huge errors for incorrect predictions.

### 4.6

### 4.7

### 4.8
Multi-class classification predicts just one label for a given datum, while multi-label classification can potentially predict multiple labels.

### 4.9
For one-vs-all multi-classs logistic regression, we are solving **k** optimization problem(s) of dimension **n**. On the other hand, for softmax logistic regression, we are solving **k** optimization problem(s) of dimension **d**.