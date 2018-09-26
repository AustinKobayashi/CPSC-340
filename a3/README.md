# l4l0b_o1j0b_a3

## 1
-----

### 1.1
1. 13
2. 13
3. -37

4. | -4 |  
| -5 |  
| 12 |  

5. 4

6. | 14 2 |  
| 2 17 |

![work](figs/q1_1.jpg)

7.  True
8.  True
9.  False
10. True
11. False
12. True

### 1.2
![work](figs/q1_2.jpg)

### 1.3
![work](figs/q1_3.jpg)

## 2
-----
### 2.1
[code](code/linear_model.py)
![plot](figs/least_squares_outliers_weighted.png)

### 2.2
![work](figs/q2_2.jpg)

### 2.3
[code](code/linear_model.py)
 ![plot](figs/least_squares_robust.png)

## 3
-----

### 3.1
Training error = 3551.3  
Test error     = 3393.9  
[code](code/linear_model.py)
![plot](figs/least_squares_bias.png)
	
### 3.2
p = 0  
Training error = 15480.5  
Test error     = 14390.8  
Approx. error  = -1089.757  

p = 1  
Training error = 3551.3  
Test error     = 3393.9  
Approx. error  = -157.477  

p = 2  
Training error = 2168.0  
Test error     = 2480.7  
Approx. error  = 312.733  

p = 3  
Training error = 252.0  
Test error     = 242.8  
Approx. error  = -9.241  

p = 4  
Training error = 251.5  
Test error     = 242.1  
Approx. error  = -9.335  

p = 5  
Training error = 251.1  
Test error     = 239.5  
Approx. error  = -11.599  

p = 6  
Training error = 248.6  
Test error     = 246.0  
Approx. error  = -2.577  

p = 7  
Training error = 247.0  
Test error     = 242.9  
Approx. error  = -4.123  

p = 8  
Training error = 241.3  
Test error     = 246.0  
Approx. error  = 4.660  

p = 9  
Training error = 235.8  
Test error     = 259.3  
Approx. error  = 23.534  

p = 10  
Training error = 235.1  
Test error     = 256.3  
Approx. error  = 21.226  

We can see from the results that p = 0 performs increadibly poorly since it is a horizontal line at the y-intercept.
![plot](figs/PolyBasis0.png)

p = 1 gives us the same result as our LeastSquaresBias. 
![plot](figs/PolyBasis1.png)

The training error decreases as p increase, with the lowest value being 235.1. The test error decreases between 
p = 0 and p = 5, from 14390.8 to 239.5, then it increases to 246.0 at p = 6. At p = 6, our training error approximates the test error the closest
for all values of p, with an approximation error of -2.577. 
![plot](figs/PolyBasis2.png)
![plot](figs/PolyBasis3.png)
![plot](figs/PolyBasis4.png)
![plot](figs/PolyBasis5.png)
![plot](figs/PolyBasis6.png)

At p = 6 our model starts overfitting and the test error starts to increase.
![plot](figs/PolyBasis7.png)
![plot](figs/PolyBasis8.png)
![plot](figs/PolyBasis9.png)
![plot](figs/PolyBasis10.png)
	
	


## 4: Non-Parametric Bases and Cross-Validation
-----

### 4.1

When running `python main.py -q 4`, we get rather large errors:

```
Training error = 2184.1
Test error     = 2495.9
Approx. error  = 311.835
```

And the plot:
![least_squares_rbf_bad](figs/least_squares_rbf_bad.png)

After modifying to randomize the validation split, in one instance we get:

```
Training error = 234.1
Test error     = 256.8
Approx. error  = 22.725
```

And the plot:
![least_squares_rbf_better](figs/least_squares_rbf_better.png)

Both the errors and the plot appear to improve significantly as a result of this change. Errors are an order of magnitude smaller and visually the plotted model appears to be good.

### 4.2

1. Running the code repeatedly most often yields σ=1. The occasional run has yielded σ=32 or σ=4, but those do not repeat.

2. [The code exists.](code/main.py#L257) This has produced σ=1 every time it has run.

### 4.3
For fixed σ, training under the linear basis has complexity O(nd^2 + d^3). Training using Gaussian RBF's has complexity O(n^3).  Classifying t examples using the linear basis has a cost O(dt), while using the Gaussian RBF has cost O(dtn). RBF's are cheaper to train and test on smaller training datasets.




# 5 Very-Short Answer Questions

1. Computing the squared error allows us to see how closely our prefiction fits the data. Testing equality would show if our prediction is equal to point i. To lower our error if we tested off equality would mean that we would need to pass through more data points, which is not optimal since the "best model" could have noise where yi != yi (hat) for all i.

2. d = 2 means that our data for the situation has 2 features. If these features are identical for all n examples, in this case n=4, thenthese features are collinear. This means that we can increase the weight on 1 feature and decrease the weight on the other feature without changing our predictions. Therefore, the solution is not unique. 

3. For a polynomial basis of degree p, we must create a new matrix Z of dimensions (n, p+1) where there first column of Z is a row of 1's and the other columns of Z are equal to X. This will take O(n*(p+1)) since we must copy all n values of X p times. Forming the matrix Z^T * Z takes O(n * (p+1)^2) since Z^T * Z has (p+1)^2 elements and each is a sum of n number. Solving the system Z^T * Z * w = Z^T * y costs O((p+1)^3). Overall the cost is O(n(p+1) + n(p+1)^2 + (p+1)^3).

4. A situation with very multi-modal data might be a good target for regression trees with linear regressions at the leaves. By decreasing the spread of each dataset we need to fit a linear regression to, we can improve our precision.

5. When our function is convex, it means a gradient of 0 is the global minimum. This makes it easier to find the "best model" since setting the gradient to 0 and finding w gives the global minimum.

6. For convex functions with large d, gradient descent can be computationally faster than solving the closed form solution. We can also control the accuracy of our model when using gradient descent by controlling the number of iterations. Gradient descent can also be used to solve non convex functions.

7. Optimization can very easily lead to overfitting of the training data. Deciding on and using the right complexity costs makes the problem more complex. Additionally, we need to be able to ensure that features that have large impact on our model are features that are relevant.

8. For robust regression, least squares doesn't give equal "weight" to every point and its proximity to the line. Least squares will disproportionally favor outliers' distances to the line when trying to create a best fit. Gradient descent instead gives the same weight to every point, meaning that it prioritizes the needs of the many over the needs of the few.

9. Too small of a learning rate means that we will converge on the optimal solution slower. This means that we will require more iterations, which is make the algorithm much slower.

10. If our learning rate is too large, we may pass over the optimal solution in an interation.