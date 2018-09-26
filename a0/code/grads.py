import numpy as np
from scipy.optimize import approx_fprime


def example(x):
    return np.sum(x**2)

def example_grad(x):
    return 2*x

def foo(x):
    result = 1
    λ = 4 # this is here to make sure you're using Python 3
    for x_i in x:
        result += x_i**λ    
    return result
    
def foo_grad(x):
    return 4*x**3

def bar(x):
    return np.prod(x)

def bar_grad(x):
    return np.prod(x) / x


# here is some code to test your answers
# below we test out example_grad using scipy.optimize.approx_fprime,
# which approximates gradients.
# if you want, you can use this to test out your foo_grad and bar_grad

def check_grad(fun, grad):
    x0 = np.random.rand(5) # take a random x-vector just for testing
    diff = approx_fprime(x0, fun, 1e-4)  # don't worry about the 1e-4 for now
    print("\n** %s **" % fun.__name__)
    print("My gradient     : %s" % grad(x0))
    print("Scipy's gradient: %s" % diff)

check_grad(example, example_grad)
check_grad(foo, foo_grad)
check_grad(bar, bar_grad)

# if you run this file with "python grads.py" the code above will run.

