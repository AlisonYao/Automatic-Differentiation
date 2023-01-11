[![.github/workflows/test.yml](https://code.harvard.edu/CS107/team02/actions/workflows/test.yml/badge.svg?branch=main)](https://code.harvard.edu/CS107/team02/actions/workflows/test.yml) [![.github/workflows/coverage.yml](https://code.harvard.edu/CS107/team02/actions/workflows/coverage.yml/badge.svg?branch=main)](https://code.harvard.edu/CS107/team02/actions/workflows/coverage.yml)

# best_autodiff: Python Package for Automatic Differentation
Harvard University AC207 Systems Development (Fall 2022) Final Project

**Team02**: Isha Vaish, Isabella Bossa, Alison Yao, Isidora Diaz

## Overview
Welcome to our python package: **best_autodiff**. 

**best_autodiff** implements the forward and reverse  modes of Automatic Differentiation (AD).

There are multiple methods for calculating derivatives such as Symbolic Differentiation and Finite Differencing. While Symbolic Differentiation is precise, it can be unefficient and costly. On the other hand, Finite Differencing is quick and easy to implement but can have rounding errors. This leads us to Automatic Differentiation which can compute derivatives efficiently and at machine precision.

Automatic Differentiation has a wide range of applications in many computer science and math fields. One such example is the use of Automatic Differentiation in backpropagation for neural networks.

For our package's full documentation, please view the documentation.ipynb in the 'docs' folder.

## Installation
### PyPI Installation
Users can install best_autodiff from [Test Python Package Index](https://test.pypi.org/) (PyPI) following the steps given below:
1. Create a virtual environment where best_autodiff will run (recommended) `python3 -m venv /path/to/new/virtual/environment`
2. Activate virtual environment `source /path/to/new/virtual/environment/bin/activate`
3. `pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple best-autodiff`

Note to developers: unit tests are included in the source distribution at <a href="https://test-files.pythonhosted.org/packages/94/2d/5f6176f07cba032475fea44a1674993fdc0d7636f71e350b9e88bdd8b94b/best_autodiff-0.0.4.tar.gz"> test PyPI </a>, however they won't be installed automatically in the site-packages directory via pip, following best practices to avoid pollution for regular users.

### GitHub Installation

1. Go to the objective directory where you want to install best_autodiff.
2. Clone the repository. Using the command line you can type git clone https://code.harvard.edu/CS107/team02.git
3. Then position yourself into the project root using cd team02
4. Create a virtual environment where best_autodiff will run (recommended) python3 -m venv /path/to/new/virtual/environment
5. Activate virtual environment source /path/to/new/virtual/environment/bin/activate
6. Install pip install -r requirements.txt
3. Import ```best_autodiff``` and run your code. An example is given in the Forward Mode section below.

## How to use
There are two ways to create functions:
```python
def f(x):
    """
    first way
    """
    return x[0] + x[1]

def f(x, y):
    """
    second way
    """
    return x + y
```
Our package only supports the first way. In order to use the second way, the user can use a wrapper function as shown below:
```python
def f(x, y):
    """
    needs to be wrapped
    """
    return x + y

def h(v):
    """
    wrapper for f(x,y)
    """
    return f(*v)

def g(x):
    """
    needs to be wrapped
    """
    return sin(x)

def j(v):
    """
    wrapper for g(x)
    """
    return g(*v)

forward_h = ad.Forward(h, 2, 1)
forward_j = ad.Forward(j, 1, 1)
```

### Forward Mode Demo
Our forward mode can be run on the following cases:
1. 𝑅→𝑅  (scalars, 1 function with 1 input)
2. 𝑅→𝑅𝑛  (vector, multiple functions with 1 input)
3. 𝑅𝑚→𝑅  (vector, 1 function with multiple inputs)
4. 𝑅𝑚→𝑅𝑛  (vectors, multiple functions/outputs with 1 or multiple inputs)

```python
# import packages
import numpy as np
from best_autodiff import Forward
from best_autodiff import functions as ff

## R -> R: scalar case (1 function, 1 input)
def f(x):
    return ff.sin(x[0])

# instantiate forward mode object for function f that has 1 input and 1 output
forward = Forward(f,1,1)

# get value and jacobian of function at x = 0
value, jacobian = forward(0)
print(f'Value: {value}') 
>> Value: 0.0
print(f'Jacobian: {jacobian}') 
>> Jacobian: 1.0


## Rm -> R: vector case (1 function, muliple inputs)
def f(x):
    return x[0]*x[1]

# instantiate forward mode object for function f with multiple inputs
forward = Forward(f,2,1)

# get value and jacobian at x = [4,5]
value, jacobian = forward(np.array([4,5]))

# can also get partial derivatives with seed vectors
partial_x0 = forward.get_jacobian(np.array([4,5]), np.array([1,0])) 
partial_x1 = forward.get_jacobian(np.array([4,5]), np.array([0,1]))

print(f'Value: {value}')
>> Value: [20]
print(f'Jacobian: {jacobian}')
>> Jacobian: [[5 4]]
print(f'Partial derivative of x0: {partial_x0}')
>> [[5 0]]
print(f'Partial derivative of x1: {partial_x1}')
>> [[0 4]]


## Rm -> Rn: multiple functions (multiple functions, 1 or multiple inputs)
def f1(x):
    return ff.sin(x[0])
def f2(x):
    return x[0]**2 + x[1]**2

# instantiate forward mode object for functions [f1,f2] that have multiple inputs x = [x0,x1]
foward = Forward([f1,f2],2,2)

# get function value and jacobian at x = [0,4]
value, jacobian = forward(np.array([0,4]))
print(f'Value:\n {value}')
>> Value: [ 0. 16.]
print(f'Jacobian:\n {jacobian}')
>> Jacobian: 
[[1. 0.]
 [0. 8.]]

# can also get weighted partial derivatives with weighted seed vector
partials_weighted = forward.get_jacobian(np.array([0,4]), np.array([2,1/2])) #uses same seed vector for each function
print(f'Weighted partial derivatives:\n {partials_weighted}')
>> Weighted partial derivatives:
[[2. 0.]
 [0. 4.]]
 
partials_weighted = forward.get_jacobian(np.array([0,4]), np.array([[2,1/2],[1,0]])) #uses different seed vector for each function
print(f'Weighted partial derivatives:\n {partial_x0_weighted}')
>> Weighted partial derivatives:
[[2. 0.]
 [0. 0.]]

```
### Reverse Mode Demo
Our reverse mode can be used for the same cases as forward mode listed above.
```python
import numpy as np
from best_autodiff import Reverse
from best_autodiff import rfunctions as rf

# R -> R
def f(x):
    return rf.sin(x[0])**3 + rf.sqrt(rf.cos(x[0]))
x = 1
reverse = Reverse(f, 1, 1)
value, jacobian = reverse(x)
print(f'Value: {value}')
>> Value: 1.3308758237356713
print(f'Jacobian: {jacobian}')
>> Jacobian: 0.5753328132280636

# R -> Rn
def f0(x):
    return x[0]**3
def f1(x):
    return x[0]
def f2(x):
    return rf.logistic(x[0])

f = [f0, f1, f2]
x = 1
reverse = Reverse(f, 1, 3) # 1 input, 3 outputs
value, jacobian = reverse(x)
print(f'Value: {value}')
>> Value: [1.         1.         0.73105858]
print(f'Jacobian: \n {jacobian}')
>> Jacobian: 
[[3.        ]
 [1.        ]
 [0.19661193]]

# Rm -> R
def f(x):
    return (rf.sqrt(x[0])/rf.log(x[1]))*x[0]
x = [5, 6]
reverse = Reverse(f, 2, 1) # 2 inputs, 1 output
value, jacobian = reverse(x)
print(f'Value: {value}')
>> Value: [6.2398665]
print(f'Jacobian: {jacobian}')
>> Jacobian: [[ 1.87195995 -0.58042263]]

# Rm -> Rn
def f1(x):
    return rf.exp(-(rf.sin(x[0])-rf.cos(x[1]))**2)

def f2(x):
    return rf.sin(-rf.log(x[0])**2+rf.tan(x[2]))
f  = [f1, f2]
x = [1, 1, 1]
reverse = Reverse(f, 3, 2) # 3 inputs, 2 outputs
value, jacobian = reverse(x)
print(f'Value: {value}')
>> [0.91328931 0.99991037]
print(f'Jacobian: \n {jacobian}')
>> Jacobian:
[[-0.29722477 -0.46290015  0.        ]
 [ 0.          0.          0.04586154]]

```
## Broader Impact
best_autodiff calculates derivatives using automatic differentiation (AD), both for forward and reverse modes. Its main advantages are that that it can estimate total derivatives to machine precision with a relatively low computational cost, surpassing other methods such as hand-coded analytical derivatives, as well as numerical and symbolic differentiation. Nevertheless, AD drawbacks should be carefully considered when deciding whether the package would be a good fit for the task at hand. Besides of being very implementation dependant (and thus, prone to development error), AD often acts as a black box. This means that AD will be most relevant if the user is not interested in the derivatives themselves, but is rather looking to use the result for applications in a specific problem or field, particularly science, technology and health. Even if this is the case, we still encourage the users to aim for a basic understanding of AD that will help them further optimize their code and approach the results with a critical eye. As a stand-alone package, we believe that best_autodiff does not pose relevant risks to society, nor compromises their users' data privacy.

## Inclusivity Statement
best_autodiff is an open-source project that welcomes collaborations from every developer, without discriminating on any aspect other than the code itself. The team has made great efforts to guarantee that the project is accessible to download and install, and extensively documented and commented the code itself to ease the first approximation to the modules' internals and allow enough flexibility to create custom features. Extensions and improvements are welcomed through Pull Requests (PR) to the project's GitHub Repository. We are commited to give each PR an equal opportunity, thorough review and promptly reply. Particularly in cases where the PR might be rejected, we will provide feedback and communication channels in order to foster an inclusive bidirectional learning environment. Furthermore, we particularly encourage developers from underrepresented groups in tech and unconventional backgrounds to help improve and extend the best_autodiff library, since their point of view is critical to find applications that will make a positive impact in society as a whole, as well as catch blind spots that may be unintentionally amplifying biases.
