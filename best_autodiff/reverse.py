"""
reverse module
Implements Reverse class: a class to perform the reverse mode of automatic differentiation
"""
import numpy as np
# from best_autodiff.graph import Graph
# from best_autodiff.rfunctions import *
from .graph import Graph
from .rfunctions import *
import warnings
warnings.filterwarnings('ignore')

__all__ = ["Reverse"]

class Reverse:
    """
    Class used to implement Reverse mode of Automatic Differentiation

    Attributes
    ----------
    function_list:
        numPy array or list of functions to evaluate

    input_list:
        numPy array or list for the initialization parameters of the functions' inputs
    
    m:
        scalar (int or float) number of inputs
    
    n:
        scalar (int or float) number of functions (outputs)
    
    function_dict:
        Initialized to None.
        Dictionary that will contain the variables and their local derivatives
    
    jacobian_:
        Initialized to array of zeros of size [n,m] once _get_gradients() is called.

    Methods
    -------
    __init__(self, function_list, input_list, m, n):
        instantiates reverse mode object.
        Transforms inputs into Graph class and evaluates function at given points
        Runs private method _get_gradients()
    
    evaluate(self):
        Returns the primal trace
    
    get_jacobian(self):
        Returns the adjoint trace
    
    Examples
    --------
    R -> R
    >>> def f(x):
    >>>    return sin(x[0])**3 + sqrt(cos(x[0]))
    >>> x = 1
    >>> ad = Reverse(f, 1, 1)
    >>> ad.evaluate(x)
    >>> J = ad.get_jacobian(x)
    >>> J
    0.5753328132280636

    Rm -> R
    >>> def f(x):
    >>>     return (sqrt(x[0])/log(x[1]))*x[0]
    >>> x = [5, 6]
    >>> ad = Reverse(f, 2, 1)
    >>> ad.evaluate(x)
    >>> J = ad.get_jacobian(x)
    >>> J
    array([[ 1.87195995, -0.58042263]])

    R -> Rn
    >>> def f0(x):
    >>>     return x[0]**3 
    >>> def f1(x):
    >>>     return x[0]
    >>> def f2(x):
    >>>     return logistic(x[0])
    >>> f = [f0, f1, f2]
    >>> x = 3
    >>> ad = Reverse(f, 1, 3)
    >>> ad.evaluate(x)
    >>> J = ad.get_jacobian(x)
    >>> J
    array([[27.        ],
       [ 1.        ],
       [ 0.04517666]])
    
    Rm -> Rn
    >>> def f1(x):
    >>>     return exp(-(sin(x[0])-cos(x[1]))**2)
    >>> def f2(x):
    >>>     return sin(-log(x[0])**2+tan(x[2]))
    >>> f = [f1, f2]
    >>> x = [1, 1, 1]
    >>> ad = Reverse(f, 3, 2)
    >>> ad.evaluate(x)
    >>> J = ad.get_jacobian(x)
    >>> J
    array([[-0.29722477, -0.46290015,  0.        ],
       [ 0.        ,  0.        ,  0.04586154]])
    """
    _scalars_types = (int, float, np.int32, np.int64, np.float64)
    _array_types = (list, np.ndarray)

    def __init__(self, function_list, m, n):
        """
        Instantiates reverse mode object

        Parameters
        ----------
        function_list:
            numPy array or list of functions to evaluate

        input_list:
            numPy array or list for the initialization parameters of the functions' inputs
        
        m:
            scalar (int or float) number of inputs
        
        n:
            scalar (int or float) number of functions (outputs)
        
        Raises
        -------
        TypeError if function_list is not of type np.ndarray or list
        TypeError if function_list elements are not callable (i.e functions)
        TypeError if input_list is not of type np.ndarray or list
        TypeError if input_list elements are not int, float, array or list

        Examples
        --------
        >>> def f1(x):
        >>>     return exp(-(sin(x[0])-cos(x[1]))**2)
        >>> def f2(x):
        >>>     return sin(-log(x[0])**2+tan(x[2]))
        >>> f = [f1, f2]
        >>> x = [1, 1, 1]
        >>> ad = Reverse(f, 3, 2)
        """
        # Validate function_list content
        if isinstance(function_list, self._array_types):
            if all([callable(f) for f in function_list]):
                self.function_list = np.array(function_list)
            else:
                raise TypeError(f"Unsupported non callable object")
        elif callable(function_list):
            self.function_list = np.array([function_list])
        else:
            raise TypeError(f"Unsupported type {type(function_list)}")

        # Save number of inputs
        if isinstance(m, self._scalars_types):
            self.m = m
        else:
            raise TypeError(f"Unsupported type {type(m)}")
        
        # Save number of outputs
        if isinstance(n, self._scalars_types):
            self.n = n  
        else:
            raise TypeError(f"Unsupported type {type(n)}")

        # Initial dictionary to store local derivatives
        self.function_dict = None


    def _get_gradients(self):
        """ 
        Estimates the local derivatives of `variable` 
        with respect to child variables.
        Each node and path is stored in function_dict
        """
        # Array to store local derivatives (needs to be reset every time)
        self.jacobian_ = np.zeros((self.n,self.m))

        function_dict = {}

        for f_value in self.function_list_:
            gradients = dict()
            
            def compute_gradients(variable, path_value):
                for child_variable, local_gradient in variable.local_gradients:
                    # Multiply the edges of a path
                    value_of_path_to_child = path_value * local_gradient
                    # Add together the different paths
                    gradients[child_variable] = gradients.get(child_variable, np.array([0]*len(value_of_path_to_child))) + np.array(value_of_path_to_child)
                    # recurse through graph
                    compute_gradients(child_variable, value_of_path_to_child)
                if gradients == {}:
                    gradients[variable] = np.array([1])

            compute_gradients(f_value, path_value=1)
            function_dict[f_value] = gradients
        # (path_value=1 is from `variable` differentiated w.r.t. itself)
        self.function_dict = function_dict
        return self.function_dict
    
    def evaluate(self, input_list):
        """
        Evaluates function_list at given points = x

        Parameters
        ----------
        input_list:
            numPy array or list for the initialization parameters of the functions' inputs

        Rm -> Rn
        >>> def f1(x):
        >>>     return exp(-(sin(x[0])-cos(x[1]))**2)
        >>> def f2(x):
        >>>     return sin(-log(x[0])**2+tan(x[2]))
        >>> f = [f1, f2]
        >>> x = [1, 1, 1]
        >>> ad = Reverse(f, 3, 2)
        >>> ad.evaluate(x)
        array([[0.91328931],
       [0.99991037]])
        """
        # Start assuming vector input
        self._scalar_input=False

        # Validate input_list content
        if isinstance(input_list, self._array_types):
            if all([isinstance(i, (self._scalars_types, self._array_types)) for i in input_list]):
                input_list = np.array(input_list)
            else:
                raise TypeError(f"Unsupported type {type(input_list)}")
        elif isinstance(input_list, self._scalars_types):
            self._scalar_input=True
            input_list = np.array([input_list])
        else:
            raise TypeError(f"Unsupported type {type(input_list)}")

        # Transform evaluation point for each input into Graph class
        self.input_list_ = np.array([Graph(ins) for ins in input_list])

        # Evaluate function at input
        self.function_list_ = [f(self.input_list_) for f in self.function_list]

        # Get gradients
        self._get_gradients()

        # Format evaluated functions from gradients
        values = np.array([k.value for k in self.function_dict.keys()])
        values = values.flatten()

        # Format output for scalar or single element array input type
        if values.size == 1:
            if self._scalar_input:
                values = values.item()
            else:
                values = np.array([values.item()])

        return values
    
    def get_jacobian(self, input_list):
        """
        Returns Jacobian matrix with total derivatives.
        Element [i,j] corresponds to the derivative of function i in functinon_list
        with respect to variable j in input_list

        Rm -> Rn
        >>> def f1(x):
        >>>     return exp(-(sin(x[0])-cos(x[1]))**2)
        >>> def f2(x):
        >>>     return sin(-log(x[0])**2+tan(x[2]))
        >>> f=[f1, f2]
        >>> x = [1, 1, 1]
        >>> ad = Reverse(f, 3, 2)
        >>> ad.evaluate(x)
        >>> ad.get_jacobian(x)
        array([[-0.29722477, -0.46290015,  0.        ],
        [ 0.        ,  0.        ,  0.04586154]])
        """
        self.evaluate(input_list)
        for i in range(self.n):
            for j in range(self.m):
                try:
                    self.jacobian_[i][j] = self.function_dict[self.function_list_[i]][self.input_list_[j]]
                except:
                    self.jacobian_[i][j] = np.array([0])

        # Format output for scalar or single element array input type
        if self.jacobian_.size == 1:
            if self._scalar_input:
                self.jacobian_ = self.jacobian_.item()
            else:
                self.jacobian_ = np.array([self.jacobian_.item()])

        return self.jacobian_
    
    def __call__(self, x):
        """
        Return function evaluated at x and its respective jacobian matrix
        """
        return (self.evaluate(x), self.get_jacobian(x))


# References
# https://sidsite.com/posts/autodiff/


# if __name__ == "__main__":
#     pass