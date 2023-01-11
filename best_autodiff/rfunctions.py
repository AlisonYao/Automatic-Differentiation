"""
rfunctions module
Implements mathematical functions for the best_autodiff package
Functions are intended to work with graph instances

sqrt(z)
	Implements sqrt function of a Graph instance

log(z, base)
	Implements log function of a Graph instance using any base

root(z, n)
	Implements function to take the nth root of a Graph instance

exp(z)
	Implements exp function of a Graph instance using any base

sin(z)
	Implements sine of a Graph instance

cos(z)
	Implements cos function of a Graph instance

tan(z)
	Implements tangent of a Graph instance

arcsin(z)
	Implements arcsin function of a Graph instance

arccos(z)
	Implements arccos function of a Graph instance

arctan(z)
	Implements arctangent of a Graph instance

sinh(z)
	Implements sinh function of a Graph instance

cosh(z)
	Implements cosh function of a Graph instance

tanh(z)
	Implements tanh function of a Graph instance

logistic(z)
	Implements logistic function of a Graph instance
"""

import numpy as np
from best_autodiff.graph import Graph

__all__ = ["sqrt", "log", "root", "exp", "sin", "cos", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "logistic"]

_scalars_types = (int, float, np.int32, np.int64, np.float64)
_array_types = (list, np.ndarray)

def sqrt(z):
    """ 
    Implements sqrt function of a Graph instance

	Parameters
	----------
	z: Graph
	base: int, float, np.int32, np.int64 or np.float64

	Raises
	------
	TypeError
		If z is not Graph
    ValueError
        If z < 0

    Examples
    --------
    >>> z = Graph([4, 25])
    >>> sqrt(z)
    Graph([2. 5.])
	"""
    if not isinstance(z, Graph):
        raise TypeError(f"Unsupported type {type(z)}")
    if (z.value < 0).any():
        raise ValueError(f"Domain out of range")
    else:
        value = np.sqrt(z.value)
        local_gradients = (
            (z, (1/2)*(z.value**(-1/2))),
        )
        return Graph(value, local_gradients)

def log(z, base=np.e):
    """ 
    Implements log function of a Graph instance using any base

	Parameters
	----------
	z: Graph
	base: int, float, np.int32, np.int64 or np.float64

	Raises
	------
	TypeError
		If z is not Graph
		If base is not int, float, np.int32, np.int64 or np.float64

    Examples
    --------
    >>> z = Graph(23)
    >>> log(z)
    Graph([3.13549422])
    >>> z = Graph([1,23])
    >>> log(z, 123)
    Graph([0.       0.651574])
	"""
    if not isinstance(z, Graph):
        raise TypeError(f"Unsupported type {type(z)}")
    if not isinstance(base, _scalars_types):
        raise TypeError(f"Unsupported type {type(base)}")
    value = np.log(z.value) / np.log(base)
    local_gradients = (
        (z, 1. / (z.value * np.log(base))),
    )
    return Graph(value, local_gradients)

def root(z, n=2):
    """ 
    Implements function to take the nth root of a Graph instance

	Parameters
	----------
	z: Graph
	n: int, float, np.int32, np.int64 or np.float64
		nth root of a Graph instance

	Raises
	------
	TypeError
		If z is not Graph
		If n is not int, float, np.int32, np.int64 or np.float64

    Examples
    --------
    >>> z = Graph([1, 2, 3])
    >>> root(z) # same as sqrt
    Graph([1.         1.41421356 1.73205081])
    >>> z = Graph(5)
    >>> root(z, 10)
    Graph([1.17461894])
	"""
    if not isinstance(z, Graph):
        raise TypeError(f"Unsupported type {type(z)}")
    if not isinstance(n, _scalars_types):
        raise TypeError(f"Unsupported type {type(n)}")
    if (z.value < 0).any():
        raise ValueError(f"Domain out of range")
    value = z.value ** (1/n)
    local_gradients = (
        (z, (1/n) * ((z.value)**(1/n - 1))),
    )
    return Graph(value, local_gradients)

def exp(z):
    """ 
    Implements exp function of a Graph instance using any base

	Parameters
	----------
	z: Graph
	base: int, float, np.int32, np.int64 or np.float64

	Raises
	------
	TypeError
		If z is not Graph
		If base is not int, float, np.int32, np.int64 or np.float64
    ValueError
        If base is negative

    Examples
    --------
    >>> z = Graph([2, 3])
    >>> exp(z, base=10)
    Graph([ 100 1000])
	"""

    if not isinstance(z, Graph):
        raise TypeError(f"Unsupported type {type(z)}")
    value = np.power(np.e, z.value)
    local_gradients = (
        (z, value),
    )
    return Graph(value, local_gradients)

def sin(z):
    """
    Implements sine of a Graph instance

	Parameters
	----------
	z: Graph

	Raises
	------
	TypeError
		If z is not a Graph instance

    Examples
    --------
    >>> z = Graph(7)
    >>> sin(z)
    Graph([0.6569866])
    >>> z = Graph([1, 2, 3])
    >>> sin(z)
    Graph([0.84147098 0.90929743 0.14112001])
    """
    if not isinstance(z, Graph):
        raise TypeError(f"Unsupported type {type(z)}")
    value = np.sin(z.value)
    local_gradients = (
        (z, np.cos(z.value)),
    )
    return Graph(value, local_gradients)

def cos(z):
    """ 
    Implements cos function of a Graph instance

	Parameters
	----------
	z: Graph

	Raises
	------
	TypeError
		If z is not Graph

    Examples
    --------
    >>> z = Graph(7)
    >>> cos(z)
    Graph([0.75390225])
    >>> z = Graph(0)
    >>> cos(z)
    Graph([1.])
	"""
    if not isinstance(z, Graph):
        raise TypeError(f"Unsupported type {type(z)}")

    value = np.cos(z.value)
    local_gradients = (
        (z, -np.sin(z.value)),
    )
    return Graph(value, local_gradients)

def tan(z):
    """
    Implements tangent of a Graph instance

	Parameters
	----------
	z: Graph

	Raises
	------
	TypeError
		If z is not a Graph instance
    
    Examples
    --------
    >>> z = Graph(7)
    >>> tan(z)
    Graph([0.87144798])
    >>> z = Graph([1, 2, 3])
    >>> tan(z)
    Graph([ 1.55740772 -2.18503986 -0.14254654])
    """
    if not isinstance(z, Graph):
        raise TypeError(f"Unsupported type {type(z)}")
    for i in z.value:
        if (i / (np.pi/2)) % 2 == 1:
            raise ValueError('Tangent domain error')
    value = np.tan(z.value)
    local_gradients = (
        (z, 1/np.cos(z.value)**2),
    )
    return Graph(value, local_gradients)

def arcsin(z):
    """ 
    Implements arcsin function of a Graph instance

	Parameters
	----------
	z: Graph

	Raises
	------
	TypeError
		If z is not Graph
    ValueError
        If z.value elements are not between -1 and 1

    Examples
    --------
    >>> z = Graph(0.9)
    >>> arcsin(z)
    Graph([1.11976951])
    >>> z = Graph(-0.5)
    >>> arcsin(z)
    Graph([-0.52359878])
	"""
    if not isinstance(z, Graph):
        raise TypeError(f"Unsupported type {type(z)}")
    if (z.value > 1).all() or (z.value < -1).all():
        raise ValueError(f"math domain error")

    value = np.arcsin(z.value)
    local_gradients = (
        (z, 1/(np.sqrt(1-z.value**2))),
    )
    return Graph(value, local_gradients)

def arccos(z):
    """ 
    Implements arccos function of a Graph instance

	Parameters
	----------
	z: Graph

	Raises
	------
	TypeError
		If z is not Graph
    ValueError
        If z.value elements are not between -1 and 1

    Examples
    --------
    >>> z = Graph(0.9)
    >>> arccos(z)
    Graph([0.45102681])
    >>> z = Graph(-0.5)
    >>> arccos(z)
    Graph([2.0943951])
	"""
    if not isinstance(z, Graph):
        raise TypeError(f"Unsupported type {type(z)}")
    if (z.value > 1).all() or (z.value < -1).all():
        raise ValueError(f"math domain error")

    value = np.arccos(z.value)
    local_gradients = (
        (z, -1/(np.sqrt(1-z.value**2))),
    )
    return Graph(value, local_gradients)

def arctan(z):
    """
    Implements arctangent of a Graph instance

	Parameters
	----------
	z: Graph

	Raises
	------
	TypeError
		If z is not a Graph instance
    
    Examples
    --------
    >>> z = Graph(7)
    >>> arctan(z)
    Graph([1.42889927])
    >>> z = Graph([1, 2, 3])
    >>> arctan(z)
    Graph([0.78539816 1.10714872 1.24904577])
    """
    if not isinstance(z, Graph):
        raise TypeError(f"Unsupported type {type(z)}")
    value = np.arctan(z.value)
    local_gradients = (
        (z, 1/(1 + z.value**2)),
    )
    return Graph(value, local_gradients)

def sinh(z):
    """ 
    Implements sinh function of a Graph instance

	Parameters
	----------
	z: Graph

	Raises
	------
	TypeError
		If z is not Graph

    Examples
    --------
    >>> z = Graph(0.9)
    >>> sinh(z)
    Graph([1.02651673])
    >>> z = Graph(-0.5)
    >>> sinh(z)
    Graph([-0.52109531]) 
	"""
    if not isinstance(z, Graph):
        raise TypeError(f"Unsupported type {type(z)}")

    value = np.sinh(z.value)
    local_gradients = (
        (z, np.cosh(z.value)),
    )
    return Graph(value, local_gradients)

def cosh(z):
    """
    Implements cosh function of a Graph instance

	Parameters
	----------
	z: Graph

	Raises
	------
	TypeError
		If z is not a Graph instance
    
    Examples
    --------
    >>> z = Graph(7)
    >>> cosh(z)
    Graph([548.31703516])
    >>> z = Graph([0,1,2])
    >>> cosh(z)
    Graph([1.         1.54308063 3.76219569])
    """
    if not isinstance(z, Graph):
        raise TypeError(f"Unsupported type {type(z)}")
    value = np.cosh(z.value)
    local_gradients = (
        (z, np.sinh(z.value)),
    )
    return Graph(value, local_gradients)

def tanh(z):
    """ 
    Implements tanh function of a Graph instance

	Parameters
	----------
	z: Graph

	Raises
	------
	TypeError
		If z is not Graph

    Examples
    --------
    >>> z = Graph(0.9)
    >>> tanh(z)
    Graph([0.71629787])
    >>> z = Graph(-0.5)
    >>> tanh(z)
    Graph([-0.46211716]) 
	"""
    if not isinstance(z, Graph):
        raise TypeError(f"Unsupported type {type(z)}")

    value = np.tanh(z.value)
    local_gradients = (
        (z, 1/(np.cosh(z.value)**2)),
    )
    return Graph(value, local_gradients)

def logistic(z):
    """
    Implements logistic function of a Graph instance

	Parameters
	----------
	z: Graph

	Raises
	------
	TypeError
		If z is not a Graph instance
    
    Examples
    --------
    >>> z = Graph(10)
    >>> logistic(z)
    Graph([0.9999546])
    >>> z = Graph(-0.1)
    >>> logistic(z)
    Graph([0.47502081])
    """
    if not isinstance(z, Graph):
        raise TypeError(f"Unsupported type {type(z)}")
    value = 1/(1+np.exp(-z.value))
    local_gradients = (
        (z, np.exp(-z.value)/((1+np.exp(-z.value))**2)),
    )
    return Graph(value, local_gradients)