"""
graph module
Implements Graph: a class to represent computational graph for reverse mode
"""

import numpy as np

class Graph:
    """
    A class used to represent computational graph

	Attributes
	---------
	value: int, float, np.int32, np.int64, np.float64, list or np.ndarray

	Methods
	-------
	__add__(other)
	    Overloads + operator to support Graph instances addition with other Graph instances, ints, floats, lists and numpy arrays

	__radd__(other)
	    Overloads reflected + operator to support Graph instances addition with other Graph instances, ints, floats, lists and numpy arrays

	__mul__(other)
	    Overloads * operator to support Graph instances multiplication with other Graph instances, ints, floats, lists and numpy arrays

	__rmul__(other)
	    Overloads reflected * operator to support Graph instances multiplication with other Graph instances, ints, floats, lists and numpy arrays

	__sub__(other)
	    Overloads - operator to support Graph instances subtraction with other Graph instances, ints, floats, lists and numpy arrays

	__rsub__(other)
	    Overloads reflected - operator to support Graph instances subtraction with other Graph instances, ints, floats, lists and numpy arrays

	__truediv__(other)
		Overloads / operator to support Graph instances division with other Graph instances, ints, floats, lists and numpy arrays

	__rtruediv__(other)
	    Overloads reflected / operator to support Graph instances division with other Graph instances, ints, floats, lists and numpy arrays

	__pow__(other)
	    Overloads ** operator to support exponentiation of graph instances
	
	__rpow__(other)
	    Overloads reverse ** operator to support reflexive exponentiation of graph instances
    
    __neg__()
	    Overloads negation operator to support graph instance negation

    __lt__():
        Overloads < operator to support checking less than for Graph instances
    
    __gt__():
        Overloads > operator to support checking greater than for Graph instances
    
    __le__():
        Overloads <= operator to support checking less or equal than for Graph instances
    
    __ge__():
        Overloads >= operator to support checking greater or equal than for Graph instances

	__eq__():
		Overloads == operator to support checking Graph instances equality
    
    __hash__():
        Complements __eq__() function to ensure Graph instances are hashable
    
    __ne__():
        Overloads != operator to support checking Graph instances inequality

	__str__()
		Returns string representation of graph

	__repr__()
		Returns object representation of graph in string format
    """
    _scalars_types = (int, float, np.int32, np.int64, np.float64)
    _array_types = (list, np.ndarray)

    def __init__(self, value, local_gradients=()):
        """
        Instantiates graph instance

        Parameters
        ----------
        value: int, float, np.int32, np.int64, np.float64, list, or np.ndarray to evaluate

        local_gradients: tuple of local gradients. If not given, the default is ()
        
        Raises
        -------
        TypeError 
            If value is not int, float, np.int32, np.int64, np.float64, list, or np.ndarray
        """
        if isinstance(value, self._scalars_types):
            self.value = np.array([value])
        elif isinstance(value, self._array_types):
            for item in value:
                if not isinstance(item, self._scalars_types):
                    raise TypeError(f"Unsupported type {type(item)} in list/numpy ndarray")
            self.value = np.array(value)
        else:
            raise TypeError(f"Unsupported type {type(value)}")
        self.local_gradients = local_gradients
    
    def __add__(self, other):
        """ 
        Overloads + operator to support Graph instances addition with other Graph instances, ints, floats, lists and numpy arrays

		Parameters
		----------
		other: int, float, np.int32, np.int64, np.float64,list, np.ndarray, or Graph

		Raises
		------
		TypeError
		    If other is not of supported types
		"""
        if isinstance(other, self._scalars_types):
            other = Graph([other]*len(self.value))
        elif isinstance(other, self._array_types):
            assert len(self.value) == len(other), "Input dimension mismatch"
            other = Graph(np.array(other))
        elif isinstance(other, Graph):
            assert len(self.value) == len(other.value), "Input dimension mismatch"
        else:
            raise TypeError(f"Unsupported type {type(other)}")
        
        value = self.value + other.value
        local_gradients = ((self, np.array([1]*len(self.value))), (other, np.array([1]*len(other.value))))
        
        return Graph(value, local_gradients)
    
    def __radd__(self, other):
        """ 
        Overloads reflected + operator to support Graph instances addition with other Graph instances, ints, floats, lists and numpy arrays

		Parameters
		----------
		other: int, float, np.int32, np.int64, np.float64,list, np.ndarray, or Graph

		Raises
		------
		TypeError
		    If other is not of supported types
		"""
        if not isinstance(other, (*self._scalars_types, *self._array_types, Graph)):
            raise TypeError(f"Unsupported type {type(other)}")
        
        return self.__add__(other)

    def __sub__(self, other):
        """
        Overloads - operator to support Graph instances subtraction with other Graph instances, ints, floats, lists and numpy arrays

		Parameters
		----------
		other: int, float, np.int32, np.int64, np.float64, list, np.ndarray or Graph to subtract from Graph instance

		Raises
		------
		TypeError
		    If other is not of type int, float, np.int32, np.int64, np.float64, list, np.ndarray or Graph
        AssertionError
            If other is a list, numpy array, or a Graph instance, but the length doesn't match with self
        """
        if isinstance(other, self._scalars_types):
            other = Graph([other]*len(self.value))
        elif isinstance(other, self._array_types):
            assert len(self.value) == len(other), "Input dimension mismatch"
            other = Graph(np.array(other))
        elif isinstance(other, Graph):
            assert len(self.value) == len(other.value), "Input dimension mismatch"
        else:
            raise TypeError(f"Unsupported type {type(other)}")
        value = self.value - other.value
        local_gradients = ((self, np.array([1]*len(value))), (other, np.array([-1]*len(value))))
        return Graph(value, local_gradients)

    def __rsub__(self, other):
        """
        Overloads reflected - operator to support Graph instances subtraction with other Graph instances, ints, floats, lists and numpy arrays

		Parameters
		----------
		other: int, float, np.int32, np.int64, np.float64, list, np.ndarray or Graph to be subtracted from

		Raises
		------
		TypeError
		    If other is not of type int, float, np.int32, np.int64, np.float64, list, np.ndarray or Graph
        AssertionError
            If other is a list, numpy array, or a Graph instance, but the length doesn't match with self
        """
        if isinstance(other, self._scalars_types):
            other = Graph([other]*len(self.value))
        elif isinstance(other, self._array_types):
            assert len(self.value) == len(other), "Input dimension mismatch"
            other = Graph(np.array(other))
        elif isinstance(other, Graph):
            assert len(self.value) == len(other.value), "Input dimension mismatch"
        else:
            raise TypeError(f"Unsupported type {type(other)}")
        value = other.value - self.value
        local_gradients = ((self, np.array([-1]*len(value))), (other, np.array([1]*len(value))))
        return Graph(value, local_gradients)
    
    def __mul__(self, other):
        """ 
        Overloads * operator to support Graph instances multiplication with other Graph instances, ints, floats, lists and numpy arrays

		Parameters
		----------
		other: int, float, np.int32, np.int64, np.float64,list, np.ndarray, or Graph

		Raises
		------
		TypeError
		    If other is not of supported types
		"""
        if isinstance(other, self._scalars_types):
            other = Graph([other]*len(self.value))
        elif isinstance(other, self._array_types):
            assert len(self.value) == len(other), "Input dimension mismatch"
            other = Graph(np.array(other))
        elif isinstance(other, Graph):
            assert len(self.value) == len(other.value), "Input dimension mismatch"
        else:
            raise TypeError(f"Unsupported type {type(other)}")
        value = self.value * other.value
        local_gradients = (
            (self, other.value),
            (other, self.value)
            )
        return Graph(value, local_gradients)
    
    def __rmul__(self, other):
        """ 
        Overloads reflected * operator to support Graph instances multiplication with other Graph instances, ints, floats, lists and numpy arrays

		Parameters
		----------
		other: int, float, np.int32, np.int64, np.float64,list, np.ndarray, or Graph

		Raises
		------
		TypeError
		    If other is not of supported types
		"""
        if not isinstance(other, (*self._scalars_types, *self._array_types, Graph)):
            raise TypeError(f"Unsupported type {type(other)}")
        
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        Overloads / operator to support Graph instances division with other Graph instances, ints, floats, lists and numpy arrays

		Parameters
		----------
		other: int, float, np.int32, np.int64, np.float64, list, np.ndarray or Graph to divide from Graph instance

		Raises
		------
		TypeError
		    If other is not of type int, float, np.int32, np.int64, np.float64, list, np.ndarray or Graph
        AssertionError
            If other is a list, numpy array, or a Graph instance, but the length doesn't match with self
        """
        if isinstance(other, self._scalars_types):
            other = Graph([other]*len(self.value))
        elif isinstance(other, self._array_types):
            assert len(self.value) == len(other), "Input dimension mismatch"
            other = Graph(np.array(other))
        elif isinstance(other, Graph):
            assert len(self.value) == len(other.value), "Input dimension mismatch"
        else:
            raise TypeError(f"Unsupported type {type(other)}")
        if 0 in other.value:
            raise ZeroDivisionError
        value = self.value / other.value
        local_gradients = (
            (self, 1 / other.value), 
            (other, -self.value/ other.value**2)
        )
        return Graph(value, local_gradients)
    
    def __rtruediv__(self, other):
        """
        Overloads reflected / operator to support Graph instances division with other Graph instances, ints, floats, lists and numpy arrays

		Parameters
		----------
		other: int, float, np.int32, np.int64, np.float64, list, np.ndarray or Graph to divide from Graph instance

		Raises
		------
		TypeError
		    If other is not of type int, float, np.int32, np.int64, np.float64, list, np.ndarray or Graph
        AssertionError
            If other is a list, numpy array, or a Graph instance, but the length doesn't match with self
        """
        if isinstance(other, self._scalars_types):
            other = Graph([other]*len(self.value))
        elif isinstance(other, self._array_types):
            assert len(self.value) == len(other), "Input dimension mismatch"
            other = Graph(np.array(other))
        elif isinstance(other, Graph):
            assert len(self.value) == len(other.value), "Input dimension mismatch"
        else:
            raise TypeError(f"Unsupported type {type(other)}")
        if 0 in self.value:
            raise ZeroDivisionError
        value = other.value / self.value
        local_gradients = (
            (other, 1 / self.value), 
            (self, -other.value/ self.value**2)
        )
        return Graph(value, local_gradients)

    def __pow__(self, other):
        """ 
        Overloads ** operator to support exponentiation of graph instances

		Parameters
		----------
		other: int, float, np.int32, np.int64, np.float64,list, np.ndarray, or Graph

		Raises
		------
		TypeError
		    If other is not of supported types
		"""
        if isinstance(other, self._scalars_types):
            other = Graph([other]*len(self.value))
        elif isinstance(other, self._array_types):
            assert len(self.value) == len(other), "Input dimension mismatch"
            other = Graph(np.array(other))
        elif isinstance(other, Graph):
            assert len(self.value) == len(other.value), "Input dimension mismatch"
        else:
            raise TypeError(f"Unsupported type {type(other)}")

        value = self.value ** other.value
        local_gradients = (
            (self, other.value * self.value ** (other.value - 1)),
            (other, self.value ** other.value * np.log(self.value))
        )
        return Graph(value, local_gradients)

    def __rpow__(self, other):
        """ 
        Overloads reverse ** operator to support reflexive exponentiation of graph instances

		Parameters
		----------
		other: int, float, np.int32, np.int64, np.float64,list, np.ndarray, or Graph

		Raises
		------
		TypeError
		    If other is not of supported types
		"""
        if isinstance(other, self._scalars_types):
            other = Graph([other]*len(self.value))
        elif isinstance(other, self._array_types):
            assert len(self.value) == len(other), "Input dimension mismatch"
            other = Graph(np.array(other))
        elif isinstance(other, Graph):
            assert len(self.value) == len(other.value), "Input dimension mismatch"
        else:
            raise TypeError(f"Unsupported type {type(other)}")
 
        return other.__pow__(self)
    
    def __neg__(self):
        """
        Overloads negation operator to support graph instance negation
        """
        value = -self.value
        local_gradients = (
            (self, np.array([-1]*len(self.value))),
            )
        return Graph(value, local_gradients)

    def __lt__(self, other):
        """
        Overloads < operator to support checking less than for Graph instances

        Parameters
		----------
		other: int, float, np.int32, np.int64, np.float64, list, np.ndarray or Graph

        Raises:
		------
		TypeError
		    If other is not of type int, float, np.int32, np.int64, np.float64, list, np.ndarray or Graph
        AssertionError
            If other is a list, numpy array, or a Graph instance, but the length doesn't match with self
        """
        if isinstance(other, self._scalars_types):
            other = Graph([other]*len(self.value))
        elif isinstance(other, self._array_types):
            assert len(self.value) == len(other), "Input dimension mismatch"
            other = Graph(np.array(other))
        elif isinstance(other, Graph):
            assert len(self.value) == len(other.value), "Input dimension mismatch"
        else:
            raise TypeError(f"Unsupported type {type(other)}")
        return np.less(self.value, other.value)

    def __gt__(self, other):
        """
        Overloads > operator to support checking greater than for Graph instances

        Parameters
		----------
		other: int, float, np.int32, np.int64, np.float64, list, np.ndarray or Graph

        Raises:
		------
		TypeError
		    If other is not of type int, float, np.int32, np.int64, np.float64, list, np.ndarray or Graph
        AssertionError
            If other is a list, numpy array, or a Graph instance, but the length doesn't match with self
        """
        if isinstance(other, self._scalars_types):
            other = Graph([other]*len(self.value))
        elif isinstance(other, self._array_types):
            assert len(self.value) == len(other), "Input dimension mismatch"
            other = Graph(np.array(other))
        elif isinstance(other, Graph):
            assert len(self.value) == len(other.value), "Input dimension mismatch"
        else:
            raise TypeError(f"Unsupported type {type(other)}")
        return np.greater(self.value, other.value)

    def __le__(self, other):
        """
        Overloads <= operator to support checking less or equal than for Graph instances

        Parameters
		----------
		other: int, float, np.int32, np.int64, np.float64, list, np.ndarray or Graph

        Raises:
		------
		TypeError
		    If other is not of type int, float, np.int32, np.int64, np.float64, list, np.ndarray or Graph
        AssertionError
            If other is a list, numpy array, or a Graph instance, but the length doesn't match with self
        """
        if isinstance(other, self._scalars_types):
            other = Graph([other]*len(self.value))
        elif isinstance(other, self._array_types):
            assert len(self.value) == len(other), "Input dimension mismatch"
            other = Graph(np.array(other))
        elif isinstance(other, Graph):
            assert len(self.value) == len(other.value), "Input dimension mismatch"
        else:
            raise TypeError(f"Unsupported type {type(other)}")
        return np.less_equal(self.value, other.value) 

    def __ge__(self, other):
        """
        Overloads >= operator to support checking greater or equal than for Graph instances

        Parameters
		----------
		other: int, float, np.int32, np.int64, np.float64, list, np.ndarray or Graph

        Raises:
		------
		TypeError
		    If other is not of type int, float, np.int32, np.int64, np.float64, list, np.ndarray or Graph
        AssertionError
            If other is a list, numpy array, or a Graph instance, but the length doesn't match with self
        """
        if isinstance(other, self._scalars_types):
            other = Graph([other]*len(self.value))
        elif isinstance(other, self._array_types):
            assert len(self.value) == len(other), "Input dimension mismatch"
            other = Graph(np.array(other))
        elif isinstance(other, Graph):
            assert len(self.value) == len(other.value), "Input dimension mismatch"
        else:
            raise TypeError(f"Unsupported type {type(other)}")
        return np.greater_equal(self.value, other.value) 

    def __eq__(self, other):
        """
        Overloads == operator to support checking Graph instances equality

        Parameters
		----------
		other: int, float, np.int32, np.int64, np.float64, list, np.ndarray or Graph

        Raises:
		------
		TypeError
		    If other is not of type int, float, np.int32, np.int64, np.float64, list, np.ndarray or Graph
        AssertionError
            If other is a list, numpy array, or a Graph instance, but the length doesn't match with self
        """
        if isinstance(other, self._scalars_types):
            other = Graph([other]*len(self.value))
        elif isinstance(other, self._array_types):
            assert len(self.value) == len(other), "Input dimension mismatch"
            other = Graph(np.array(other))
        elif isinstance(other, Graph):
            assert len(self.value) == len(other.value), "Input dimension mismatch"
        else:
            raise TypeError(f"Unsupported type {type(other)}")
        return np.equal(self.value, other.value)
    
    def __hash__(self):
        """
        Complements __eq__() function to ensure Graph instances are hashable
        """
        return id(self)
        # Ref: https://stackoverflow.com/questions/1608842/types-that-define-eq-are-unhashable

    def __ne__(self, other):
        """
        Overloads != operator to support checking Graph instances inequality

        Parameters
		----------
		other: int, float, np.int32, np.int64, np.float64, list, np.ndarray or Graph

        Raises:
		------
		TypeError
		    If other is not of type int, float, np.int32, np.int64, np.float64, list, np.ndarray or Graph
        AssertionError
            If other is a list, numpy array, or a Graph instance, but the length doesn't match with self
        """
        if isinstance(other, self._scalars_types):
            other = Graph([other]*len(self.value))
        elif isinstance(other, self._array_types):
            assert len(self.value) == len(other), "Input dimension mismatch"
            other = Graph(np.array(other))
        elif isinstance(other, Graph):
            assert len(self.value) == len(other.value), "Input dimension mismatch"
        else:
            raise TypeError(f"Unsupported type {type(other)}")
        return np.not_equal(self.value, other.value)

    def __str__(self):
        """
        Returns string representation of graph
        """
        return f"Graph({self.value})"

    def __repr__(self):
        """
        Returns internal object representation of graph in string format
        """
        return f"Graph({self.value})"
        

# References
# https://sidsite.com/posts/autodiff/