"""
dualnumber module
Implements DualNumber: a class to represent dual numbers
"""

import numpy as np  

__all__ = ["DualNumber"]

class DualNumber:
	""" A class used to represent Dual Numbers 

	Attributes
	---------
	real: int, float, np.int32, np.int64 or np.float64
	    the real part of the dual number 

	dual: int, float, np.int32, np.int64 or np.float64
	    the dual part of the dual number

	Methods
	-------
	__add__(other)
	    Overloads + operator to support dual number addition with other dual numbers, floats and ints

	__radd__(other)
	    Overloads reverse + operator to support addition of dual numbers

	__mul__(other)
	    Overloads * operator to support dual number multiplication with other dual numbers, floats and ints

	__rmul__(other)
	    Overloads reverse * operator to support multiplication of dual numbers

	__sub__(other)
	    Overloads - operator to support dual number subtraction with other dual numbers, floats and ints

	__rsub__(other)
	    Overloads reverse - operator to support subtraction of dual numbers

	__truediv__(other)
		Overloads / operator to support dual number division with other dual numbers, floats and ints

	__rtruediv__(other)
	    Overloads reverse / operator to support dual number division

	__neg__()
	    Overloads - operator to support dual number negation

	__pow__(other)
	    Overloads ** operator to support exponentiation of dual numbers
	
	__rpow__(other)
	    Overloads reverse ** operator to support reflexive exponentiation of dual numbers

	__eq__(z)
		Overloads == operator to support checking equality for dual numbers

	__ne__(z)
		Overloads != operator to support checking for dual number inequality

	__str__()
		Returns string representation of dual numbers

	__repr__()
		Returns internal object representation of dual numbers in string format
	"""
	_scalars_types = (int, float, np.int32, np.int64, np.float64)

	def __init__(self, real, dual=1.0):
		""" Initializes instance of class Dual Number

		Parameters
		---------
		real: int, float, np.int32, np.int64 or np.float64
		    Real part of dual number 

		dual: int, float, np.int32, np.int64 or np.float64, optional
		    Dual part of dual number. If not given, the default is 1.0

		Raises
		------
		TypeError
		    If real or dual arguments are not int, float, np.int32, np.int64 or np.float64
		"""
		if not isinstance(real, self._scalars_types):
			raise TypeError(f"Unsupported type {type(real)}")
		elif not isinstance(dual, self._scalars_types):
			raise TypeError(f"Unsupported type {type(dual)}")

		self.real = real
		self.dual = dual 

	def __add__(self, other):
		""" Overloads + operator to support dual number addition with other dual numbers, floats and ints

		Parameters
		----------
		other: int, float, np.int32, np.int64 or np.float64 or DualNumber
		    Number to add to dual number

		Returns
		-------
		DualNumber
			Result of the addition

		Raises
		------
		TypeError
		    If other is not of type int, float, np.int32, np.int64 or np.float64 or DualNumber
		"""
		if not isinstance(other, (*self._scalars_types, DualNumber)):
			raise TypeError(f"Unsupported type {type(other)}")

		if isinstance(other, self._scalars_types):
			return DualNumber(other + self.real, self.dual)

		else:
			return DualNumber(other.real + self.real, other.dual + self.dual)

	def __radd__(self, other):
		""" Overloads reverse + operator to support addition of dual numbers

		Parameters
		----------
		other: int, float, np.int32, np.int64 or np.float64 or DualNumber
		    Number to add to dual number

		Returns
		-------
		DualNumber
			Result of the reflected addition

		Raises
		------
		TypeError
		    If other is not of type int, float, np.int32, np.int64 or np.float64 or DualNumber
		"""
		if not isinstance(other, (*self._scalars_types, DualNumber)):
			raise TypeError(f"Unsupported type {type(other)}")

		return self.__add__(other)

	def __mul__(self, other):
		""" Overloads * operator to support dual number multiplication with other dual numbers, floats and ints

		Parameters
		----------
		other: int, float, np.int32, np.int64 or np.float64 or DualNumber
		    Number to multiply to dual number
		
		Returns
		-------
		DualNumber
			Result of the multiplication

		Raises
		------
		TypeError
		    If other is not of type int, float, np.int32, np.int64 or np.float64 or DualNumber
		"""
		if not isinstance(other, (*self._scalars_types, DualNumber)):
			raise TypeError(f"Unsupported type {type(other)}")

		if isinstance(other, self._scalars_types):
			return DualNumber(other*self.real, other*self.dual)

		else:
			return DualNumber(other.real*self.real, (other.dual*self.real) + (other.real*self.dual))

	def __rmul__(self, other):
		""" Overloads reverse * operator to support multiplication of dual numbers

		Parameters
		----------
		other: int, float, np.int32, np.int64 or np.float64 or DualNumber
		    Number to multiply to dual number

		Returns
		-------
		DualNumber
			Result of the reflected multiplication

		Raises
		------
		TypeError
		    If other is not of type int, float, np.int32, np.int64 or np.float64 or DualNumber
		"""
		if not isinstance(other, (*self._scalars_types, DualNumber)):
			raise TypeError(f"Unsupported type {type(other)}")


		return self.__mul__(other)

	def __sub__(self, other):
		""" Overloads - operator to support dual number subtraction with other dual numbers, floats and ints

		Parameters
		----------
		other: int, float, np.int32, np.int64 or np.float64 or DualNumber
		    Number to subtract from dual number

		Returns
		-------
		DualNumber
			Result of the subtraction

		Raises
		------
		TypeError
		    If other is not of type int, float, np.int32, np.int64 or np.float64 or DualNumber
		"""
		if not isinstance(other, (*self._scalars_types, DualNumber)):
			raise TypeError(f"Unsupported type {type(other)}")

		if isinstance(other, self._scalars_types):
			return DualNumber(self.real - other.real, self.dual)

		else:
			return DualNumber(self.real - other.real, self.dual - other.dual)

	def __rsub__(self, other):
		""" Overloads reverse - operator to support subtraction of dual numbers

		Parameters
		----------
		other: int, float, np.int32, np.int64 or np.float64
		    Number to do substraction with

		Returns
		-------
		DualNumber
			Result of the reflected subtraction

		Raises
		------
		TypeError
		    If other is not of type int, float, np.int32, np.int64 or np.float64
		"""
		if not isinstance(other, self._scalars_types):
			raise TypeError(f"Unsupported type {type(other)}")

		else:
			return DualNumber(other - self.real, -self.dual)

	def __truediv__(self, other):
		""" Overloads / operator to support dual number division with other dual numbers, floats and ints

		Parameters
		----------
		other: int, float, np.int32, np.int64 or np.float64 or DualNumber
		    Number to divide from dual number

		Returns
		-------
		DualNumber
			Result of the divison

		Raises
		------
		TypeError
		    If other is not of type int, float, np.int32, np.int64 or np.float64 or DualNumber

		ZeroDivisionError
			If other is 0 or if the real part of other is 0
		"""
    
		if not isinstance(other, (*self._scalars_types, DualNumber)):
			raise TypeError(f"Unsupported type {type(other)}")

		if isinstance(other, DualNumber) and other.real == 0:
			raise ZeroDivisionError(f"Attempted division by zero")

		if not isinstance(other, DualNumber) and other == 0:
			raise ZeroDivisionError(f"Attempted division by zero")

		if isinstance(other, self._scalars_types):
			return DualNumber(self.real / other, self.dual / other)

		else:
			return DualNumber(self.real / other.real, 
				((self.dual * other.real) - (self.real * other.dual)) / (other.real * other.real))

	def __rtruediv__(self, other):
		""" Overloads reverse / operator to support dual number division

		Parameters
		----------
		other: int, float, np.int32, np.int64 or np.float64 or DualNumber
		    Number to do division with

		Returns
		-------
		DualNumber
			Result of the reflected divison

		Raises
		------
		TypeError
		    If other is not of type int, float, np.int32, np.int64 or np.float64 or DualNumber

		ZeroDivisionError
			If the real part self is 0
		"""
		if not isinstance(other, self._scalars_types):
			raise TypeError(f"Unsupported type {type(other)}")

		else:
			return DualNumber(other / self.real, -(other * self.dual) / (self.real * self.real))

	def __neg__(self):
		""" Overloads - operator to support dual number negation

		Returns
		-------
		DualNumber
			Result of negation
		"""
		return DualNumber(-self.real, -self.dual)

	def __pow__(self, other):
		""" Overloads ** operator to support exponentiation of dual numbers

		Parameters
		----------
		other: int, float, np.int32, np.int64 or np.float64 or DualNumber
		    Power to exponentiate dual number

		Returns
		-------
		DualNumber
			Result of the exponentiation

		Raises
		------
		TypeError
		    If other is not of type int, float, np.int32, np.int64 or np.float64 or DualNumber
		"""
		if not isinstance(other,  (*self._scalars_types, DualNumber)):
			raise TypeError(f"Unsupported type {type(other)}")

		if isinstance(other, DualNumber):
			return DualNumber(self.real ** other.real,
				self.real ** other.real * ((other.dual * np.log(self.real)) + (self.dual * other.real / self.real)))

		else:
			return DualNumber(self.real**other, other * self.real**(other-1) * self.dual)

	def __rpow__(self, other):
		""" Overloads reflected ** operator to support exponentiation to the power of dual numbers

		Parameters
		----------
		other: int, float, np.int32, np.int64 or np.float64
		    Number to exponentiate

		Returns
		-------
		DualNumber
			Result of the reflected exponentiation

		Raises
		------
		TypeError
		    If other is not of type int, float, np.int32, np.int64 or np.float64
		"""
		if not isinstance(other, self._scalars_types):
			raise TypeError(f"Unsupported type {type(other)}")

		return DualNumber(other, 0).__pow__(self)

	def __eq__(self, other):
		""" Overloads == operator to support checking for dual number equality

		Parameters
		----------
		other: DualNumber

		Returns
		-------
		DualNumber
			Result of the equality check

		Raises:
		------
		TypeError
			If other is not a DualNumber
		"""
		if not isinstance(other, DualNumber):
			raise TypeError(f"Unsupported type {type(other)}")

		if self.real != other.real:
			return False 
		if self.dual != other.dual:
			return False 
		return True

	def __str__(self):
		"""
		String repersentation of a DualNumber
		"""
		return f"Dual Number {self.real} + {self.dual}E"

	def __repr__(self):
		"""
		String (internal) representation of a DualNumber object
		"""
		return f"DualNumber({self.real}, {self.dual})"