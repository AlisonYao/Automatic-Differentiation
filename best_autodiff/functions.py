"""
functions module
Implements mathematical functions for the best_autodiff package
Functions are intended to work with dual numbers

sqrt(z)
	Implements squared root of a DualNumber

log(z, base=np.e)
	Implements log function of a DualNumber using any base
	Default base is np.e

root(z, n)
	Implements function to take the nth root of a DualNumber

exp(z)
	Implements exponential function exp to the power of a DualNumber

sin(z)
	Implements sine of a DualNumber

cos(z)
	Implements cosine of a DualNumber

tan(z)
	Implements tangent of a DualNumber

arcsin(z)
	Implements arcsine of a DualNumber

arccos(z)
	Implements arccosine of a DualNumber

arctan(z)
	Implements arctangend of a DualNumber

sinh(z)
	Implements hyperbolic sine of a DualNumber

cosh(z)
	Implements hyperbolic cosine of a DualNumber

tanh(z)
	Implements hyperbolic tangent of a DualNumber

logistic(z)
	Implements the sigmoid logistic function of a DualNumber
"""

import numpy as np
from .dualnumber import DualNumber


__all__ = ["sqrt", "log", "root", "exp", "sin", "cos", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "logistic"]

_scalars_types = (int, float, np.int32, np.int64, np.float64)

def sqrt(z):
	""" Implements squared root of a DualNumber

	Parameters
	----------
	z: DualNumber

	Returns
	-------
	DualNumber
		Result of square root

	Raises
	------
	TypeError
		If z is not a DualNumber

	Example
	-------
	>>>z1 = DualNumber(1, 9)
	>>>sqrt(z1)
	DualNumber(1.0, 4.5)

	"""

	if not isinstance(z, DualNumber):
		raise TypeError(f"Unsupported type {type(z)}")

	if z.real < 0:
		raise ValueError(f"Domain out of range")

	return DualNumber(np.sqrt(z.real), (1 / 2) * (1 / np.sqrt(z.real)) * z.dual)


def log(z, base = np.e):
	""" Implements log function of a DualNumber using any base (default base is np.e)

	Parameters
	----------
	z: DualNumber
	base: int, float, np.int32, np.int64 or np.float64

	Returns
	-------
	DualNumber
		Result of log

	Raises
	------
	TypeError
		If z is not DualNumber
		If base is not int, float, np.int32, np.int64 or np.float64
	
	Example
	-------
	>>>z1 = DualNumber(100, 2)
	>>>z2 = log(z1, 10)
	DualNumber(np.log10(100), 2 / (100 * np.log(10))
	"""

	if not isinstance(z, DualNumber):
		raise TypeError(f"Unsupported type {type(z)}")

	if not isinstance(base, (int, float, np.int32, np.int64, np.float64)):
		raise TypeError(f"Unsupported type {type(base)}")

	return DualNumber(np.log(z.real) / np.log(base),
		z.dual / (np.log(base) * z.real))

def root(z, n):
	""" Implements function to take the nth root of a DualNumber

	Parameters
	----------
	z: DualNumber
	n: int, float, np.int32, np.int64 or np.float64
		nth root of a dual number

	Returns
	-------
	DualNumber
		Result of root

	Raises
	------
	TypeError
		If z is not DualNumber
		If n is not int, float, np.int32, np.int64 or np.float64
	
	Example
	-------
	>>>z1 = DualNumber(8, 4)
	>>>z2 = root(z1, 3)
	DualNumber(np.power(8, 1/3), np.power(8, 1/3) * 4 / (8*3))
	"""

	if not isinstance(z, DualNumber):
		raise TypeError(f"Unsupported type {type(z)}")

	if not isinstance(n, (int, float, np.int32, np.int64, np.float64)):
		raise TypeError(f"Unsupported type {type(n)}")

	if z.real < 0:
		raise ValueError(f"Domain out of range")

	return DualNumber(np.power(z.real, 1/n),
		(np.power(z.real, 1/n) * z.dual) / (n * z.real))

def exp(z):
	""" Implements exponential function exp to the power of a DualNumber

	Parameters
	----------
	z: DualNumber

	Returns
	-------
	DualNumber
		Result of exp

	Raises
	------
	TypeError
		If z is not a DualNumber
	
	Example
	-------
	>>>z1 = exp(DualNumber(5,2))
	DualNumber(np.exp(5),np.exp(5) * 2)
	"""

	if not isinstance(z, DualNumber):
		raise TypeError(f"Unsupported type {type(z)}")

	return DualNumber(np.exp(z.real), np.exp(z.real) * z.dual)

def sin(z):
	""" Implements sine of a DualNumber

	Parameters
	----------
	z: DualNumber

	Returns
	-------
	DualNumber
		Result of sin

	Raises
	------
	TypeError
		If z is not a DualNumber
	
	Example
	-------
	>>>z1 = sin(DualNumber(5,2))
	DualNumber(np.sin(5), 2*np.cos(5))
	"""

	if not isinstance(z, DualNumber):
		raise TypeError(f"Unsupported type {type(z)}")

	return DualNumber(np.sin(z.real), np.cos(z.real) * z.dual)

def cos(z):
	""" Implements cosine of a DualNumber

	Parameters
	----------
	z: DualNumber

	Returns
	-------
	DualNumber
		Result of cos

	Raises
	------
	TypeError
		If z is not a DualNumber
	
	Example
	-------
	>>>z1 = cos(DualNumber(5,2))
	DualNumber(np.cos(5),-2*np.sin(5))
	"""

	if not isinstance(z, DualNumber):
		raise TypeError(f"Unsupported type {type(z)}")

	return DualNumber(np.cos(z.real), -np.sin(z.real) * z.dual)

def tan(z):
	""" Implements tangent of a DualNumber

	Parameters
	----------
	z: DualNumber

	Returns
	-------
	DualNumber
		Result of tan

	Raises
	------
	TypeError
		If z is not a DualNumber
	
	Example
	-------
	>>>z1 = tan(DualNumber(5,2))
	DualNumber(np.tan(5),2 / (np.cos(5)**2))
	"""

	if not isinstance(z, DualNumber):
		raise TypeError(f"Unsupported type {type(z)}")

	return DualNumber(np.tan(z.real), (1 / (np.cos(z.real)**2)) * z.dual)

def arcsin(z):
	""" Implements arcsine of a DualNumber

	Parameters
	----------
	z: DualNumber

	Returns
	-------
	DualNumber
		Result of arcsin

	Raises
	------
	TypeError
		If z is not a DualNumber
	
	Example
	-------
	>>>z1 = arcsin(DualNumber(1/2, np.sqrt(3)))
	DualNumber(np.arcsin(1/2),2.0)
	"""

	if not isinstance(z, DualNumber):
		raise TypeError(f"Unsupported type {type(z)}")

	return DualNumber(np.arcsin(z.real), (1 / np.sqrt(1 - (z.real**2))) * z.dual)

def arccos(z):
	""" Implements arccosine of a DualNumber

	Parameters
	----------
	z: DualNumber

	Returns
	-------
	DualNumber
		Result of arccos

	Raises
	------
	TypeError
		If z is not a DualNumber
	
	Example
	-------
	>>>z1 = arccos(DualNumber(0.5))
	DualNumber(np.arccos(0.5),-2 / np.sqrt(3))
	"""

	if not isinstance(z, DualNumber):
		raise TypeError(f"Unsupported type {type(z)}")

	return DualNumber(np.arccos(z.real), (1 / np.sqrt(1 - (z.real**2))) * z.dual * -1)

def arctan(z):
	""" Implements arctan of a DualNumber

	Parameters
	----------
	z: DualNumber

	Returns
	-------
	DualNumber
		Result of arctan

	Raises
	------
	TypeError
		If z is not a DualNumber
	
	Example
	-------
	>>>z1 = arctan(DualNumber(5))
	DualNumber(np.arctan(5),1/26)
	"""

	if not isinstance(z, DualNumber):
		raise TypeError(f"Unsupported type {type(z)}")

	return DualNumber(np.arctan(z.real), (1 / (1 + (z.real**2))) * z.dual)

def sinh(z):
	""" Implements hyperbolic sine of a DualNumber

	Parameters
	----------
	z: DualNumber

	Returns
	-------
	DualNumber
		Result of sinh

	Raises
	------
	TypeError
		If z is not a DualNumber
	
	Example
	-------
	>>>z1 = sinh(DualNumber(np.pi,2))
	DualNumber(np.sinh(np.pi),np.cosh(np.pi) * 2)
	"""

	if not isinstance(z, DualNumber):
		raise TypeError(f"Unsupported type {type(z)}")

	return DualNumber(np.sinh(z.real), np.cosh(z.real) * z.dual)

def cosh(z):
	""" Implements hyperbolic cosine of a DualNumber

	Parameters
	----------
	z: DualNumber

	Returns
	-------
	DualNumber
		Result of cosh

	Raises
	------
	TypeError
		If z is not a DualNumber
	
	Example
	-------
	>>>z1 = cosh(DualNumber(np.e))
	DualNumber(np.cosh(np.e),np.sinh(np.e))
	"""

	if not isinstance(z, DualNumber):
		raise TypeError(f"Unsupported type {type(z)}")

	return DualNumber(np.cosh(z.real), np.sinh(z.real) * z.dual)

def tanh(z):
	""" Implements hyperbolic tangent of a DualNumber

	Parameters
	----------
	z: DualNumber

	Returns
	-------
	DualNumber
		Result of tanh

	Raises
	------
	TypeError
		If z is not a DualNumber
	
	Example
	-------
	>>>z1 = tanh(DualNumber(0, 5))
	DualNumber(np.tanh(0), (1 - np.tanh(0)**2) * 5)
	"""

	if not isinstance(z, DualNumber):
		raise TypeError(f"Unsupported type {type(z)}")

	return DualNumber(np.tanh(z.real), (1 / np.cosh(z.real))**2 * z.dual)

def logistic(z):
	""" Implements the sigmoid logistic function of a DualNumber

	Parameters
	----------
	z: DualNumber

	Returns
	-------
	DualNumber
		Result of sigmoid function

	Raises
	------
	TypeError
		If z is not a DualNumber
	
	Example
	-------
	>>>z2 = logistic(DualNumber(0, 4))
	DualNumber(1/2 , (1/4) * 4)
	"""

	if not isinstance(z, DualNumber):
		raise TypeError(f"Unsupported type {type(z)}")

	return DualNumber(1 / (1 + np.exp(-z.real)),\
		(np.exp(-z.real) / (1 + np.exp(-z.real))**2) * z.dual )