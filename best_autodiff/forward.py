"""
forward module
Implements Forward class: a class to perform the forward mode of automatic differentiation
"""

import numpy as np
from .dualnumber import DualNumber
from .functions import *

__all__ = ["Forward"]

class Forward:
	""" Class used to implement Forward mode of Automatic Differentiation for the following cases:
	1. Scalar function
	2. Vector function
	3. Multiple functions

	Attributes
	----------

	f: function or list of functions
		Function(s) to evaluate
 
	m: int
		Number of inputs to function

	n: int
		Number of outputs to function


	Methods
	-------
	__init__(self, f, m, n)
		Instantiate forward mode object

	evaluate(self, x)
		Calculate value of functions evaluated at x

	get_jacobian(self, x)
		Calculate Jacobian for function evaluated at x

	Example
	-------
	1. SCALAR CASE 
		1 input and 1 output

		def f(x):
			return x**2

		fw = Forward(f, 1, 1)
		print("value evaluated at 4:", fw.evaluate(4))
		>> value evaluated at 4: 16
		print("Jacobian evaluated at 4:", fw.get_jacobian(4))
		>> Jacobian evaluated at 4: 8

	2. Multiple inputs and 1 output

		def f(x):
			return x[0]**2 + sin(x[1])

		fw = Forward(f, 2, 1)
		print("value evaluated at [5, np.pi/2]:", fw.evaluate(np.array([5, np.pi/2])))
		>> value evaluated at [5, np.pi/2]: 26.0
		print("Jacobian evaluated at [5, 0]:", fw.get_jacobian(np.array([5, 0])))
		>> Jacobian evaluated at [5, 0]: [10.0, 1.0]

	3. 1 input and multiple outputs

		def f1(x):
			return exp(x**2)

		def f2(x):
			return 2*cos(x)

		fw = Forward([f1, f2], 1, 2)
		print("value evaluated at 1:", fw.evaluate(1))
		>> value evaluated at 1: [2.718281828459045, 1.0806046117362795]
		print("Jacobian evaluated at 1:", fw.get_jacobian(1))
		>> Jacobian evaluated at 1: [5.43656365691809, -1.682941969615793]


	4. Multiple inputs and multiple outputs

		def f1(x):
			return x[0] + x[1]

		def f2(x):
			return x[0] * x[1]

		def f3(x):
			return exp(x[0])

		fw = Forward([f1, f2, f3], 2, 3)
		print("value evaluated at [2, 5]:", fw.evaluate(np.array([2, 5])))
		>> value evaluated at [2, 5]: [7, 10, 7.38905609893065]
		print("Jacobian evaluated at [1, 2]:", fw.get_jacobian(np.array([1, 2])))
		>> Jacobian evaluated at [1, 2]: [[1, 1], [2, 1], [2.718281828459045, 0.0]]

	"""

	_supported_scalars = (float, int, np.int64, np.float64, np.int32, np.float32)
	_supported_lists = (list, np.ndarray)

	def __init__(self, f, m, n):
		""" Instantiates forward mode object

		Parameters
		----------
		f: list of functions
			function(s) user wants to evaluate

		m: int
			number of inputs to function(s)

		n: int
			number of outputs of function

		Raises
		------
		TypeError
			If the number of inputs is not an integer or less than 1
			If the number of outputs is not an integer or less than 1
			If there is 1 output and the function is not callable
			If there is more than 1 output and function argument is not a list of callable functions
		"""

		if not isinstance(m, int):
			raise TypeError(f"Unsupported type {type(m)} for number of inputs")

		if not isinstance(n, int):
			raise TypeError(f"Unsupported type {type(m)} for number of outputs")

		if m < 1:
			raise TypeError(f"Number of inputs {m} should be at least 1")

		if n < 1:
			raise TypeError(f"Number of outputs {n} should be at least 1")

		if (n == 1) and (not callable(f)) and (not isinstance(f, self._supported_lists)):
			raise TypeError(f"Unsupported function type {type(f)}")

		if n > 1 and not isinstance(f, self._supported_lists):
			raise TypeError(f"Expected a list of functions")

		if isinstance(f, self._supported_lists):
			for function in f:
				if not callable(function):
					raise TypeError(f"Functions must be callable")

		if isinstance(f, self._supported_lists) and len(f) != n:
			raise TypeError(f"Unsupported number of functions: expected {n} but received {len(f)} elements")

		self.f = []
		self.m = m
		self.n = n

		if callable(f):
			self.f.append(f)
		else:
			for function in f:
				self.f.append(function)

	def _get_value_highd(self, x):
		""" Calculates value for function with multiple outputs evaluated at x
		"""

		# evaluate each function given in instantiation
		values = []
		if self.m == 1 and isinstance(x, self._supported_scalars):
			for f in self.f:
				values.append(f([DualNumber(x)]).real)

		elif self.m == 1:
			for f in self.f:
				values.append(f([DualNumber(x[0])]).real)

		else:
			for f in self.f:
				val = self._get_value(f, x)
				values.append(self._get_value(f, x).squeeze())

		return np.array(values) # .reshape(self.n, 1)

	def _get_value(self, f, x):
		""" Calculates value for function with 1 output evaluated at x
		"""

		# if user inputs a scalar, return a scalar
		if self.m == 1 and isinstance(x, self._supported_scalars):
				return f([DualNumber(x)]).real

		# otherwise, if user inputs an array, return an array
		elif self.m == 1:
			return np.array([f([DualNumber(x[0])]).real])

		# convert each input into a Dual Number to calculate function evaluation
		else:
			inputs = []
			for z in x:
				inputs.append(DualNumber(z, 1))
			return np.array([f(inputs).real])

	def evaluate(self, x):
		""" Evaluate function at given x

		Parameters
		----------
		x: int, float, np.float64, np.int64, np.int32, np.float32 (for scalars) or np.ndarray (for vectors)
			Value(s) to evaluate function at

		Returns
		-------
		value: int or float (if function has 1 output) or np.ndarray (if function has more than 1 output)
			Result of function evaluated at given x

		Raises
		------
		TypeError
			If the function has 1 input and x is not of type int, float, np.float64, np.int32 or np.float32
			If the function has more than 1 input and x is not of type np.ndarray
			If the function has more than 1 input and x is not an array of np.float64, np.int32 or np.float32 elements
			If the function has more than 1 input and x is not an array of the same length as 
				the number of inputs defined in the AD instantiation

		"""

		if (self.m == 1) and (not isinstance(x, self._supported_scalars)) and (not isinstance(x, np.ndarray)):
			raise TypeError(f"Unsupported type {type(x)} for inputs")

		if self.m > 1 and not isinstance(x, np.ndarray):
			raise TypeError(f"Unsupported type {type(x)} for function with {self.m} inputs")

		if isinstance(x, np.ndarray):
			for element in x:
				if not isinstance(element, self._supported_scalars):
					raise TypeError(f"Unsupported type {type(element)} for element {element} inside inputs")

		if isinstance(x, np.ndarray) and len(x) != self.m:
			raise TypeError(f"Unsupported number of inputs: expected {self.m} and received {len(x)}")

		if self.n > 1:
			return self._get_value_highd(x)
		else:
			return self._get_value(self.f[0], x)

	def _get_jacobian_highd(self, x):
		""" Calculates Jacobian for function(s) with multiple outputs evaluated at x
		"""

		jacobian = []

		# calculate each row of the Jacobian: one for each of the functions given
		for f in self.f:
			jacobian.append(self._get_jacobian(f, x))

		return np.array(jacobian).reshape(self.n, self.m)

	def _get_jacobian(self, f, x):
		""" Calculates Jacobian for function(s) with 1 output evaluated at x
		"""

		# if user inputs a scalar, return a scalar
		if self.m == 1 and isinstance(x, self._supported_scalars):
			return f([DualNumber(x,1)]).dual

		# otherwise, if user inputs an array, return an array
		elif self.m == 1:
			return np.array([f([DualNumber(x[0])]).dual])

		else:
			jacobian = []
			inputs = []
			for z in x:
				inputs.append(DualNumber(z, 1))

			# calculate the partial derivative with respect to each input
			for i, val in enumerate(inputs):
				partial_values = [DualNumber(z.real, 0) for z in inputs]
				partial_values[i] = val
				jacobian.append([f(partial_values).dual])

			return np.reshape(np.array(jacobian), (1, self.m))

	def get_jacobian(self, x, seed = None):
		""" Calculates Jacobian for function evaluated at x

		Parameters
		----------
		x: int, float, np.float64, np.int64, np.int32, np.float32 (if function has 1 input) or np.ndarray (if function has more than 1 input)
			Value(s) to evaluate Jacobian at
		
		seed: np.ndarray 
			Directional seed vector

		Returns
		-------
		value: np.ndarray
			Result of Jacobian evaluated at given x

		Raises
		------
		TypeError
			If the function has 1 input and x is not of type int, float, np.int64, np.float64, np.int32, np.float32
			If the function has more than 1 input and x is not of type np.ndarray
			If the function has more than 1 input and x is not an array of np.float64, np.int32 or np.float32 elements
			If the function has more than 1 input and x is not an array of the same length as 
				the number of inputs defined in the AD instantiation
		"""

		if (self.m == 1) and (not isinstance(x, self._supported_scalars)) and (not isinstance(x, np.ndarray)):
			raise TypeError(f"Unsupported type {type(x)} for inputs")

		if self.m > 1 and not isinstance(x, np.ndarray):
			raise TypeError(f"Unsupported type {type(x)} for function with {self.m} inputs")

		if seed is not None:
			if not isinstance(seed, np.ndarray):
				raise TypeError(f"Unsupported type {type(seed)} for seed vector")
			else:
				if seed.ndim == 1:
					if self.m != seed.shape[0]:
						raise TypeError(f"Seed of size {seed.shape} doesn't match number of inputs {self.m}")
				else:
					if self.m != seed.shape[1]:
						raise TypeError(f"Seed of size {seed.shape} doesn't match number of inputs {self.m}")
					elif self.n != seed.shape[0]:
						raise TypeError(f"Seed of size {seed.shape} doesn't match number of outputs {self.n}")
				if not all(isinstance(item, self._supported_scalars) for item in seed.flatten()):
					raise TypeError(f"Seed contains unsupported datatypes.")

		if isinstance(x, np.ndarray):
			for element in x:
				if not isinstance(element, self._supported_scalars):
					raise TypeError(f"Unsupported type {type(element)} for element {element} inside inputs")

		if isinstance(x, np.ndarray) and len(x) != self.m:
			raise TypeError(f"Unsupported number of inputs: expected {self.m} and received {len(x)}")

		if self.n > 1: # more than one function
			jacobian = self._get_jacobian_highd(x)
		else: # 1 function
			jacobian = self._get_jacobian(self.f[0], x)
		
		if seed is not None: # seed specified
			return jacobian * seed
		else: # return full jacobian
			return jacobian

	def __call__(self, x):
		"""
		calls the evaluate and get_jacobian methods from the forward object
		"""
		return (self.evaluate(x), self.get_jacobian(x))