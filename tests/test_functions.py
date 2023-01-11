import pytest
import sys
import numpy as np

sys.path.append("../")
from best_autodiff.dualnumber import DualNumber
from best_autodiff.functions import *
# from best_autodiff import DualNumber
# from best_autodiff import *

class TestFunctions:
	"""
	Test class for best_autodiff package functions
	"""

	_tolerance = 0.0001

	def test_sqrt(self):
		z1 = DualNumber(1, 9)
		assert sqrt(z1) == DualNumber(1.0, 4.5)

		z2 = DualNumber(2, 4)
		z3 = sqrt(z2)
		assert z3.real == np.sqrt(2)
		assert z3.dual == (1/2) * (1/np.sqrt(2)) * 4

		# invalid inputs
		with pytest.raises(TypeError):
			sqrt(2,3)

		with pytest.raises(TypeError):
			sqrt(2)

		with pytest.raises(TypeError):
			sqrt("24")

		with pytest.raises(ValueError):
			sqrt(DualNumber(-2, 3))

	def test_log(self):
		z1 = DualNumber(2,4)
		z2 = log(z1, 2)

		assert z2.real == np.log2(2)
		assert z2.dual == 4 / (2 * np.log(2))

		z1 = DualNumber(np.e, 1)
		z2 = log(z1, np.e)

		assert abs(z2.real - np.log(np.e)) <= self._tolerance
		assert abs(z2.dual - (1 / (np.log(np.e) * np.e))) <= self._tolerance

		z1 = DualNumber(100, 2)
		z2 = log(z1, 10)

		assert z2.real == np.log10(100)
		assert z2.dual == 2 / (100 * np.log(10))

		# invalid inputs
		with pytest.raises(TypeError):
			log(3, 3)

		with pytest.raises(TypeError):
			log(DualNumber(2), "3")

		with pytest.raises(TypeError):
			log((2,3), "3")

	def test_root(self):
		z1 = DualNumber(4, 4)
		z2 = root(z1, 2)

		assert z2.real == np.power(4, 1/2)
		assert z2.dual == np.power(4, 1/2) * 4 / (4*2)

		z1 = DualNumber(8, 4)
		z2 = root(z1, 3)

		assert z2.real == np.power(8, 1/3)
		assert z2.dual == np.power(8, 1/3) * 4 / (8*3)

		# invalid inputs
		with pytest.raises(TypeError):
			root(z2)

		with pytest.raises(TypeError):
			root(3, DualNumber(3,2))

		with pytest.raises(TypeError):
			root((4, 3), 2)

		with pytest.raises(TypeError):
			root(DualNumber(1), DualNumber(2))

		with pytest.raises(TypeError):
			root(3)

		with pytest.raises(ValueError):
			root(DualNumber(-2, 3), 3)

	def test_exp(self):
		z1 = exp(DualNumber(5,2))
		assert z1.real == np.exp(5)
		assert z1.dual == np.exp(5) * 2

		z2 = exp(DualNumber(-2))
		assert z2.real == np.exp(-2)
		assert z2.dual == np.exp(-2)

		# invalid inputs
		with pytest.raises(TypeError):
			z = exp("hi")

		with pytest.raises(TypeError):
			z =  exp(2)

		with pytest.raises(TypeError):
			z =  exp(3.25)

	def test_sin(self):
		z1 = sin(DualNumber(5,2))
		assert z1.real == np.sin(5)
		assert z1.dual == 2*np.cos(5)

		z2 = sin(DualNumber(5))
		assert z2.real == np.sin(5)
		assert z2.dual == np.cos(5)

		z3 = sin(DualNumber(3.14/4))
		assert z3.real == np.sin(3.14/4)
		assert z3.dual == np.cos(3.14/4)

		# invalid inputs
		with pytest.raises(TypeError):
			z = sin("hi")

		with pytest.raises(TypeError):
			z = sin(2)

		with pytest.raises(TypeError):
			z = sin(3.25)

	def test_cos(self):
		z1 = cos(DualNumber(5,2))
		assert z1.real == np.cos(5)
		assert z1.dual == -2*np.sin(5)

		z2 = cos(DualNumber(5))
		assert z2.real == np.cos(5)
		assert z2.dual == -1*np.sin(5)

		z3 = cos(DualNumber(3.14/4))
		assert z3.real == np.cos(3.14/4)
		assert z3.dual == -1*np.sin(3.14/4)

		# invalid inputs
		with pytest.raises(TypeError):
			z = cos("hi")

		with pytest.raises(TypeError):
			z = cos(2)

		with pytest.raises(TypeError):
			z = cos(3.25)

	def test_tan(self):
		z1 = tan(DualNumber(5,2))
		assert z1.real == np.tan(5)
		assert z1.dual == 2 / (np.cos(5)**2)

		z2 = tan(DualNumber(5))
		assert z2.real == np.tan(5)
		assert z2.dual == 1 / (np.cos(5)**2)

		z3 = tan(DualNumber(3.14/4))
		assert z3.real == np.tan(3.14/4)
		assert z3.dual == 1 / (np.cos(3.14/4)**2)

		# invalid inputs
		with pytest.raises(TypeError):
			z = tan("hi")

		with pytest.raises(TypeError):
			z = tan(2)

		with pytest.raises(TypeError):
			z = tan(3.25)

	def test_arcsin(self):
		z1 = arcsin(DualNumber(1/2, np.sqrt(3)))
		assert z1.real == np.arcsin(1/2)
		assert z1.dual == 2.0

		z2 = arcsin(DualNumber(0))
		assert z2.real == np.arcsin(0)
		assert z2.dual == 1.0

		# invalid inputs
		with pytest.raises(TypeError):
			z = arcsin(0)

		with pytest.raises(TypeError):
			z = arcsin(0.2)

		with pytest.raises(TypeError):
			z = arcsin((2,3))

	def test_arccos(self):
		z1 = arccos(DualNumber(0.5))
		assert z1.real == np.arccos(0.5)
		assert z1.dual == -2 / np.sqrt(3)

		z2 = arccos(DualNumber(0, 2))
		assert z2.real == np.arccos(0)
		assert z2.dual == -2.0

		# invalid inputs
		with pytest.raises(TypeError):
			z = arccos(6)

		with pytest.raises(TypeError):
			z = arccos((8, 3))

		with pytest.raises(TypeError):
			z = arccos("345")

	def test_arctan(self):
		z1 = arctan(DualNumber(5))
		assert z1.real == np.arctan(5)
		assert z1.dual == 1/26

		z2 = arctan(DualNumber(0, 10))
		assert z2.real == np.arctan(0)
		assert z2.dual == 10.0

		# invalid inputs
		with pytest.raises(TypeError):
			z = arctan(4.5)

		with pytest.raises(TypeError):
			z = arctan((1, 3))

		with pytest.raises(TypeError):
			z = arctan("-0")

	def test_sinh(self):
		z1 = sinh(DualNumber(np.pi,2))
		assert abs(z1.real - np.sinh(np.pi)) <= self._tolerance
		assert z1.dual == np.cosh(np.pi) * 2

		z2 = sinh(DualNumber(100, -100))
		assert z2.real == np.sinh(100)
		assert z2.dual == np.cosh(100) * -100

		# invalid inputs
		with pytest.raises(TypeError):
			z = sinh(2)

		with pytest.raises(TypeError):
			z = sinh(-100)

		with pytest.raises(TypeError):
			z = sinh("A function")

	def test_cosh(self):
		z1 = cosh(DualNumber(np.e))
		assert z1.real == np.cosh(np.e)
		assert z1.dual == np.sinh(np.e)

		z2 = cosh(DualNumber(5, 0))
		assert z2.real == np.cosh(5)
		assert z2.dual == np.sinh(5) * 0

		# invalid inputs
		with pytest.raises(TypeError):
			z = cosh(-2.5)

		with pytest.raises(TypeError):
			z = cosh((-1, 199))

		with pytest.raises(TypeError):
			z = cosh("cosh")

	def test_tanh(self):
		z1 = tanh(DualNumber(0, 5))
		assert z1.real == np.tanh(0)
		assert abs(z1.dual - (1 - np.tanh(0)**2) * 5) < self._tolerance

		z2 = tanh(DualNumber(2))
		assert z2.real == np.tanh(2)
		assert abs(z2.dual - (1 - np.tanh(2)**2)) < self._tolerance

		with pytest.raises(TypeError):
			z = tanh(-100.0)

		with pytest.raises(TypeError):
			z = tanh(("hi", 199))

		with pytest.raises(TypeError):
			z = tanh(2)

	def test_logistic(self):
		z1 = logistic(DualNumber(0))
		assert z1.real == 1/2
		assert z1.dual == 1/4

		z2 = logistic(DualNumber(0, 4))
		assert z2.real == 1/2
		assert z2.dual == (1/4) * 4

		with pytest.raises(TypeError):
			z = logistic(np.pi)

		with pytest.raises(TypeError):
			z = logistic((4.0, 199))

		with pytest.raises(TypeError):
			z = logistic(2)

if __name__ == "__main__":
	pytest.main()