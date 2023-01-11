import pytest
import sys
import numpy as np

sys.path.append("../")
from best_autodiff.forward import Forward
from best_autodiff.dualnumber import DualNumber
from best_autodiff.functions import *

class TestForward:
	""" Test class for Forward method implementation 
	"""

	_tolerance = 0.0001

	def test_init(self):
		# 1 input 1 output
		def f(x):
			return x[0]**2

		fw = Forward(f, 1, 1)
		assert fw.f == [f]
		assert fw.m == 1
		assert fw.n == 1

		def f(x):
			return x[0] * x[1]

		fw = Forward([f], 2, 1)
		assert fw.f == [f]
		assert fw.m == 2
		assert fw.n == 1

		# wrong inputs
		# invalid number of inputs
		with pytest.raises(TypeError):
			fw = Forward(f, 0, 1)

		# invalid number of outputs
		with pytest.raises(TypeError):
			fw = Forward(f, 1, 0)

		# invalid input type
		with pytest.raises(TypeError):
			fw = Forward(f, 1.5, 1)

		# invalid input type
		with pytest.raises(TypeError):
			fw = Forward(f, 1, 2.5)

		# invalid function type
		with pytest.raises(TypeError):
			fw = Forward("hello", 1, 1)

		# invalid function type
		with pytest.raises(TypeError):
			fw = Forward(f, 1, 2)

		# invalid function type
		with pytest.raises(TypeError):
			fw = Forward([f, 2], 1, 2)

		# invalid functions / outputs
		with pytest.raises(TypeError):
			fw = Forward([f, f, f], 1, 2)

	def test_evaluate(self):
		# 1 input 1 output - scalar
		def f(x):
			return cos(x[0]) / exp(x[0])

		fw = Forward(f, 1, 1)
		assert fw.evaluate(0) == 1.0

		# 1 input 1 output - vector
		def f(x):
			return sqrt(x[0])

		fw = Forward(f, 1, 1)
		assert np.all(np.isclose(fw.evaluate(np.array([16])), np.array([4])))

		# 1 input multiple outputs
		def f1(x):
			return sinh(x[0])

		def f2(x):
			return sqrt(x[0]) - log(x[0], np.e)

		fw = Forward([f1, f2], 1, 2)
		assert np.all(np.isclose(fw.evaluate(1), np.array([np.sinh(1), 1.0])))

		# 1 input multiple outputs
		def f1(x):
			return sinh(x[0]) / cosh(x[0])

		def f2(x):
			return sqrt(x[0]) - log(x[0], 5)

		fw = Forward([f1, f2], 1, 2)
		assert np.allclose(fw.evaluate(np.array([25])), np.array([np.sinh(25) / np.cosh(25), 3]))

		# multiple inputs 1 output
		def f(x):
			return (log(exp(x[0]), np.e) * log(x[1], 2)) + tan(x[2])

		fw = Forward(f, 3, 1)
		assert np.all(np.isclose(fw.evaluate(np.array([1/4, 16, np.pi])), np.array([1])))

		# multiple inputs and multiple outputs
		def f1(x):
			return root(x[0], 4)

		def f2(x):
			return arcsin(x[1]) / logistic(x[0]) * logistic(x[2]**4)

		fw = Forward([f1, f2], 3, 2)
		assert np.all(np.isclose(fw.evaluate(np.array([81, 0, 3])), np.array([3.0, np.arcsin(0)])))

		with pytest.raises(TypeError):
			fw = Forward(lambda x: x*2, 1, 1)
			fw.evaluate([2])

		with pytest.raises(TypeError):
			fw = Forward(lambda x: x*3, 2, 1)
			fw.evaluate(2)

		with pytest.raises(TypeError):
			fw = Forward(lambda x: x[0] + x[1], 2, 1)
			x = np.array(["a", "b"])
			fw.evaluate(x)

		with pytest.raises(TypeError):
			fw = Forward(lambda x: x[0] * x[1], 2, 1)
			x = np.array([1, 1, 1])
			fw.evaluate(x)


	def test_get_jacobian(self):
		# 1 input 1 output - scalar 
		def f(x):
			return 2*x[0] + tan(x[0])

		fw = Forward(f, 1, 1)
		assert fw.get_jacobian(0) == 3.0 
    
		# testing seed for 1 input - should throw error
		with pytest.raises(TypeError):
			fw.get_jacobian(0, np.array([1,0]))

		# 1 input 1 output - vector
		def f(x):
			return log(x[0], 10)

		fw = Forward(f, 1, 1)
		jacobian = fw.get_jacobian(np.array([100]))
		expected_jacobian = np.array([1 / (np.log(10) * 100)])
		assert np.all(np.isclose(jacobian, expected_jacobian))	
    
		# testing seed for 1 input - should throw error
		with pytest.raises(TypeError):
			fw.get_jacobian(np.array([100]),np.array([0,1]))

		# 3 inputs, 2 outputs
		def f1(x):
			return x[0]**x[1]

		def f2(x):
			return sqrt(logistic(x[2]))

		fw = Forward([f1, f2], 3, 2)
		jacobian = fw.get_jacobian(np.array([4, 2, 0]))
		expected_jacobian = np.array([np.array([8, 16 * np.log(4) , 0]), np.array([0, 0, np.sqrt(2) / 8])])
		assert np.all(np.isclose(jacobian, expected_jacobian))
	
		## testing seed vector
		partial_jacobian = fw.get_jacobian(np.array([4, 2, 0]),np.array([1,0,0]))
		expected_partial_jacobian = np.array([[8, 0, 0], [0, 0, 0]])
		assert np.allclose(partial_jacobian, expected_partial_jacobian)

		# invalid inputs
		with pytest.raises(TypeError):
			fw = Forward(lambda x: x, 1, 1)
			fw.get_jacobian(np.array([2, 3, 4]))

		with pytest.raises(TypeError):
			fw = Forward(lambda x: x[0] * x[1], 2, 1)
			fw.get_jacobian(np.array(["1", "2"]))

		with pytest.raises(TypeError):
			fw = Forward(lambda x: x[0]**x[1], 2, 1)
			fw.get_jacobian((10, 20))

		with pytest.raises(TypeError):
			fw = Forward(lambda x: x[0]**x[1] + x[2], 3, 1)
			fw.get_jacobian(np.array([1, 10]))

		with pytest.raises(TypeError):
			fw = Forward(lambda x: x+2, 1, 1)
			fw.get_jacobian("6")
		
		with pytest.raises(TypeError):
			fw = Forward(lambda x: x+2, 1, 1)
			fw.get_jacobian(np.array([1]), 50)

		# multiple inputs 1 output
		def f(x):
			return sinh(x[0]) - cosh(x[1])

		fw = Forward(f, 2, 1)
		assert np.all(np.isclose(fw.get_jacobian(np.array([0, 0])), np.array([1, 0])))

		## testing seed vector
		partial_jacobian = fw.get_jacobian(np.array([0, 0]), np.array([1,0]))
		assert np.allclose(partial_jacobian, np.array([1, 0]))

		## testing weighted seed vector
		partial_weighted = fw.get_jacobian(np.array([0,0]), np.array([2,-2]))
		assert np.allclose(partial_weighted, np.array([2, 0]))
		

		# multiple inputs multiple outputs
		def f1(x):
			return 5 + tanh(x[0])

		def f2(x):
			return 5 - arcsin(x[1])

		def f3(x):
			return 5 / arccos(x[2])

		def f4(x):
			return 5 * arctan(x[3])

		fw = Forward([f1, f2, f3, f4], 4, 4)
		jacobian = fw.get_jacobian(np.array([0, 1/4, 0, 3]))
		expected_jacobian = np.array([1, 0, 0, 0, 0, -4 * np.sqrt(15) / 15, 0, 0, 0, 0, 20 / np.pi**2, 0, 0, 0, 0, 1/2])
		expected_jacobian = np.reshape(expected_jacobian, (4, 4))
		assert np.all(np.isclose(jacobian, expected_jacobian))

		#testing seed matrix
		seed = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
		partial = fw.get_jacobian(np.array([0, 1/4, 0, 3]), seed)
		assert np.all(np.isclose(expected_jacobian*np.diag([1,1,1,1]), partial))

		#testing wrong seed shape (number of functions)
		with pytest.raises(TypeError):
			partial = fw.get_jacobian(np.array([0, 1/4, 0, 3]), np.array([[1,0,0,0]]))
		
		#testing inavlid seed 
		with pytest.raises(TypeError):
			partial = fw.get_jacobian(np.array([0, 1/4, 0, 3]), np.array([['hi',0,0,0]]))
		with pytest.raises(TypeError):
			seed = np.array([[1,0,'hola',0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
			partial = fw.get_jacobian(np.array([0, 1/4, 0, 3]), seed)

		# testing seed vector
		partial_jacobian = fw.get_jacobian(np.array([0, 1/4, 0, 3]), np.array([0,0,0,1]))
		expected_partial_jacobian = np.zeros((4, 4))
		expected_partial_jacobian[3][3] = 1/2
		assert np.allclose(partial_jacobian, expected_partial_jacobian)

	def test_call(self):
		def f(x):
			return x[0]**3

		fw = Forward(f, 1, 1)
		assert fw(5) == (125, 75)

if __name__ == "__main__":
	pytest.main()