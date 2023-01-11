import pytest
import sys, os
import numpy as np

sys.path.append("../")
from best_autodiff.forward import Forward
from best_autodiff.dualnumber import DualNumber
from best_autodiff.functions import *

class TestDualNumber:
	"""
	Test class for Dual Number implementation 
	"""
	def test_init(self):
		# testing invalid inputs
		with pytest.raises(TypeError):
			x = DualNumber("hi")
		
		with pytest.raises(TypeError):
			x = DualNumber(DualNumber(2))
		
		with pytest.raises(TypeError):
			x = DualNumber(3, "hi")
		
		# testing setting of class attributes
		z1 = DualNumber(3)
		assert z1.real == 3
		assert z1.dual == 1
		
		z2 = DualNumber(1.45, 2.0)
		assert z2.real == 1.45
		assert z2.dual == 2.0
	
	def test_add(self):
		z1 = DualNumber(3.0) + DualNumber(4.0)
		assert z1.real == 7.0
		assert z1.dual == 2.0

		z2 = DualNumber(2.0, 3.0) + DualNumber(3.0)
		assert z2.real == 5.0
		assert z2.dual == 4.0

		z3 = DualNumber(1,2) + DualNumber(3,4)
		assert z3.real == 4
		assert z3.dual == 6

		z4 = DualNumber(1) + DualNumber(3,4)
		assert z4.real == 4
		assert z4.dual == 5

		z5 = DualNumber(2) + 5
		assert z5.real == 7
		assert z5.dual == 1

		z6 = DualNumber(3,4) + 10
		assert z6.real == 13
		assert z6.dual == 4

		z7 = DualNumber(3) + DualNumber(-1,-2)
		assert z7.real == 2
		assert z7.dual == -1

		# testing invalid input
		with pytest.raises(TypeError):
			z = DualNumber(2) + "hi"
  
	def test_radd(self):
		z1 = 5 + DualNumber(2)
		assert z1.real == 7
		assert z1.dual == 1
		  
		z2 = 10.0 + DualNumber(3,4)
		assert z2.real == 13.0
		assert z2.dual == 4

		z3 = -6 + DualNumber(2)
		assert z3.real == -4
		assert z3.dual == 1

		#testing invalid input
		with pytest.raises(TypeError):
		  z = "hi" + DualNumber(2)
  
	def test_mul(self):
		z1 = DualNumber(2,3) * DualNumber(4,5)
		assert z1.real == 8
		assert z1.dual == 22

		z2 = DualNumber(16) * DualNumber(.25,2)
		assert z2.real == 4
		assert z2.dual == 32.25

		z3 = DualNumber(2,3)*5
		assert z3.real == 10
		assert z3.dual == 15

		z4 = DualNumber(4,-2)* DualNumber(-2, -2)
		assert z4.real == -8
		assert z4.dual == -4

		z5 = DualNumber(2,3) * DualNumber(-1,-1)
		assert z5.real == -2
		assert z5.dual == -5

		z6 = DualNumber(2)*0
		assert z6.real == 0
		assert z6.dual == 0

		# testing invalid input
		with pytest.raises(TypeError):
			z = DualNumber(2)*"hi"
  
	def test_rmul(self):
		z1 = 5.0 * DualNumber(2.0, 1.45)
		assert z1.real == 10.0
		assert z1.dual == 7.25

		z2 = -5 * DualNumber(2,-2)
		assert z2.real == -10
		assert z2.dual == 10

		z3 = -5 * DualNumber(-2,2)
		assert z3.real == 10
		assert z3.dual == -10

		z4 = 0 * DualNumber(3)
		assert z4.real == 0
		assert z4.dual == 0

		# testing invalid input
		with pytest.raises(TypeError):
			z = "hi" * DualNumber(2)
	
	def test_sub(self):
		z1 = DualNumber(10) - DualNumber(5, 2)
		assert z1.real == 5
		assert z1.dual == -1

		z2 = DualNumber(5.0,2.0) - DualNumber(2.0)
		assert z2.real == 3.0
		assert z2.dual == 1.0

		z3 = DualNumber(-10) - DualNumber(2)
		assert z3.real == -12
		assert z3.dual == 0

		z4 = DualNumber(-4, -2) - DualNumber(-3, -2)
		assert z4.real == -1
		assert z4.dual == 0

		z5 = DualNumber(-4, 2) - 4
		assert z5.real == -8
		assert z5.dual == 2

		#testing invalid input
		with pytest.raises(TypeError):
			z = DualNumber(2) - "hi"
  
	def test_rsub(self):
		z1 = -4 - DualNumber(-4, 2)
		assert z1.real == 0
		assert z1.dual == -2

		z2 = 4 - DualNumber(3)
		assert z2.real == 1
		assert z2.dual == -1

		#testing invalid input
		with pytest.raises(TypeError):
			z =  "hi" - DualNumber(2)
	  
	def test_truediv(self):
		z1 = DualNumber(16,2) / DualNumber(4)
		assert z1.real == 4
		assert z1.dual == -.5

		z2 = DualNumber(0)/ DualNumber(2,3)
		assert z2.real == 0
		assert z2.dual == .5

		z3 = DualNumber(-10, 4)/ DualNumber(2, -2)
		assert z3.real == -5
		assert z3.dual == -3

		z4 = DualNumber(-10, -4)/ DualNumber(-2, 2)
		assert z4.real == 5
		assert z4.dual == 7

		z5 = DualNumber(16, 4) / 4
		assert z5.real == 4
		assert z5.dual == 1

		z6 = DualNumber(16, 4) / -4
		assert z6.real == -4
		assert z6.dual == -1

		# testing invalid input
		with pytest.raises(TypeError):
			z =  DualNumber(2) / "hi"

		# testing division by zero with other dual numbers
		with pytest.raises(ZeroDivisionError):
			z = DualNumber(2) / DualNumber(0)

		# testing division by zero with ints
		with pytest.raises(ZeroDivisionError):
			z = DualNumber(2) / 0

		# testing division by zero with float
		with pytest.raises(ZeroDivisionError):
			z = DualNumber(2) / 0.0
	  
	def test_rtruediv(self):
		z1 = 16 / DualNumber(4)
		assert z1.real == 4
		assert z1.dual == -1

		z2 = -18 / DualNumber(3,2)
		assert z2.real == -6
		assert z2.dual == 4

		z2 = -18 / DualNumber(-3,-2)
		assert z2.real == 6
		assert z2.dual == -4

		# testing invalid input
		with pytest.raises(TypeError):
			z =  "hi"/DualNumber(2)
  
	def test_neg(self):
		z1 = -DualNumber(2,3)
		assert z1.real == -2
		assert z1.dual == -3

		z2 = -DualNumber(4)
		assert z2.real == -4
		assert z2.dual == -1
  
	def test_pow(self):
		z1 = DualNumber(2,3) ** DualNumber(3,2)
		assert z1.real == 8
		assert z1.dual == 8*(2*np.log(2)+(9/2))

		z2 = DualNumber(3)**4
		assert z2.real == 81
		assert z2.dual == 108

		z3 = DualNumber(2)**-3
		assert z3.real == 1/8
		assert z3.dual == -3/16

		z4 = DualNumber(4)**.5
		assert z4.real == 2
		assert z4.dual == .25

		#testing invalid input
		with pytest.raises(TypeError):
			z =  DualNumber(2)**"hi"
  
	def test_rpow(self):
		z1 = 4 ** DualNumber(2,2)
		assert z1.real == 16
		assert z1.dual == 32*np.log(4)

		z2 = 3 ** DualNumber(-2)
		assert z2.real == 1/9
		assert z2.dual == (1/9)*np.log(3)

		# testing invalid input
		with pytest.raises(TypeError):
			z =  "hi" ** DualNumber(2)

	def test_eq(self):
		z1 = DualNumber(2,3)
		z2 = DualNumber(2,3)
		z3 = DualNumber(2,4)
		z4 = DualNumber(4,3)
		z5 = DualNumber(4,4)

		assert z1 == z2 
		assert z1 != z3
		assert z1 != z4
		assert z2 != z5

		# testing invalid input
		with pytest.raises(TypeError):
			z1 == 2

	def test_str(self):
		z1 = DualNumber(2, 3)
		assert str(z1) == "Dual Number 2 + 3E"

		z2 = DualNumber(10)
		assert str(z2) == "Dual Number 10 + 1.0E"

	def test_repr(self):
		z1 = DualNumber(2, 4)
		assert repr(z1) == "DualNumber(2, 4)"

if __name__ == "__main__":
	pytest.main()