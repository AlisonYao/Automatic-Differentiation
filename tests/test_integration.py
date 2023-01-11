import pytest
import sys, os
import numpy as np

sys.path.append("../")
from best_autodiff.forward import Forward
from best_autodiff.reverse import Reverse
from best_autodiff import functions as f 
from best_autodiff import rfunctions as rf

class TestIntegration():
	array_types = (list, np.ndarray)

	def isclose(self, x, y, tolerance=0.0001):
		if isinstance(x, self.array_types):
			return np.all(np.isclose(x, y))

		else:
			return abs(x - y) < tolerance

	def check_value(self, fun_fw, fun_rv, inputs, outputs, x):
		fw = Forward(fun_fw, inputs, outputs)
		rv = Reverse(fun_rv, inputs, outputs)

		fw_val, fw_der = fw(x)
		rv_val, rv_der = rv(x)

		return self.isclose(fw_val, rv_val) and self.isclose(fw_der, rv_der)

	def test_1(self):
		# 3 inputs 1 output
		def f1(x):
			return (f.log(f.exp(x[0]), np.e) * f.log(x[1], 2)) + f.tan(x[2])

		def f2(x):
			return (rf.log(rf.exp(x[0]), np.e) * rf.log(x[1], 2)) + rf.tan(x[2])

		inputs = 3 
		outputs = 1
		x = np.array([10, 10, 10])
		assert self.check_value(f1, f2, inputs, outputs, x)

	def test_2(self):
		# 1 input 1 output
		def f1(x):
			return x[0]**2

		inputs = 1 
		outputs = 1
		x = 5
		assert self.check_value(f1, f1, inputs, outputs, x)

	def test_3(self):
		# 1 input multiple outputs
		def f1(x):
			return f.exp(x[0])

		def f2(x):
			return rf.exp(x[0])

		def f3(x):
			return f.sin(x[0])

		def f4(x):
			return rf.sin(x[0])

		inputs = 1 
		outputs = 2
		x = 0
		assert self.check_value([f1, f3], [f2, f4], inputs, outputs, x)

	def test_4(self):
		# multiple inputs multiple outputs
		def f1(x):
			return x[0] * x[1]

		def f2(x):
			return x[0] + x[1]

		inputs = 2 
		outputs = 2
		x = np.array([3, 4])
		assert self.check_value([f1, f2], [f1, f2], inputs, outputs, x)

	def test_5(self):
		def f(x):
			return x[0] * x[1]

		inputs = 2 
		outputs = 1 
		x = np.array([10, 20])
		assert self.check_value(f, f, inputs, outputs, x)

	def test_6(self):
		def f1(x):
			return f.cos(x[0]) / f.exp(x[0])

		def f2(x):
			return rf.cos(x[0]) / rf.exp(x[0])

		inputs = 2 
		outputs = 1 
		x = np.array([np.pi, np.pi])
		assert self.check_value(f1, f2, inputs, outputs, x)

	def test_7(self):
		def f1(x):
			return f.sinh(x[0])

		def f2(x):
			return f.sqrt(x[0]) - f.log(x[0], np.e)

		def f3(x):
			return rf.sinh(x[0])

		def f4(x):
			return rf.sqrt(x[0]) - rf.log(x[0], np.e)

		inputs = 1
		outputs = 2
		x = np.array([10])
		assert self.check_value([f1, f2], [f3, f4], inputs, outputs, x)

	def test_8(self):
		def f1(x):
			return (f.log(f.exp(x[0]), np.e) * f.log(x[1], 2)) + f.tan(x[2])

		def f2(x):
			return (rf.log(rf.exp(x[0]), np.e) * rf.log(x[1], 2)) + rf.tan(x[2])

		inputs = 3
		outputs = 1
		x = np.array([10, 5, 0])
		assert self.check_value(f1, f2, inputs, outputs, x)

	def test_9(self):
		def f1(x):
			return f.root(x[0], 4)

		def f2(x):
			return f.arcsin(x[1]) / f.logistic(x[0]) * f.logistic(x[2]**4)

		def f3(x):
			return rf.root(x[0], 4)

		def f4(x):
			return rf.arcsin(x[1]) / rf.logistic(x[0]) * rf.logistic(x[2]**4)

		inputs = 3
		outputs = 2
		x = np.array([81, 0, 3])
		assert self.check_value([f1, f2], [f3, f4], inputs, outputs, x)

	def test_10(self):
		def f1(x):
			return 2*x[0] + f.tan(x[0])

		def f2(x):
			return 2*x[0] + rf.tan(x[0])

		inputs = 2
		outputs = 1
		x = np.array([5, 10])
		assert self.check_value(f1, f2, inputs, outputs, x)

	def test_11(self):
		def f1(x):
			return f.log(x[0], 10)

		def f2(x):
			return rf.log(x[0], 10)

		inputs = 1
		outputs = 1
		x = np.array([100])
		# x = 100
		assert self.check_value(f1, f2, inputs, outputs, x)

	def test_12(self):
		def f1(x):
			return x[0]**x[1]

		def f2(x):
			return f.sqrt(f.logistic(x[2]))

		def f3(x):
			return x[0]**x[1]

		def f4(x):
			return rf.sqrt(rf.logistic(x[2]))

		inputs = 3
		outputs = 2
		x = np.array([50, 2, 100])
		assert self.check_value([f1, f2], [f3, f4], inputs, outputs, x)

	def test_13(self):
		def f1(x):
			return f.sinh(x[0]) - f.cosh(x[1])

		def f2(x):
			return rf.sinh(x[0]) - rf.cosh(x[1])

		inputs = 2
		outputs = 1
		x = np.array([0, 0])
		assert self.check_value([f1], [f2], inputs, outputs, x)

	def test_14(self): 
		def f1(x):
			return 5 + f.tanh(x[0])

		def f2(x):
			return 5 - f.arcsin(x[1])

		def f3(x):
			return 5 / f.arccos(x[2])

		def f4(x):
			return 5 * f.arctan(x[3])

		def f5(x):
			return 5 + rf.tanh(x[0])

		def f6(x):
			return 5 - rf.arcsin(x[1])

		def f7(x):
			return 5 / rf.arccos(x[2])

		def f8(x):
			return 5 * rf.arctan(x[3])

		inputs = 4
		outputs = 4
		x = np.array([0, 1/4, 0, 3])

		assert self.check_value([f1, f2, f3, f4], [f5, f6, f7, f8], inputs, outputs, x)

	def test_15(self):
		def f3(x):
			return -5 / -f.arccos(x[0])

		def f7(x):
			return -5 / -rf.arccos(x[0])

		inputs = 1
		outputs = 1
		x = np.array([0])
		assert self.check_value(f3, f7, inputs, outputs, x)

	def test_16(self):
		def f1(x):
			return f.sin(x[0])**3 + f.sqrt(f.cos(x[0]))

		def f2(x):
			return rf.sin(x[0])**3 + rf.sqrt(rf.cos(x[0]))

		inputs = 1
		outputs = 1
		x = np.array([2*np.pi])
		assert self.check_value(f1, f2, inputs, outputs, x)

	def test_17(self):
		def f1(x):
			return 3*x[0] + f.exp(x[0]) * f.sin(2*x[0]) + 1

		def f2(x):
			return 3*x[0] + rf.exp(x[0]) * rf.sin(2*x[0]) + 1

		inputs = 1
		outputs = 1
		x = np.pi/4
		assert self.check_value(f1, f2, inputs, outputs, x)

	def test_18(self):
		def f1(x):
			return (f.sqrt(x[0])/f.log(x[1]))*x[0]

		def f2(x):
			return (rf.sqrt(x[0])/rf.log(x[1]))*x[0]

		inputs = 2
		outputs = 1
		x = np.array([16, 2])
		assert self.check_value(f1, f2, inputs, outputs, x)

	def test_19(self):
		def f0(x):
			return x[0]**3

		def f1(x):
			return x[0]

		def f2(x):
			return f.logistic(x[0])

		def f3(x):
			return x[0]**3

		def f4(x):
			return x[0]

		def f5(x):
			return rf.logistic(x[0])

		inputs = 1
		outputs = 3
		x = np.array([5])
		assert self.check_value([f0, f1, f2], [f3, f4, f5], inputs, outputs, x)

	def test_20(self):
		def f1(x):
			return f.exp(-(f.sin(x[0]) - f.cos(x[1]))**2)

		def f2(x):
			return f.sin(-f.log(x[0])**2 + f.tan(x[2]))

		def f3(x):
			return rf.exp(-(rf.sin(x[0]) - rf.cos(x[1]))**2)

		def f4(x):
			return rf.sin(-rf.log(x[0])**2 + rf.tan(x[2]))

		inputs = 3
		outputs = 2
		x = np.array([10, 1, 5])
		assert self.check_value([f1, f2], [f3, f4], inputs, outputs, x)

	def test_21(self):
		def f(x):
			return x[0] - x[1] - x[2]

		inputs = 3
		outputs = 1
		x = np.array([10, 1, 5])
		assert self.check_value([f], [f], inputs, outputs, x)

	def test_22(self):
		def f1(x):
			return 1 - x[0]

		def f2(x):
			return x[0] - x[1]

		inputs = 2
		outputs = 2
		x = np.array([1.5, 16.2])
		assert self.check_value([f1, f2], [f1, f2], inputs, outputs, x)

	def test_23(self):
		def f1(x):
			return 2*x[0] + x[1]

		def f2(x):
			return 3 + 2*x[0]

		inputs = 2
		outputs = 2
		x = np.array([-7.5, 3])
		assert self.check_value([f1, f2], [f1, f2], inputs, outputs, x)

	def test_24(self):
		def f1(x):
			return x[0]*x[1]

		def f2(x):
			return x[1]*x[0]

		def f3(x):
			return x[0]/x[1]

		def f4(x):
			return 1/x[0]

		def f5(x):
			return x[0]**3

		def f6(x):
			return 2**x[0]

		def f7(x):
			return -x[0]

		def f8(x):
			return f.sqrt(x[0])

		def f9(x):
			return f.log(x[0])

		def f10(x):
			return f.log(x[0], base=2)

		def f11(x):
			return rf.sqrt(x[0])

		def f12(x):
			return rf.log(x[0])

		def f13(x):
			return rf.log(x[0], base=2)

		inputs = 2
		outputs = 10
		x = np.array([7.5, 3])
		assert self.check_value([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10],\
			[f1, f2, f3, f4, f5, f6, f7, f11, f12, f13], inputs, outputs, x)

	def test_25(self):
		def f1(x):
			return f.exp(x[0]) + f.cos(x[1])

		def f2(x):
			return f.sin(x[0]) + f.tan(x[1])

		def f3(x):
			return rf.exp(x[0]) + rf.cos(x[1])

		def f4(x):
			return rf.sin(x[0]) + rf.tan(x[1])

		inputs = 2
		outputs = 2
		x = np.array([np.pi, np.pi/8])
		assert self.check_value([f1, f2], [f3, f4], inputs, outputs, x)

	def test_26(self):
		def f1(x):
			return (f.arcsin(x[0]) ** f.arccos(x[1])) / f.arctan(x[2])

		def f2(x):
			return f.exp(f.sinh(x[0]) * f.cosh(x[1])) + f.logistic(x[3])

		def f3(x):
			return (rf.arcsin(x[0]) ** rf.arccos(x[1])) / rf.arctan(x[2])

		def f4(x):
			return rf.exp(rf.sinh(x[0]) * rf.cosh(x[1])) + rf.logistic(x[3])

		inputs = 4
		outputs = 2
		x = np.array([0.1, 0.2, 0.3, 0.4])
		assert self.check_value([f1, f2], [f3, f4], inputs, outputs, x)