import pytest
import sys
import numpy as np

sys.path.append("../../team02")
from best_autodiff.graph import Graph
from best_autodiff.reverse import Reverse
from best_autodiff.rfunctions import *

class TestReverse:

    def test_init(self):
        with pytest.raises(TypeError):
            Reverse(1, 1, 1)
        with pytest.raises(TypeError):
            Reverse([lambda x:x**2, 4], 1, 1)
        with pytest.raises(TypeError):
            Reverse(lambda x:x**2, 'string', 1)
        with pytest.raises(TypeError):
            Reverse(lambda x:x**2, 1, 'string')
        
        z1 = Graph(3)
        assert (z1.value == np.array([3])).all()
        assert z1.local_gradients == ()

        z2 = Graph([1,2,3])
        assert (z2.value == np.array([1,2,3])).all()
        assert z2.local_gradients == ()
    
    def test_mix_1(self):
        """
        R -> R
        """
        def f(x):
            return sin(x[0])**3 + sqrt(cos(x[0]))
        x = 1
        ad = Reverse(f, 1, 1)
        ad.evaluate(x)
        J = ad.get_jacobian(x)
        assert np.allclose(J, np.array([0.5753328132280637]))

    def test_mix_2(self):
        """
        R -> R
        """
        def f(x):
            return 3*x[0] + 2 ** x[0] * sin(2*x[0]) + 1
        x = np.array([3])
        ad = Reverse(f, 1, 1)
        ad.evaluate(x)
        J = ad.get_jacobian(x)
        assert np.allclose(J, np.array([16.813316068155153]))
    
    def test_mix_3(self):
        """
        Rm -> R
        """
        def f(x):
            return (sqrt(x[0])/log(x[1]))*x[0]
        x = [5, 6]
        ad = Reverse(f, 2, 1)
        ad.evaluate(x)
        J = ad.get_jacobian(x)
        assert np.allclose(J, np.array([1.8719599499003818, -0.5804226336209687]))

    def test_mix_3(self):
        """
        R -> Rn
        """
        def f0(x):
            return x[0]**3
        def f1(x):
            return x[0]
        def f2(x):
            return logistic(x[0])

        f = [f0, f1, f2]
        x = 3
        ad = Reverse(f, 1, 3)
        ad.evaluate(x)
        J = ad.get_jacobian(x)
        assert np.allclose(J, np.array([[27],[1],[0.045176659730912144]]))


    def test_mix_4(self):
        """
        Rm -> Rn
        """
        def f1(x):
            return exp(-(sin(x[0]) - cos(x[1]))**2)

        def f2(x):
            return sin(-log(x[0])**2 + tan(x[2]))
        f  = [f1, f2]
        x = [1, 1, 1]
        ad = Reverse(f, 3, 2)
        ad.evaluate(x)
        J = ad.get_jacobian(x)
        assert np.allclose(J[0], np.array([-0.29722477, -0.46290015,  0]))
        assert np.allclose(J[1], np.array([0, 0, 0.04586154]))

    def test_mix_5(self):
        """
        R -> R
        """
        def f(x):
            return x[0] - x[1] - x[2]
        x = np.array([1, 2, 3])
        ad = Reverse(f, 3, 1)
        ad.evaluate(x)
        J = ad.get_jacobian(x)
        assert np.allclose(J, np.array([1, -1, -1]))

    def test_add(self):
        def f(x):
            return 2*x[0] + x[1]
        x = [3, 4]
        ad = Reverse(f, 2, 1)
        ad.evaluate(x)
        J = ad.get_jacobian(x)
        assert np.allclose(J, np.array([2, 1]))

    def test_radd(self):
        def f(x):
            return 3 + 2*x[0]
        x = 3
        ad = Reverse(f, 1, 1)
        ad.evaluate(x)
        J = ad.get_jacobian(x)
        assert np.allclose(J, np.array([2]))

    def test_sub(self):
        def f(x):
            return x[0] - x[1]
        x = [3, 10]
        ad = Reverse(f, 2, 1)
        ad.evaluate(x)
        J = ad.get_jacobian(x)
        assert np.allclose(J, np.array([1, -1]))

    def test_rsub(self):
        def f(x):
            return 1 - x[0]
        x = 3
        ad = Reverse(f, 1, 1)
        ad.evaluate(x)
        J = ad.get_jacobian(x)
        assert np.allclose(J, np.array([-1]))

    def test_mul(self):
        def f(x):
            return x[0]*x[1]
        x = [3, 4]
        ad = Reverse(f, 2, 1)
        ad.evaluate(x)
        J = ad.get_jacobian(x)
        assert np.allclose(J, np.array([4, 3]))

    def test_rmul(self):
        def f(x):
            return x[1]*x[0]
        x = [3, 4]
        ad = Reverse(f, 2, 1)
        ad.evaluate(x)
        J = ad.get_jacobian(x)
        assert np.allclose(J, np.array([4, 3]))

    def test_truediv(self):
        def f(x):
            return x[0]/x[1]
        x = [2, 10]
        ad = Reverse(f, 2, 1)
        ad.evaluate(x)
        J = ad.get_jacobian(x)
        assert np.allclose(J, np.array([0.1, -0.02]))

    def test_rtruediv(self):
        def f(x):
            return 1/x[0]
        x = 10
        ad = Reverse(f, 1, 1)
        ad.evaluate(x)
        J = ad.get_jacobian(x)
        assert np.allclose(J, np.array([-0.01]))

    def test_pow(self):
        def f(x):
            return x[0]**3
        x = 2
        ad = Reverse(f, 1, 1)
        ad.evaluate(x)
        J = ad.get_jacobian(x)
        assert np.allclose(J, np.array([12]))

    def test_rpow(self):
        def f(x):
            return 2**x[0]
        x = 2
        ad = Reverse(f, 1, 1)
        ad.evaluate(x)
        J = ad.get_jacobian(x)
        assert np.allclose(J, np.array([2.772588722239781]))

        x = 3
        ad = Reverse(f, 1, 1)
        ad.evaluate(x)
        J = ad.get_jacobian(x)
        assert np.allclose(J, np.array([5.545177444479562]))

    def test_neg(self):
        def f(x):
            return -x[0]
        x = 2
        ad = Reverse(f, 1, 1)
        ad.evaluate(x)
        J = ad.get_jacobian(x)
        assert np.allclose(J, np.array([-1]))
    
    def test_sqrt(self):
        def f(x):
            return sqrt(x[0])
        x = 16
        ad = Reverse(f, 1, 1)
        ad.evaluate(x)
        J = ad.get_jacobian(x)
        assert np.allclose(J, np.array([0.125]))
    
    def test_log(self):
        def f(x):
            return log(x[0])
        x = 10
        ad = Reverse(f, 1, 1)
        ad.evaluate(x)
        J = ad.get_jacobian(x)
        assert np.allclose(J, np.array([0.1]))

        def f(x):
            return log(x[0], base=2)
        x = 10
        ad = Reverse(f, 1, 1)
        ad.evaluate(x)
        J = ad.get_jacobian(x)
        assert np.allclose(J, np.array([0.14426950408]))

    def test_root(self):
        def f(x):
            return root(x[0])
        x = 4
        ad = Reverse(f, 1, 1)
        ad.evaluate(x)
        J = ad.get_jacobian(x)
        assert np.allclose(J, np.array([0.25]))

        def f(x):
            return root(x[0], n=3)
        x = 4
        ad = Reverse(f, 1, 1)
        ad.evaluate(x)
        J = ad.get_jacobian(x)
        assert np.allclose(J, np.array([0.13228342099]))
    
    def test_exp(self):
        def f(x):
            return exp(x[0])
        x = 2
        ad = Reverse(f, 1, 1)
        ad.evaluate(x)
        J = ad.get_jacobian(x)
        assert np.allclose(J, np.array([7.3890560989306495]))

    def test_sin(self):
        def f(x):
            return sin(x[0])
        x = 3
        ad = Reverse(f, 1, 1)
        ad.evaluate(x)
        J = ad.get_jacobian(x)
        assert np.allclose(J, np.array([-0.9899924966]))

    def test_cos(self):
        def f(x):
            return cos(x[0])
        x = 3
        ad = Reverse(f, 1, 1)
        ad.evaluate(x)
        J = ad.get_jacobian(x)
        assert np.allclose(J, np.array([-0.1411200080598672]))

    def test_tan(self):
        def f(x):
            return tan(x[0])
        x = 0
        ad = Reverse(f, 1, 1)
        ad.evaluate(x)
        J = ad.get_jacobian(x)
        assert np.allclose(J, np.array([1]))

    def test_arcsin(self):
        def f(x):
            return arcsin(x[0])
        x = -0.5
        ad = Reverse(f, 1, 1)
        ad.evaluate(x)
        J = ad.get_jacobian(x)
        assert np.allclose(J, np.array([1.1547005383792517]))
    
    def test_arccos(self):
        def f(x):
            return arccos(x[0])
        x = -0.5
        ad = Reverse(f, 1, 1)
        ad.evaluate(x)
        J = ad.get_jacobian(x)
        assert np.allclose(J, np.array([-1.1547005383792517]))

    def test_arctan(self):
        def f(x):
            return arctan(x[0])
        x = -0.5
        ad = Reverse(f, 1, 1)
        ad.evaluate(x)
        J = ad.get_jacobian(x)
        assert np.allclose(J, np.array([0.8]))
    
    def test_sinh(self):
        def f(x):
            return sinh(x[0])
        x = 2
        ad = Reverse(f, 1, 1)
        ad.evaluate(x)
        J = ad.get_jacobian(x)
        assert np.allclose(J, np.array([3.7621956910836314]))

    def test_cosh(self):
        def f(x):
            return cosh(x[0])
        x = -1
        ad = Reverse(f, 1, 1)
        ad.evaluate(x)
        J = ad.get_jacobian(x)
        assert np.allclose(J, np.array([-1.1752011936438014]))
    
    def test_tanh(self):
        def f(x):
            return tanh(x[0])
        x = -1
        ad = Reverse(f, 1, 1)
        ad.evaluate(x)
        J = ad.get_jacobian(x)
        assert np.allclose(J, np.array([0.4199743416140261]))

    def test_logistic(self):
        def f(x):
            return logistic(x[0])
        x = 3
        ad = Reverse(f, 1, 1)
        ad.evaluate(x)
        J = ad.get_jacobian(x)
        assert np.allclose(J, np.array([0.045176659730912144]))
    
    def test_evaluate(self):
        with pytest.raises(TypeError):
            ad = Reverse(lambda x:x**2, 1, 1)
            ad.evaluate('string')
        with pytest.raises(TypeError):
            ad = Reverse(lambda x:x**2, 1, 1)
            ad.evaluate(['string', 1])

# if __name__ == "__main__":
#     pytest.main()
