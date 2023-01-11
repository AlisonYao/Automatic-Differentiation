import pytest
import sys
import numpy as np

sys.path.append("../../team02")
from best_autodiff.graph import Graph
from best_autodiff.rfunctions import *

class TestGraph:
    """
    Test Graph class for implementation
    """
    def test_init(self):
        # test invalid inputs
        with pytest.raises(TypeError):
            Graph('string')
        with pytest.raises(TypeError):
            Graph(Graph([1, 2, 3]))
        with pytest.raises(TypeError):
            Graph([[1,2,3], 1, 2, 3])

        z1 = Graph(3)
        assert (z1.value == np.array([3])).all()
        assert z1.local_gradients == ()

        z2 = Graph([1,2,3])
        assert (z2.value == np.array([1,2,3])).all()
        assert z2.local_gradients == ()
        
    def test_add(self):
        # testing invalid input
        with pytest.raises(TypeError):
            Graph(3) + 'string'
        with pytest.raises(AssertionError):
            Graph(3) + Graph([1,2,3])
        with pytest.raises(AssertionError):
            Graph([3, 4]) + Graph(3)

        z1 = Graph(10) + Graph(5)
        assert (z1.value == np.array([15])).all()

        z2 = Graph([5.0, 2.0]) + 2.0
        assert (z2.value == np.array([7.0, 4.0])).all()

        z3 = [5, 6] + Graph([1, 3])
        assert (z3.value == np.array([6, 9])).all()

    def test_radd(self):
        # testing invalid input
        with pytest.raises(TypeError):
            'string' + Graph(3)
        with pytest.raises(AssertionError):
            Graph([1,2,3]) + Graph(3)
        with pytest.raises(AssertionError):
            Graph(3) + Graph([3, 4])

        z1 = Graph(10) + Graph(5)
        assert (z1.value == np.array([15])).all()

        z2 = 2.0 + Graph([5.0, 2.0])
        assert (z2.value == np.array([7.0, 4.0])).all()

        z3 = Graph([5, 6]) + [1, 3]
        assert (z3.value == np.array([6, 9])).all()

    def test_sub(self):
        # testing invalid input
        with pytest.raises(TypeError):
            Graph(3) - 'string'
        with pytest.raises(AssertionError):
            Graph(3) - Graph([1,2,3])
        with pytest.raises(AssertionError):
            Graph([3, 4]) - Graph(3)

        z1 = Graph(10) - Graph(5)
        assert (z1.value == np.array([5])).all()

        z2 = Graph([5.0, 2.0]) - 2.0
        assert (z2.value == np.array([3.0, 0.0])).all()

        z3 = [5, 6] - Graph([1, 3])
        assert (z3.value == np.array([4, 3])).all()
    
        z4 = Graph([5, 6]) - Graph([1, 3])
        assert (z4.value == np.array([4, 3])).all()

    def test_rsub(self):
        # testing invalid input
        with pytest.raises(TypeError):
            'string' - Graph(3)
        with pytest.raises(AssertionError):
            Graph([1,2,3]) - Graph(3)
        with pytest.raises(AssertionError):
            Graph(3) - Graph([3, 4])

        z1 = Graph(10) - Graph(5)
        assert (z1.value == np.array([5])).all()

        z2 = 2.0 - Graph([5.0, 2.0])
        assert (z2.value == np.array([-3.0, 0.0])).all()

        z3 = Graph([5, 6]) - [1, 3]
        assert (z3.value == np.array([4, 3])).all()

    def test_mul(self):
        # testing invalid input
        with pytest.raises(TypeError):
            Graph(3) * 'string'
        with pytest.raises(AssertionError):
            Graph(3) * Graph([1,2,3])
        with pytest.raises(AssertionError):
            Graph([3, 4]) * Graph(3)

        z1 = Graph(10) * Graph(5)
        assert (z1.value == np.array([50])).all()

        z2 = Graph([5.0, 2.0]) * 2.0
        assert (z2.value == np.array([10.0, 4.0])).all()

        z3 = [5, 6] * Graph([1, 3])
        assert (z3.value == np.array([5, 18])).all()

    def test_rmul(self):
        # testing invalid input
        with pytest.raises(TypeError):
            'string' * Graph(3)
        with pytest.raises(AssertionError):
            Graph([1,2,3]) * Graph(3)
        with pytest.raises(AssertionError):
            Graph(3) * Graph([3, 4])

        z1 = Graph(10) * Graph(5)
        assert (z1.value == np.array([50])).all()

        z2 = 2.0 * Graph([5.0, 2.0])
        assert (z2.value == np.array([10.0, 4.0])).all()

        z3 = Graph([5, 6]) * [1, 3]
        assert (z3.value == np.array([5, 18])).all()

    def test_pow(self):
        # testing invalid input
        with pytest.raises(TypeError):
            Graph(3) ** 'string'
        with pytest.raises(AssertionError):
            Graph(3) ** Graph([1,2,3])
        with pytest.raises(AssertionError):
            Graph([3, 4]) ** Graph(3)
        with pytest.raises(AssertionError):
            Graph(3) ** [3, 6]

        z1 = Graph(3) ** Graph(2)
        assert (z1.value == np.array([9])).all()

        z2 = Graph([3.0, 4.0]) ** 2.0
        assert (z2.value == np.array([9.0, 16.0])).all()

        z3 = Graph([5, 6]) ** [1, 2]
        assert (z3.value == np.array([5, 36])).all()
    
    def test_rpow(self):
        # testing invalid input
        with pytest.raises(TypeError):
            'string' ** Graph(3)
        with pytest.raises(AssertionError):
            Graph([1,2,3]) ** Graph(3)
        with pytest.raises(AssertionError):
            Graph(3) ** Graph([3, 4])
        with pytest.raises(AssertionError):
            [3, 6] ** Graph(3)

        z1 = 2.0 ** Graph([3.0, 4.0])
        assert (z1.value == np.array([8.0, 16.0])).all()

        z2 = [5, 6] ** Graph([1, 2])
        assert (z2.value == np.array([5, 36])).all()

    def test_truediv(self):
        # testing invalid input
        with pytest.raises(TypeError):
            'string' / Graph(3)
        with pytest.raises(TypeError):
            Graph(3) / 'string'
        with pytest.raises(AssertionError):
            Graph([1,2,3]) / Graph(3)
        with pytest.raises(AssertionError):
            Graph(3) / Graph([3, 4])
        with pytest.raises(ZeroDivisionError):
            Graph(3) / 0
        
        z1 = Graph(10) / Graph(5)
        assert (z1.value == np.array([2])).all()

        z2 = Graph([5.0, 2.0]) / 2.0
        assert (z2.value == np.array([2.5, 1.0])).all()

        z3 = Graph([2, 6]) / [2, 3]
        assert (z3.value == np.array([1, 2])).all()

    def test_rtruediv(self):
        # testing invalid input
        with pytest.raises(TypeError):
            'string' / Graph(3)
        with pytest.raises(AssertionError):
            Graph([1,2,3]) / Graph(3)
        with pytest.raises(AssertionError):
            Graph(3) / Graph([3, 4])
        with pytest.raises(ZeroDivisionError):
            3 / Graph(0)
        
        z1 = Graph(10) / Graph(5)
        assert (z1.value == np.array([2])).all()

        z2 = 10 / Graph([5.0, 2.0])
        assert (z2.value == np.array([2.0, 5.0])).all()

        z3 = [2, 6] / Graph([2, 3])
        assert (z3.value == np.array([1, 2])).all()

        z4 = Graph([2, 6]) / Graph([2, 3])
        assert (z4.value == np.array([1, 2])).all()

    def test_lt(self):
        # testing invalid input
        with pytest.raises(TypeError):
            'string' < Graph(3)
        with pytest.raises(AssertionError):
            Graph([1,2,3]) < Graph(3)
        with pytest.raises(AssertionError):
            Graph(3) < Graph([3, 4])

        assert (Graph([1,2]) < 3).all()
        assert Graph(2) < 12
        assert (Graph([1,5]) < [5,8]).all()
        assert (Graph([1,5]) < Graph([5, 8])).all()
    
    def test_gt(self):
        # testing invalid input
        with pytest.raises(TypeError):
            'string' > Graph(3)
        with pytest.raises(AssertionError):
            Graph([1,2,3]) > Graph(3)
        with pytest.raises(AssertionError):
            Graph(3) > Graph([3, 4])

        assert (Graph([1,2]) > 0).all()
        assert Graph(2) > 0
        assert (Graph([1,5]) > [0,0]).all()
        assert (Graph([1,5]) > Graph([0, 0])).all()

    def test_le(self):
        # testing invalid input
        with pytest.raises(TypeError):
            'string' <= Graph(3)
        with pytest.raises(AssertionError):
            Graph([1,2,3]) <= Graph(3)
        with pytest.raises(AssertionError):
            Graph(3) <= Graph([3, 4])

        assert (Graph([1,2]) <= 2).all()
        assert Graph(2) <= 12
        assert (Graph([1,5]) <= [1,8]).all()
        assert (Graph([1,5]) <= Graph([5, 8])).all()

    def test_ge(self):
        # testing invalid input
        with pytest.raises(TypeError):
            'string' >= Graph(3)
        with pytest.raises(AssertionError):
            Graph([1,2,3]) >= Graph(3)
        with pytest.raises(AssertionError):
            Graph(3) >= Graph([3, 4])

        assert (Graph([1,2]) >= 1).all()
        assert Graph(2) >= 0
        assert (Graph([1,5]) >= [1,0]).all()
        assert (Graph([1,5]) >= Graph([0, 5])).all()

    def test_eq(self):
        # testing invalid input
        with pytest.raises(TypeError):
            'string' == Graph(3)
        with pytest.raises(AssertionError):
            Graph([1,2,3]) == Graph(3)
        with pytest.raises(AssertionError):
            Graph(3) == Graph([3, 4])

        assert (Graph([1,1]) == 1).all()
        assert Graph(2) == 2
        assert (Graph([1,5]) == [1,5]).all()
        assert (Graph([1.0,5.0]) == Graph([1, 5])).all()

    def test_ne(self):
        # testing invalid input
        with pytest.raises(TypeError):
            'string' != Graph(3)
        with pytest.raises(AssertionError):
            Graph([1,2,3]) != Graph(3)
        with pytest.raises(AssertionError):
            Graph(3) != Graph([3, 4])

        assert (Graph([1,1]) != 5).any()
        assert Graph(2) != 7
        assert (Graph([1,5]) != np.array([1,1])).any()
        assert (Graph([1.0,5.0]) != Graph([1.7, 1])).any()
    
    def test_neg(self):
        assert (-Graph([1, 5]) == np.array([-1,-5])).all()
        assert (-Graph([1.0, 5.0]) == Graph([-1.0, -5.0])).all()
    
    def test_str(self):
        assert str(Graph([1,2])) == 'Graph([1 2])'

    def test_repr(self):
        assert repr(Graph([1,2])) == 'Graph([1 2])'