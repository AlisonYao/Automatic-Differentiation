import pytest
import sys
import numpy as np

sys.path.append("../../team02")
from best_autodiff.graph import Graph
from best_autodiff.rfunctions import *

class TestFunctions:
    """
    Test class for best_autodiff package rfunctions
    """
    def test_log(self):
        # invalid inputs
        with pytest.raises(TypeError):
            log(3, 3)
        with pytest.raises(TypeError):
            log(Graph(2), "3")
        with pytest.raises(TypeError):
            log((2,3), "3")

        assert log(Graph(np.e)) == 1.0
        assert (log(Graph(23), 5).value == np.array([1.9481920934663797])).all()
        assert np.allclose(log(Graph([2,3]), 5).value, np.array([0.43067655807339306, 0.6826061944859854]))
    
    def test_root(self):
        # invalid inputs
        with pytest.raises(TypeError):
            root(3, 3)
        with pytest.raises(TypeError):
            root(Graph(2), "3")
        with pytest.raises(TypeError):
            root((2,3), "3")
        with pytest.raises(ValueError):
            root(Graph(-2), 10)
        
        assert (root(Graph([1,2,3])) == np.array([1.0, 1.4142135623730951, 1.7320508075688772])).all()
        assert (root(Graph(5), 10) == np.array([1.174618943088019])).all()

    def test_sin(self):
        # invalid inputs
        with pytest.raises(TypeError):
            sin(4)
        with pytest.raises(TypeError):
            sin('string')
        with pytest.raises(TypeError):
            sin([1,2,3])
        
        assert np.allclose(sin(Graph(5)).value, np.array([-0.9589242746631385]))
        assert np.allclose(sin(Graph([0,1,2,3])).value, np.array([0, 0.8414709848078965, 0.9092974268256817, 0.1411200080598672]))

    def test_sqrt(self):
        # invalid inputs
        with pytest.raises(TypeError):
            sqrt("string")
        with pytest.raises(ValueError):
            sqrt(Graph(-4))
        with pytest.raises(ValueError):
            sqrt(Graph([-4, 1]))

        assert sqrt(Graph(4.0)) == 2.0

        assert (sqrt(Graph([16, 36])).value == np.array([4, 6])).all()

    def test_exp(self):
        # invalid inputs
        with pytest.raises(TypeError):
            exp("string")

        assert exp(Graph(2.0)) == 7.3890560989306495
        assert (exp(Graph([2.0])) == np.array([7.3890560989306495])).all()

    def test_tan(self):
        # invalid inputs
        with pytest.raises(TypeError):
            tan(4)
        with pytest.raises(TypeError):
            tan('string')
        with pytest.raises(TypeError):
            tan([1,2,3])
        with pytest.raises(ValueError):
            tan(Graph([0,1,2,3,np.pi/2]))
        
        assert np.allclose(tan(Graph([0,1,2,3])).value, np.array([0, 1.557407724654902, -2.185039863261519, -0.1425465430742778]))
        assert np.allclose(tan(Graph(7)).value, np.array([0.8714479827243188]))

    def test_arctan(self):
        # invalid inputs
        with pytest.raises(TypeError):
            arctan(4)
        with pytest.raises(TypeError):
            arctan('string')
        with pytest.raises(TypeError):
            arctan([1,2,3])

        assert np.allclose(arctan(Graph(7)).value, np.array([1.4288992721907328]))
        assert np.allclose(arctan(Graph([1,2,3])).value, np.array([0.7853981633974483, 1.1071487177940906, 1.2490457723982544]))

    def test_cos(self):
        # invalid inputs
        with pytest.raises(TypeError):
            cos(4)
        with pytest.raises(TypeError):
            cos('string')
        with pytest.raises(TypeError):
            cos([1,2,3])
        
        assert np.allclose(cos(Graph([0,1,2,3])).value, np.array([1.0, 0.54030230586 , -0.41614683654 , -0.9899924966 ]))
        assert np.allclose(cos(Graph(7)).value, np.array([0.75390225434]))


    def test_arccos(self):
        # invalid inputs
        with pytest.raises(TypeError):
            arccos(0.4)
        with pytest.raises(TypeError):
            arccos('string')
        with pytest.raises(TypeError):
            arccos([1,2,3])
        with pytest.raises(ValueError):
            arccos(Graph(-1.5))

        assert np.allclose(arccos(Graph(0.5)).value, np.array([np.pi/3]))
        assert np.allclose(arccos(Graph([0, 1, -1])).value, np.array([np.pi/2, 0, np.pi]))
    
    def test_arcsin(self):
        # invalid inputs
        with pytest.raises(TypeError):
            arcsin(0.4)
        with pytest.raises(TypeError):
            arcsin('string')
        with pytest.raises(TypeError):
            arcsin([1,2,3])
        with pytest.raises(ValueError):
            arcsin(Graph(-1.5))

        assert np.allclose(arcsin(Graph(0.5)).value, np.array([np.pi/6]))
        assert np.allclose(arcsin(Graph([0, 1, -1])).value, np.array([0, np.pi/2, -np.pi/2]))

    def test_sinh(self):
        # invalid inputs
        with pytest.raises(TypeError):
            sinh(0.4)
        with pytest.raises(TypeError):
            sinh('string')
        with pytest.raises(TypeError):
            sinh([1,2,3])

        assert np.allclose(sinh(Graph(0.5)).value, np.array([0.52109530549]))
        assert np.allclose(sinh(Graph([0, 1, -1])).value, np.array([0, 1.17520119364, -1.17520119364]))
    
    def test_cosh(self):
        # invalid inputs
        with pytest.raises(TypeError):
            cosh(4)
        with pytest.raises(TypeError):
            cosh('string')
        with pytest.raises(TypeError):
            cosh([1,2,3])

        assert np.allclose(cosh(Graph(7)).value, np.array([548.3170351552121]))
        assert np.allclose(cosh(Graph([0,1,2])).value, np.array([1.0, 1.5430806348152437, 3.7621956910836314]))

    def test_tanh(self):
        # invalid inputs
        with pytest.raises(TypeError):
            tanh(4)
        with pytest.raises(TypeError):
            tanh('string')
        with pytest.raises(TypeError):
            tanh([1,2,3])

        assert np.allclose(tanh(Graph(7)).value, np.array([0.99999833694]))
        assert np.allclose(tanh(Graph([0, 1, -0.5])).value, np.array([0, 0.76159415595, -0.46211715726]))

    def test_logistic(self):
        # invalid inputs
        with pytest.raises(TypeError):
            logistic(4)
        with pytest.raises(TypeError):
            logistic('string')
        with pytest.raises(TypeError):
            logistic([1,2,3])

        assert np.allclose(logistic(Graph(7)).value, np.array([0.9990889488055994]))
        assert np.allclose(logistic(Graph([0,1,2])).value, np.array([0.5, 0.7310585786300049, 0.8807970779778823]))