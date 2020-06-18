import bempp.api
import numpy as np
import pytest


@pytest.mark.parametrize('fun_type', ["real", "complex"])
def test_jit_disabled(fun_type):
    def f(x, n, dom, res):
        res[:] = 1

    grid = bempp.api.shapes.cube(h=0.4)
    space = bempp.api.function_space(grid, "DP", 0)
    fun = bempp.api.GridFunction(space, fun=f, jit=False, function_type=fun_type)
    assert np.isclose(fun.l2_norm() ** 2, 6)


def test_real_callable():
    @bempp.api.real_callable
    def f(x, n, dom, res):
        res[:] = 1

    grid = bempp.api.shapes.cube(h=0.4)
    space = bempp.api.function_space(grid, "DP", 0)
    fun = bempp.api.GridFunction(space, fun=f)
    assert np.isclose(fun.l2_norm() ** 2, 6)


def test_complex_callable():
    @bempp.api.complex_callable
    def f(x, n, dom, res):
        res[:] = 1

    grid = bempp.api.shapes.cube(h=0.4)
    space = bempp.api.function_space(grid, "DP", 0)
    fun = bempp.api.GridFunction(space, fun=f)
    assert np.isclose(fun.l2_norm() ** 2, 6)
