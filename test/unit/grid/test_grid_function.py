import numpy as np
from math import sqrt

import bempp.api


def test_interpolation_real_callable():
    @bempp.api.real_callable
    def f(x, n, d, r):
        r[:] = x[0]

    grid = bempp.api.shapes.cube(h=0.5)
    space = bempp.api.function_space(grid, "P", 1)
    fun = bempp.api.GridFunction(space, fun=f)

    assert np.isclose(fun.l2_norm(), sqrt(7 / 3))


def test_interpolation_complex_callable():
    @bempp.api.complex_callable
    def f(x, n, d, r):
        r[:] = x[0] + 1j * x[0]

    grid = bempp.api.shapes.cube(h=0.5)
    space = bempp.api.function_space(grid, "P", 1)
    fun = bempp.api.GridFunction(space, fun=f)

    assert np.isclose(fun.l2_norm(), sqrt(14 / 3))


def test_interpolation_python_callable():
    # This array can't be assembled by Numba, so jit cannot be used
    parameters = [1, None, 1.0]

    @bempp.api.complex_callable(jit=False)
    def f(x, n, d, r):
        r[:] = x[0] * parameters[0] + 1j * x[1] * parameters[2]

    grid = bempp.api.shapes.cube(h=0.5)
    space = bempp.api.function_space(grid, "P", 1)
    fun = bempp.api.GridFunction(space, fun=f)

    assert np.isclose(fun.l2_norm(), sqrt(14 / 3))
