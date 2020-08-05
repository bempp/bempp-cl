"""Unit tests for the dense assembler."""

import pytest
import bempp.api
import numpy as np
from bempp.api import function_space
from bempp.api.operators.potential import laplace, helmholtz, maxwell

scalar_spaces = [("DP", 0), ("DP", 1), ("P", 1)]
div_spaces = [("RWG", 0)]
curl_spaces = [("SNC", 0)]


@pytest.fixture
def points():
    return np.array([[2., 3.], [2., 1.], [1., 1.]])


@pytest.mark.parametrize("operator", [
    laplace.single_layer, laplace.double_layer])
@pytest.mark.parametrize("space_type", scalar_spaces)
def test_laplace_operators(points, operator, space_type):
    """Test dense assembler for the Laplace operators."""
    grid = bempp.api.shapes.regular_sphere(0)
    space = function_space(grid, *space_type)
    fun = bempp.api.GridFunction(
        space, coefficients=np.random.rand(space.global_dof_count))

    operator(space, points).evaluate(fun)


@pytest.mark.parametrize("operator", [
    helmholtz.single_layer, helmholtz.double_layer])
@pytest.mark.parametrize("space_type", scalar_spaces)
@pytest.mark.parametrize("wavenumber", [2.5, 2.5 + 1j])
def test_helmholtz_operators(points, operator, wavenumber, space_type):
    """Test dense assembler for the Helmholtz operators."""
    grid = bempp.api.shapes.regular_sphere(0)
    space = function_space(grid, *space_type)
    fun = bempp.api.GridFunction(
        space, coefficients=np.random.rand(space.global_dof_count))

    operator(space, points, wavenumber).evaluate(fun)


@pytest.mark.parametrize("assembler", ["fmm", "dense"])
@pytest.mark.parametrize("operator", [
    maxwell.electric_field, maxwell.magnetic_field])
@pytest.mark.parametrize("space_type", div_spaces)
@pytest.mark.parametrize("wavenumber", [2.5, 2.5 + 1j])
def test_maxwell_operators(points, operator, wavenumber, space_type, assembler):
    """Test dense assembler for the Helmholtz operators."""
    grid = bempp.api.shapes.regular_sphere(0)
    space = function_space(grid, *space_type)
    fun = bempp.api.GridFunction(
        space, coefficients=np.random.rand(space.global_dof_count))

    operator(space, points, wavenumber, assembler=assembler).evaluate(fun)
