"""Unit tests for the FMM assembler."""

# pylint: disable=redefined-outer-name
# pylint: disable=C0103

import pytest
import numpy as np
import bempp.api

pytestmark = pytest.mark.usefixtures("default_parameters", "helpers")

NPOINTS = 10
TOL = 2e-3
bempp.api.GLOBAL_PARAMETERS.fmm.expansion_order = 10
# bempp.api.GLOBAL_PARAMETERS.fmm.ncrit = 100


@pytest.fixture
def grid():
    return bempp.api.shapes.ellipsoid(1, 0.5, 0.3, h=0.15)


@pytest.fixture
def grid1():
    return bempp.api.shapes.ellipsoid(0.5, 0.5, 0.3, h=0.15)


@pytest.fixture
def grid2():
    return bempp.api.shapes.sphere(r=1.5, h=0.4)


@pytest.mark.parametrize("operator", [
    bempp.api.operators.boundary.laplace.single_layer])
def test_laplace_boundary_fmm(operator, grid):
    """Test Laplace boundary operators."""
    space = bempp.api.function_space(grid, "P", 1)

    rand = np.random.RandomState(0)
    vec = rand.rand(space.global_dof_count)

    dense = operator(space, space, space, assembler="dense").weak_form()
    fmm = operator(space, space, space, assembler="fmm").weak_form()

    np.testing.assert_allclose(dense @ vec, fmm @ vec, rtol=TOL)


@pytest.mark.parametrize("operator", [
    bempp.api.operators.potential.laplace.single_layer])
def test_laplace_potential_fmm(operator, grid):
    """Test Laplace potential operators."""
    space = bempp.api.function_space(grid, "P", 1)

    rand = np.random.RandomState(0)
    vec = rand.rand(space.global_dof_count)

    grid_fun = bempp.api.GridFunction(space, coefficients=vec)

    points = np.vstack(
        [
            2 * np.ones(NPOINTS, dtype="float64"),
            rand.randn(NPOINTS),
            rand.randn(NPOINTS),
        ]
    )

    dense = operator(space, points, assembler="dense")
    fmm = operator(space, points, assembler="fmm")

    res_dense = dense.evaluate(grid_fun)
    res_fmm = fmm.evaluate(grid_fun)

    np.testing.assert_allclose(res_dense, res_fmm, rtol=TOL)


@pytest.mark.parametrize("operator", [
    bempp.api.operators.boundary.helmholtz.single_layer,
    bempp.api.operators.boundary.modified_helmholtz.double_layer])
def test_helmholtz_boundary_fmm(operator, grid):
    """Test Helmholtz boundary operators."""
    space = bempp.api.function_space(grid, "P", 1)

    rand = np.random.RandomState(0)
    vec = rand.rand(space.global_dof_count)

    wavenumber = 1.5

    dense = operator(space, space, space, wavenumber, assembler="dense").weak_form()
    fmm = operator(space, space, space, wavenumber, assembler="fmm").weak_form()

    np.testing.assert_allclose(dense @ vec, fmm @ vec, rtol=TOL)


@pytest.mark.parametrize("operator", [
    bempp.api.operators.potential.helmholtz.single_layer,
    bempp.api.operators.potential.modified_helmholtz.double_layer
])
def test_helmholtz_potential_fmm(operator, grid):
    """Test Helmholtz potential operators."""
    space = bempp.api.function_space(grid, "P", 1)

    rand = np.random.RandomState(0)
    vec = rand.rand(space.global_dof_count)

    wavenumber = 1.5

    grid_fun = bempp.api.GridFunction(space, coefficients=vec)

    points = np.vstack(
        [
            2 * np.ones(NPOINTS, dtype="float64"),
            rand.randn(NPOINTS),
            rand.randn(NPOINTS),
        ]
    )

    dense = operator(space, points, wavenumber, assembler="dense")
    fmm = operator(space, points, wavenumber, assembler="fmm")

    res_dense = dense.evaluate(grid_fun)
    res_fmm = fmm.evaluate(grid_fun)

    np.testing.assert_allclose(res_dense, res_fmm, rtol=TOL)


@pytest.mark.parametrize("operator", [
    bempp.api.operators.boundary.maxwell.electric_field,
    bempp.api.operators.boundary.maxwell.magnetic_field,
])
def test_maxwell_boundary_fmm(operator, grid):
    """Test Maxwell boundary operators."""
    rwg = bempp.api.function_space(grid, "RWG", 0)
    snc = bempp.api.function_space(grid, "SNC", 0)

    rand = np.random.RandomState(0)
    vec = rand.rand(rwg.global_dof_count)

    wavenumber = 1.5

    dense = operator(rwg, rwg, snc, wavenumber, assembler="dense").weak_form()
    fmm = operator(rwg, rwg, snc, wavenumber, assembler="fmm").weak_form()

    np.testing.assert_allclose(dense @ vec, fmm @ vec, rtol=TOL)


@pytest.mark.parametrize("operator", [
    bempp.api.operators.potential.maxwell.electric_field,
    bempp.api.operators.potential.maxwell.magnetic_field,
])
def test_maxwell_potential_fmm(operator, grid):
    """Test Maxwell potential operators."""
    rwg = bempp.api.function_space(grid, "RWG", 0)

    rand = np.random.RandomState(0)
    vec = rand.rand(rwg.global_dof_count)

    wavenumber = 1.5

    grid_fun = bempp.api.GridFunction(rwg, coefficients=vec)

    points = np.vstack(
        [
            2 * np.ones(NPOINTS, dtype="float64"),
            rand.randn(NPOINTS),
            rand.randn(NPOINTS),
        ]
    )

    dense = operator(rwg, points, wavenumber, assembler="dense")
    fmm = operator(rwg, points, wavenumber, assembler="fmm")

    res_dense = dense.evaluate(grid_fun)
    res_fmm = fmm.evaluate(grid_fun)

    np.testing.assert_allclose(res_dense, res_fmm, rtol=TOL)


@pytest.mark.parametrize("operator", [
    bempp.api.operators.boundary.laplace.single_layer,
    bempp.api.operators.boundary.laplace.hypersingular])
def test_fmm_two_grids_laplace(operator, grid1, grid2):
    """Test the FMM for Laplace between two different grids."""
    rand = np.random.RandomState(0)

    p1_space1 = bempp.api.function_space(grid1, "P", 1)
    p1_space2 = bempp.api.function_space(grid2, "P", 1)

    vec = rand.rand(p1_space1.global_dof_count)

    dense = operator(p1_space1, p1_space2, p1_space2,
                     assembler="dense").weak_form()
    fmm = operator(p1_space1, p1_space2, p1_space2,
                   assembler="fmm").weak_form()

    np.testing.assert_allclose(dense @ vec, fmm @ vec, rtol=TOL)
