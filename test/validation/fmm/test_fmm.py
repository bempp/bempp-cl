"""Unit tests for the FMM assembler."""

# pylint: disable=redefined-outer-name
# pylint: disable=C0103

import pytest
import numpy as np
import bempp.api

pytestmark = pytest.mark.usefixtures("default_parameters", "helpers")

GRID_SIZE = 4
NPOINTS = 100
TOL = 1e-5


bempp.api.GLOBAL_PARAMETERS.fmm.expansion_order = 10
# bempp.api.GLOBAL_PARAMETERS.fmm.ncrit = 100


def check_close(op1, op2, vec):
    res1 = op1 @ vec
    res2 = op2 @ vec
    return np.allclose(res1, res2, rtol=TOL)


def test_laplace():
    """Test Laplace operators."""

    grid = bempp.api.shapes.ellipsoid(1, .5, .3)
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

    assembler = "dense"

    slp_dense = bempp.api.operators.boundary.laplace.single_layer(
        space, space, space, assembler=assembler,
    ).weak_form()
    dlp_dense = bempp.api.operators.boundary.laplace.double_layer(
        space, space, space, assembler=assembler,
    ).weak_form()
    adjoint_dlp_dense = bempp.api.operators.boundary.laplace.adjoint_double_layer(
        space, space, space, assembler=assembler,
    ).weak_form()
    hyp_dense = bempp.api.operators.boundary.laplace.hypersingular(
        space, space, space, assembler=assembler,
    ).weak_form()
    slp_pot_dense = bempp.api.operators.potential.laplace.single_layer(
        space, points, assembler=assembler)
    dlp_pot_dense = bempp.api.operators.potential.laplace.double_layer(
        space, points, assembler=assembler)

    assembler = "fmm"

    slp_fmm = bempp.api.operators.boundary.laplace.single_layer(
        space, space, space, assembler=assembler,
    ).weak_form()
    dlp_fmm = bempp.api.operators.boundary.laplace.double_layer(
        space, space, space, assembler=assembler,
    ).weak_form()
    adjoint_dlp_fmm = bempp.api.operators.boundary.laplace.adjoint_double_layer(
        space, space, space, assembler=assembler,
    ).weak_form()
    hyp_fmm = bempp.api.operators.boundary.laplace.hypersingular(
        space, space, space, assembler=assembler,
    ).weak_form()

    slp_pot_fmm = bempp.api.operators.potential.laplace.single_layer(
        space, points, assembler=assembler)
    dlp_pot_fmm = bempp.api.operators.potential.laplace.double_layer(
        space, points, assembler=assembler)

    slp_pot_res_dense = slp_pot_dense.evaluate(grid_fun)
    dlp_pot_res_dense = dlp_pot_dense.evaluate(grid_fun)

    slp_pot_res_fmm = slp_pot_fmm.evaluate(grid_fun)
    dlp_pot_res_fmm = dlp_pot_fmm.evaluate(grid_fun)

    np.testing.assert_allclose(slp_dense @ vec, slp_fmm @ vec, rtol=TOL)
    np.testing.assert_allclose(dlp_dense @ vec, dlp_fmm @ vec, rtol=TOL)
    np.testing.assert_allclose(adjoint_dlp_dense @ vec, adjoint_dlp_fmm @ vec, rtol=TOL)
    np.testing.assert_allclose(hyp_dense @ vec, hyp_fmm @ vec, rtol=TOL)

    np.testing.assert_allclose(slp_pot_res_dense, slp_pot_res_fmm, rtol=1E-4)
    np.testing.assert_allclose(dlp_pot_res_dense, dlp_pot_res_fmm, rtol=1E-4)


def xtest_helmholtz():
    """Test Helmholtz operators."""

    grid = bempp.api.shapes.ellipsoid(1, .5, .3)
    space = bempp.api.function_space(grid, "P", 1)

    rand = np.random.RandomState(0)
    vec = rand.rand(space.global_dof_count)

    wavenumber = 1.5

    grid_fun = bempp.api.GridFunction(space, coefficients=vec)  # noqa: F841

    points = np.vstack(
        [
            2 * np.ones(NPOINTS, dtype="float64"),
            rand.randn(NPOINTS),
            rand.randn(NPOINTS),
        ]
    )

    assembler = "dense"

    test_vec = np.zeros(space.global_dof_count, dtype='float64')
    test_vec[0] = 1

    slp_dense = bempp.api.operators.boundary.helmholtz.single_layer(
        space, space, space, wavenumber, assembler=assembler,
    ).weak_form()
    # dlp_dense = bempp.api.operators.boundary.helmholtz.double_layer(
        # space, space, space, wavenumber, assembler=assembler,
    # ).weak_form()
    # adjoint_dlp_dense = bempp.api.operators.boundary.helmholtz.adjoint_double_layer(
        # space, space, space, wavenumber, assembler=assembler,
    # ).weak_form()
    # hyp_dense = bempp.api.operators.boundary.helmholtz.hypersingular(
        # space, space, space, wavenumber, assembler=assembler,
    # ).weak_form()
    # slp_pot_dense =
    bempp.api.operators.potential.helmholtz.single_layer(
        space, points, wavenumber, assembler=assembler)
    # dlp_pot_dense = bempp.api.operators.potential.helmholtz.double_layer(
            # space, points, wavenumber, assembler=assembler)

    assembler = "fmm"

    slp_fmm = bempp.api.operators.boundary.helmholtz.single_layer(
        space, space, space, wavenumber, assembler=assembler,
    ).weak_form()
    # dlp_fmm = bempp.api.operators.boundary.helmholtz.double_layer(
        # space, space, space, wavenumber, assembler=assembler,
    # ).weak_form()
    # adjoint_dlp_fmm = bempp.api.operators.boundary.helmholtz.adjoint_double_layer(
        # space, space, space, wavenumber, assembler=assembler,
    # ).weak_form()
    # hyp_fmm = bempp.api.operators.boundary.helmholtz.hypersingular(
        # space, space, space, wavenumber, assembler=assembler,
    # ).weak_form()

    # slp_pot_fmm =
    bempp.api.operators.potential.helmholtz.single_layer(
        space, points, wavenumber, assembler=assembler)
    # dlp_pot_fmm = bempp.api.operators.potential.helmholtz.double_layer(
            # space, points, wavenumber, assembler=assembler)

    # slp_pot_res_dense = slp_pot_dense.evaluate(grid_fun)
    # dlp_pot_res_dense = dlp_pot_dense.evaluate(grid_fun)

    # slp_pot_res_fmm = slp_pot_fmm.evaluate(grid_fun)
    # dlp_pot_res_fmm = dlp_pot_fmm.evaluate(grid_fun)

    np.testing.assert_allclose(slp_dense @ vec, slp_fmm @ vec, rtol=TOL)
    # np.testing.assert_allclose(dlp_dense @ vec, dlp_fmm @ vec, rtol=TOL)
    # np.testing.assert_allclose(adjoint_dlp_dense @ vec, adjoint_dlp_fmm @ vec, rtol=TOL)
    # np.testing.assert_allclose(hyp_dense @ vec, hyp_fmm @ vec, rtol=TOL)

    # np.testing.assert_allclose(slp_pot_res_dense, slp_pot_res_fmm, rtol=1E-4)
    # np.testing.assert_allclose(dlp_pot_res_dense, dlp_pot_res_fmm, rtol=1E-4)
