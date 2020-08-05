"""Unit tests for the FMM assembler."""

# pylint: disable=redefined-outer-name
# pylint: disable=C0103

import pytest
import numpy as np
import bempp.api

pytestmark = pytest.mark.usefixtures("default_parameters", "helpers")

NPOINTS = 10
TOL = 1e-5

bempp.api.GLOBAL_PARAMETERS.fmm.expansion_order = 10
# bempp.api.GLOBAL_PARAMETERS.fmm.ncrit = 100


def check_close(op1, op2, vec):
    res1 = op1 @ vec
    res2 = op2 @ vec
    return np.allclose(res1, res2, rtol=TOL)


def test_laplace_boundary_fmm():
    """Test Laplace boundary operators."""

    grid = bempp.api.shapes.ellipsoid(1, 0.5, 0.3)
    space = bempp.api.function_space(grid, "P", 1)

    rand = np.random.RandomState(0)
    vec = rand.rand(space.global_dof_count)

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

    np.testing.assert_allclose(slp_dense @ vec, slp_fmm @ vec, rtol=TOL)
    np.testing.assert_allclose(dlp_dense @ vec, dlp_fmm @ vec, rtol=TOL)
    np.testing.assert_allclose(adjoint_dlp_dense @ vec, adjoint_dlp_fmm @ vec, rtol=TOL)
    np.testing.assert_allclose(hyp_dense @ vec, hyp_fmm @ vec, rtol=TOL)


def test_laplace_potential_fmm():
    """Test Laplace potential operators."""

    grid = bempp.api.shapes.ellipsoid(1, 0.5, 0.3)
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

    slp_pot_dense = bempp.api.operators.potential.laplace.single_layer(
        space, points, assembler=assembler
    )
    dlp_pot_dense = bempp.api.operators.potential.laplace.double_layer(
        space, points, assembler=assembler
    )

    assembler = "fmm"

    slp_pot_fmm = bempp.api.operators.potential.laplace.single_layer(
        space, points, assembler=assembler
    )
    dlp_pot_fmm = bempp.api.operators.potential.laplace.double_layer(
        space, points, assembler=assembler
    )

    slp_pot_res_dense = slp_pot_dense.evaluate(grid_fun)
    dlp_pot_res_dense = dlp_pot_dense.evaluate(grid_fun)

    slp_pot_res_fmm = slp_pot_fmm.evaluate(grid_fun)
    dlp_pot_res_fmm = dlp_pot_fmm.evaluate(grid_fun)

    np.testing.assert_allclose(slp_pot_res_dense, slp_pot_res_fmm, rtol=1e-4)
    np.testing.assert_allclose(dlp_pot_res_dense, dlp_pot_res_fmm, rtol=1e-4)


def test_modified_helmholtz_boundary_fmm():
    """Test modified Helmholtz boundary operators."""

    grid = bempp.api.shapes.ellipsoid(1, 0.5, 0.3)
    space = bempp.api.function_space(grid, "P", 1)

    omega = 1.5

    rand = np.random.RandomState(0)
    vec = rand.rand(space.global_dof_count)

    assembler = "dense"

    slp_dense = bempp.api.operators.boundary.modified_helmholtz.single_layer(
        space, space, space, omega, assembler=assembler,
    ).weak_form()
    dlp_dense = bempp.api.operators.boundary.modified_helmholtz.double_layer(
        space, space, space, omega, assembler=assembler,
    ).weak_form()
    adjoint_dlp_dense = bempp.api.operators.boundary.modified_helmholtz.adjoint_double_layer(
        space, space, space, omega, assembler=assembler,
    ).weak_form()
    hyp_dense = bempp.api.operators.boundary.modified_helmholtz.hypersingular(
        space, space, space, omega, assembler=assembler,
    ).weak_form()

    assembler = "fmm"

    slp_fmm = bempp.api.operators.boundary.modified_helmholtz.single_layer(
        space, space, space, omega, assembler=assembler,
    ).weak_form()
    dlp_fmm = bempp.api.operators.boundary.modified_helmholtz.double_layer(
        space, space, space, omega, assembler=assembler,
    ).weak_form()
    adjoint_dlp_fmm = bempp.api.operators.boundary.modified_helmholtz.adjoint_double_layer(
        space, space, space, omega, assembler=assembler,
    ).weak_form()
    hyp_fmm = bempp.api.operators.boundary.modified_helmholtz.hypersingular(
        space, space, space, omega, assembler=assembler,
    ).weak_form()

    np.testing.assert_allclose(slp_dense @ vec, slp_fmm @ vec, rtol=TOL)
    np.testing.assert_allclose(dlp_dense @ vec, dlp_fmm @ vec, rtol=TOL)
    np.testing.assert_allclose(adjoint_dlp_dense @ vec, adjoint_dlp_fmm @ vec, rtol=TOL)
    np.testing.assert_allclose(hyp_dense @ vec, hyp_fmm @ vec, rtol=TOL)


def test_modified_helmholtz_potential_fmm():
    """Test modified Helmholtz potential operators."""

    grid = bempp.api.shapes.ellipsoid(1, 0.5, 0.3)
    space = bempp.api.function_space(grid, "P", 1)

    omega = 1.5

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

    slp_pot_dense = bempp.api.operators.potential.modified_helmholtz.single_layer(
        space, points, omega, assembler=assembler
    )
    dlp_pot_dense = bempp.api.operators.potential.modified_helmholtz.double_layer(
        space, points, omega, assembler=assembler
    )

    assembler = "fmm"

    slp_pot_fmm = bempp.api.operators.potential.modified_helmholtz.single_layer(
        space, points, omega, assembler=assembler
    )
    dlp_pot_fmm = bempp.api.operators.potential.modified_helmholtz.double_layer(
        space, points, omega, assembler=assembler
    )

    slp_pot_res_dense = slp_pot_dense.evaluate(grid_fun)
    dlp_pot_res_dense = dlp_pot_dense.evaluate(grid_fun)

    slp_pot_res_fmm = slp_pot_fmm.evaluate(grid_fun)
    dlp_pot_res_fmm = dlp_pot_fmm.evaluate(grid_fun)

    np.testing.assert_allclose(slp_pot_res_dense, slp_pot_res_fmm, rtol=1e-4)
    np.testing.assert_allclose(dlp_pot_res_dense, dlp_pot_res_fmm, rtol=1e-4)


def test_helmholtz_boundary_fmm():
    """Test Helmholtz boundary operators."""

    grid = bempp.api.shapes.ellipsoid(1, 0.5, 0.3)
    space = bempp.api.function_space(grid, "P", 1)

    rand = np.random.RandomState(0)
    vec = rand.rand(space.global_dof_count)

    wavenumber = 1.5

    assembler = "dense"

    slp_dense = bempp.api.operators.boundary.helmholtz.single_layer(
        space, space, space, wavenumber, assembler=assembler,
    ).weak_form()
    dlp_dense = bempp.api.operators.boundary.helmholtz.double_layer(
        space, space, space, wavenumber, assembler=assembler,
    ).weak_form()
    adjoint_dlp_dense = bempp.api.operators.boundary.helmholtz.adjoint_double_layer(
        space, space, space, wavenumber, assembler=assembler,
    ).weak_form()
    hyp_dense = bempp.api.operators.boundary.helmholtz.hypersingular(
        space, space, space, wavenumber, assembler=assembler,
    ).weak_form()

    assembler = "fmm"

    slp_fmm = bempp.api.operators.boundary.helmholtz.single_layer(
        space, space, space, wavenumber, assembler=assembler,
    ).weak_form()
    dlp_fmm = bempp.api.operators.boundary.helmholtz.double_layer(
        space, space, space, wavenumber, assembler=assembler,
    ).weak_form()
    adjoint_dlp_fmm = bempp.api.operators.boundary.helmholtz.adjoint_double_layer(
        space, space, space, wavenumber, assembler=assembler,
    ).weak_form()
    hyp_fmm = bempp.api.operators.boundary.helmholtz.hypersingular(
        space, space, space, wavenumber, assembler=assembler,
    ).weak_form()

    np.testing.assert_allclose(slp_dense @ vec, slp_fmm @ vec, rtol=TOL)
    np.testing.assert_allclose(dlp_dense @ vec, dlp_fmm @ vec, rtol=TOL)
    np.testing.assert_allclose(adjoint_dlp_dense @ vec, adjoint_dlp_fmm @ vec, rtol=TOL)
    np.testing.assert_allclose(hyp_dense @ vec, hyp_fmm @ vec, rtol=TOL)


def test_helmholtz_potential_fmm():
    """Test Helmholtz potential operators."""

    grid = bempp.api.shapes.ellipsoid(1, 0.5, 0.3)
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

    slp_pot_dense = bempp.api.operators.potential.helmholtz.single_layer(
        space, points, wavenumber, assembler=assembler
    )
    dlp_pot_dense = bempp.api.operators.potential.helmholtz.double_layer(
        space, points, wavenumber, assembler=assembler
    )

    assembler = "fmm"

    slp_pot_fmm = bempp.api.operators.potential.helmholtz.single_layer(
        space, points, wavenumber, assembler=assembler
    )
    dlp_pot_fmm = bempp.api.operators.potential.helmholtz.double_layer(
        space, points, wavenumber, assembler=assembler
    )

    slp_pot_res_dense = slp_pot_dense.evaluate(grid_fun)
    dlp_pot_res_dense = dlp_pot_dense.evaluate(grid_fun)

    slp_pot_res_fmm = slp_pot_fmm.evaluate(grid_fun)
    dlp_pot_res_fmm = dlp_pot_fmm.evaluate(grid_fun)

    np.testing.assert_allclose(slp_pot_res_dense, slp_pot_res_fmm, rtol=1e-4)
    np.testing.assert_allclose(dlp_pot_res_dense, dlp_pot_res_fmm, rtol=1e-4)


def test_maxwell_boundary_fmm():
    """Test Maxwell boundary operators."""

    TOL = 5e-5

    grid = bempp.api.shapes.ellipsoid(1, 0.5, 0.3)
    rwg = bempp.api.function_space(grid, "RWG", 0)
    snc = bempp.api.function_space(grid, "SNC", 0)

    rand = np.random.RandomState(0)
    vec = rand.rand(rwg.global_dof_count)

    wavenumber = 1.5

    assembler = "dense"

    efield_dense = bempp.api.operators.boundary.maxwell.electric_field(
        rwg, rwg, snc, wavenumber, assembler=assembler,
    ).weak_form()
    mfield_dense = bempp.api.operators.boundary.maxwell.magnetic_field(
        rwg, rwg, snc, wavenumber, assembler=assembler,
    ).weak_form()

    assembler = "fmm"

    efield_fmm = bempp.api.operators.boundary.maxwell.electric_field(
        rwg, rwg, snc, wavenumber, assembler=assembler,
    ).weak_form()
    mfield_fmm = bempp.api.operators.boundary.maxwell.magnetic_field(
        rwg, rwg, snc, wavenumber, assembler=assembler,
    ).weak_form()

    np.testing.assert_allclose(efield_dense @ vec, efield_fmm @ vec, rtol=TOL)
    np.testing.assert_allclose(mfield_dense @ vec, mfield_fmm @ vec, rtol=TOL)


def test_maxwell_potential_fmm():
    """Test Maxwell potential operators."""

    grid = bempp.api.shapes.ellipsoid(1, 0.5, 0.3)
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

    assembler = "dense"

    efield_pot_dense = bempp.api.operators.potential.maxwell.electric_field(
        rwg, points, wavenumber, assembler=assembler
    )
    mfield_pot_dense = bempp.api.operators.potential.maxwell.magnetic_field(
        rwg, points, wavenumber, assembler=assembler
    )

    assembler = "fmm"

    efield_pot_fmm = bempp.api.operators.potential.maxwell.electric_field(
        rwg, points, wavenumber, assembler=assembler
    )
    mfield_pot_fmm = bempp.api.operators.potential.maxwell.magnetic_field(
        rwg, points, wavenumber, assembler=assembler
    )

    efield_pot_res_dense = efield_pot_dense.evaluate(grid_fun)
    mfield_pot_res_dense = mfield_pot_dense.evaluate(grid_fun)

    efield_pot_res_fmm = efield_pot_fmm.evaluate(grid_fun)
    mfield_pot_res_fmm = mfield_pot_fmm.evaluate(grid_fun)

    np.testing.assert_allclose(efield_pot_res_dense, efield_pot_res_fmm, rtol=1e-4)
    np.testing.assert_allclose(mfield_pot_res_dense, mfield_pot_res_fmm, rtol=1e-4)


def test_fmm_two_grids_laplace():
    """Test the FMM for Laplace between two different grids."""
    import bempp.api

    TOL = 2e-3

    rand = np.random.RandomState(0)

    wavenumber = 1.5

    grid1 = bempp.api.shapes.ellipsoid(0.5, 0.5, 0.3)
    grid2 = bempp.api.shapes.sphere(r=1.5)

    p1_space1 = bempp.api.function_space(grid1, "P", 1)
    p1_space2 = bempp.api.function_space(grid2, "P", 1)

    vec = rand.rand(p1_space1.global_dof_count)

    assembler = "dense"

    laplace_slp_dense = bempp.api.operators.boundary.laplace.single_layer(
        p1_space1, p1_space2, p1_space2, assembler=assembler
    ).weak_form()
    laplace_hyp_dense = bempp.api.operators.boundary.laplace.hypersingular(
        p1_space1, p1_space2, p1_space2, assembler=assembler
    ).weak_form()

    assembler = "fmm"

    laplace_slp_fmm = bempp.api.operators.boundary.laplace.single_layer(
        p1_space1, p1_space2, p1_space2, assembler=assembler
    ).weak_form()
    laplace_hyp_fmm = bempp.api.operators.boundary.laplace.hypersingular(
        p1_space1, p1_space2, p1_space2, wavenumber, assembler=assembler
    ).weak_form()

    np.testing.assert_allclose(
        laplace_slp_dense @ vec, laplace_slp_fmm @ vec, rtol=TOL
    )
    np.testing.assert_allclose(
        laplace_hyp_dense @ vec, laplace_hyp_fmm @ vec, rtol=TOL
    )


def test_fmm_two_grids_helmholtz():
    """Test the FMM for Helmholtz between two different grids."""
    import bempp.api

    TOL = 2e-3

    rand = np.random.RandomState(0)

    wavenumber = 1.5

    grid1 = bempp.api.shapes.ellipsoid(0.5, 0.5, 0.3)
    grid2 = bempp.api.shapes.sphere(r=1.5)

    p1_space1 = bempp.api.function_space(grid1, "P", 1)
    p1_space2 = bempp.api.function_space(grid2, "P", 1)

    vec = rand.rand(p1_space1.global_dof_count)

    assembler = "dense"

    mod_helmholtz_hyp_dense = bempp.api.operators.boundary.modified_helmholtz.hypersingular(
        p1_space1, p1_space2, p1_space2, wavenumber, assembler=assembler
    ).weak_form()
    helmholtz_hyp_dense = bempp.api.operators.boundary.helmholtz.hypersingular(
        p1_space1, p1_space2, p1_space2, wavenumber, assembler=assembler
    ).weak_form()

    assembler = "fmm"

    mod_helmholtz_hyp_fmm = bempp.api.operators.boundary.modified_helmholtz.hypersingular(
        p1_space1, p1_space2, p1_space2, wavenumber, assembler=assembler
    ).weak_form()
    helmholtz_hyp_fmm = bempp.api.operators.boundary.helmholtz.hypersingular(
        p1_space1, p1_space2, p1_space2, wavenumber, assembler=assembler
    ).weak_form()

    np.testing.assert_allclose(
        helmholtz_hyp_dense @ vec, helmholtz_hyp_fmm @ vec, rtol=TOL
    )
    np.testing.assert_allclose(
        mod_helmholtz_hyp_dense @ vec, mod_helmholtz_hyp_fmm @ vec, rtol=TOL
    )


def test_fmm_two_grids_maxwell():
    """Test the FMM for Maxwell between two different grids."""
    import bempp.api

    TOL = 2e-3

    rand = np.random.RandomState(0)

    wavenumber = 1.5

    grid1 = bempp.api.shapes.ellipsoid(0.5, 0.5, 0.3)
    grid2 = bempp.api.shapes.sphere(r=1.5)

    rwg1 = bempp.api.function_space(grid1, "RWG", 0)
    rwg2 = bempp.api.function_space(grid2, "RWG", 0)
    snc2 = bempp.api.function_space(grid2, "SNC", 0)

    vec = rand.rand(rwg1.global_dof_count)

    assembler = "dense"

    maxwell_efield_dense = bempp.api.operators.boundary.maxwell.electric_field(
        rwg1, rwg2, snc2, wavenumber, assembler=assembler
    ).weak_form()
    maxwell_mfield_dense = bempp.api.operators.boundary.maxwell.magnetic_field(
        rwg1, rwg2, snc2, wavenumber, assembler=assembler
    ).weak_form()

    assembler = "fmm"

    maxwell_efield_fmm = bempp.api.operators.boundary.maxwell.electric_field(
        rwg1, rwg2, snc2, wavenumber, assembler=assembler
    ).weak_form()
    maxwell_mfield_fmm = bempp.api.operators.boundary.maxwell.magnetic_field(
        rwg1, rwg2, snc2, wavenumber, assembler=assembler
    ).weak_form()

    np.testing.assert_allclose(
        maxwell_efield_dense @ vec, maxwell_efield_fmm @ vec, rtol=TOL
    )
    np.testing.assert_allclose(
        maxwell_mfield_dense @ vec, maxwell_mfield_fmm @ vec, rtol=TOL
    )
