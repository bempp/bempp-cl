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

def test_maxwell_potential_fmm():
    """Test Maxwell potential operators."""

    grid = bempp.api.shapes.ellipsoid(1, 0.5, 0.3)
    rwg = bempp.api.function_space(grid, "RWG", 0)

    rand = np.random.RandomState(0)
    vec = rand.rand(rwg.global_dof_count)

    wavenumber = 1.5

    grid_fun = bempp.api.GridFunction(rwg, coefficients=vec)  # noqa: F841

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
