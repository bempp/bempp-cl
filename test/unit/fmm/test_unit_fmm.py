"""Unit tests for the dense assembler."""

import numpy as np
import pytest
import bempp.api
from bempp.api import function_space, check_for_fmm
from bempp.api.operators.boundary import laplace, helmholtz


def test_laplace_single_layer(allow_external_skips):
    """Test dense assembler for the Laplace operators."""
    if allow_external_skips and not check_for_fmm():
        pytest.skip("ExaFMM must be installed to run this test.")

    grid = bempp.api.shapes.regular_sphere(2)
    space = function_space(grid, "DP", 0)

    op1 = laplace.single_layer(space, space, space, assembler="dense")
    op2 = laplace.single_layer(space, space, space, assembler="fmm")

    fun = bempp.api.GridFunction(
        space, coefficients=np.random.rand(space.global_dof_count)
    )

    assert np.allclose((op1 * fun).coefficients, (op2 * fun).coefficients)

    bempp.api.clear_fmm_cache()


@pytest.mark.parametrize("wavenumber", [2.5])  # , 2.5 + 1j])
def test_helmholtz_single_layer(allow_external_skips, wavenumber):
    """Test dense assembler for the Laplace operators."""
    if allow_external_skips and not check_for_fmm():
        pytest.skip("ExaFMM must be installed to run this test.")

    grid = bempp.api.shapes.regular_sphere(2)
    space = function_space(grid, "DP", 0)

    op1 = helmholtz.single_layer(space, space, space, wavenumber, assembler="dense")
    op2 = helmholtz.single_layer(space, space, space, wavenumber, assembler="fmm")

    fun = bempp.api.GridFunction(
        space, coefficients=np.random.rand(space.global_dof_count)
    )

    assert np.allclose((op1 * fun).coefficients, (op2 * fun).coefficients)

    bempp.api.clear_fmm_cache()
