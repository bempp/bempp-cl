"""Unit tests for the dense assembler."""

import pytest
import bempp.api
from bempp.api import function_space
from bempp.api.operators.boundary import (
    laplace, helmholtz, modified_helmholtz, maxwell, sparse)


@pytest.mark.parametrize("operator", [
    sparse.identity])
@pytest.mark.parametrize("orders", [(0, 0), (0, 1), (1, 1)])
def test_sparse_operators(operator, orders):
    """Test dense assembler for the Laplace operators."""
    grid = bempp.api.shapes.regular_sphere(0)
    space0 = function_space(grid, "DP", orders[0])
    space1 = function_space(grid, "DP", orders[1])

    op = operator(space0, space1, space1)
    op.weak_form()


@pytest.mark.parametrize("operator", [
    sparse.identity])
def test_maxwell_sparse_operators(operator):
    """Test dense assembler for the Laplace operators."""
    grid = bempp.api.shapes.regular_sphere(0)
    space0 = function_space(grid, "RWG", 0)
    space1 = function_space(grid, "SNC", 0)

    op = operator(space0, space0, space1)
    op.weak_form()


@pytest.mark.parametrize("operator", [
    laplace.single_layer, laplace.double_layer,
    laplace.adjoint_double_layer, laplace.hypersingular])
@pytest.mark.parametrize("orders", [(0, 0), (0, 1), (1, 1)])
def test_laplace_operators(operator, orders):
    """Test dense assembler for the Laplace operators."""
    grid = bempp.api.shapes.regular_sphere(0)
    space0 = function_space(grid, "DP", orders[0])
    space1 = function_space(grid, "DP", orders[1])

    op = operator(space0, space1, space1,
                  assembler="dense")
    op.weak_form()


@pytest.mark.parametrize("operator", [
    helmholtz.single_layer, helmholtz.double_layer,
    helmholtz.adjoint_double_layer, helmholtz.hypersingular])
@pytest.mark.parametrize("wavenumber", [2.5, 2.5 + 1j])
@pytest.mark.parametrize("orders", [(0, 0), (0, 1), (1, 1)])
def test_helmholtz_operators(operator, wavenumber, orders):
    """Test dense assembler for the Helmholtz operators."""
    grid = bempp.api.shapes.regular_sphere(0)
    space0 = function_space(grid, "DP", orders[0])
    space1 = function_space(grid, "DP", orders[1])

    op = operator(space0, space1, space1, wavenumber,
                  assembler="dense")
    op.weak_form()


@pytest.mark.parametrize("operator", [
    modified_helmholtz.single_layer, modified_helmholtz.double_layer,
    modified_helmholtz.adjoint_double_layer, modified_helmholtz.hypersingular])
@pytest.mark.parametrize("wavenumber", [2.5])
@pytest.mark.parametrize("orders", [(0, 0), (0, 1), (1, 1)])
def test_modified_helmholtz_operators(operator, wavenumber, orders):
    """Test dense assembler for the modified Helmholtz operators."""
    grid = bempp.api.shapes.regular_sphere(0)
    space0 = function_space(grid, "DP", orders[0])
    space1 = function_space(grid, "DP", orders[1])

    op = operator(space0, space1, space1, wavenumber,
                  assembler="dense")
    op.weak_form()


@pytest.mark.parametrize("operator", [
    maxwell.magnetic_field, maxwell.electric_field])
@pytest.mark.parametrize("wavenumber", [2.5, 2.5 + 1j])
@pytest.mark.parametrize("space0", ["RWG"])
@pytest.mark.parametrize("space1", ["SNC"])
def test_maxwell_operators(operator, wavenumber, space0, space1):
    """Test dense assembler for the Maxwell operators."""
    grid = bempp.api.shapes.regular_sphere(0)
    space0 = function_space(grid, space0, 0)
    space1 = function_space(grid, space1, 0)

    op = operator(space0, space0, space1, wavenumber,
                  assembler="dense")
    op.weak_form()
