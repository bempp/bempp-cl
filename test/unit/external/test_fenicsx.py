"""Unit tests for the FEniCSx interface."""

import pytest
import numpy as np
from mpi4py import MPI
import bempp.api
from bempp.api.external.fenicsx import fenics_to_bempp_trace_data


def test_p1_trace(has_dolfinx):
    """Test the trace of a P1 Dolfin function."""
    if has_dolfinx:
        import dolfinx
    else:
        try:
            import dolfinx
        except ImportError:
            pytest.skip("DOLFINx must be installed to run this test")

    import dolfinx.geometry
    from dolfinx.generation import UnitCubeMesh
    from dolfinx.fem import FunctionSpace, Function

    fenics_mesh = UnitCubeMesh(MPI.COMM_WORLD, 2, 2, 2)
    fenics_space = FunctionSpace(fenics_mesh, ("CG", 1))

    bempp_space, trace_matrix = fenics_to_bempp_trace_data(fenics_space)

    fenics_coeffs = np.random.rand(fenics_space.dofmap.index_map.size_global)
    bempp_coeffs = trace_matrix @ fenics_coeffs

    fenics_fun = Function(fenics_space)
    fenics_fun.vector[:] = fenics_coeffs
    bempp_fun = bempp.api.GridFunction(bempp_space, coefficients=bempp_coeffs)

    tree = dolfinx.geometry.BoundingBoxTree(fenics_mesh, 3)

    midpoint_tree = dolfinx.geometry.create_midpoint_tree(
        fenics_mesh, 3, list(range(fenics_mesh.topology.connectivity(3, 0).num_nodes))
    )

    for cell in bempp_space.grid.entity_iterator(0):
        mid = cell.geometry.centroid
        bempp_val = bempp_fun.evaluate(cell.index, np.array([[1 / 3], [1 / 3]]))

        fenics_cell = dolfinx.geometry.compute_closest_entity(
            tree, midpoint_tree, fenics_mesh, mid
        )[0]
        fenics_val = fenics_fun.eval([mid.T], [fenics_cell])
        assert np.isclose(bempp_val[0, 0], fenics_val[0])
