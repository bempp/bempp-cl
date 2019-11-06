import dolfin as _dolfin


def boundary_grid_from_fenics_mesh(fenics_mesh):
    """
    Return a boundary grid from a FEniCS Mesh.
    """
    from bempp.api import grid_from_element_data

    boundary_mesh = _dolfin.BoundaryMesh(fenics_mesh, "exterior", False)
    bm_coords = boundary_mesh.coordinates()
    bm_cells = boundary_mesh.cells()
    bempp_boundary_grid = grid_from_element_data(
        bm_coords.transpose(), bm_cells.transpose()
    )
    return bempp_boundary_grid


def fenics_to_bempp_trace_data(fenics_space):
    """
    Returns tuple (space,trace_matrix)
    """
    family, degree = fenics_space_info(fenics_space)

    if family == "Lagrange":
        if degree == 1:
            return p1_trace(fenics_space)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()


def fenics_space_info(fenics_space):
    """
    Returns tuple (family,degree) containing information about a FEniCS space
    """
    element = fenics_space.ufl_element()
    family = element.family()
    degree = element.degree()
    return (family, degree)


class FenicsOperator(object):
    """Wraps a FEniCS Operator into a BEM++ operator."""

    def __init__(self, fenics_weak_form):
        """Construct an operator from a weak form in FEniCS."""
        self._fenics_weak_form = fenics_weak_form
        self._sparse_mat = None

    def weak_form(self):
        """Return the weak form."""
        from bempp.api.assembly.discrete_boundary_operator import (
            SparseDiscreteBoundaryOperator,
        )
        from dolfin import as_backend_type, assemble, parameters
        from scipy.sparse import csr_matrix

        if self._sparse_mat is not None:
            return SparseDiscreteBoundaryOperator(self._sparse_mat)

        backend = parameters["linear_algebra_backend"]
        if backend != "PETSc":
            raise ValueError("Only the PETSc linear algebra backend is supported.")
        mat = as_backend_type(assemble(self._fenics_weak_form)).mat()
        (indptr, indices, data) = mat.getValuesCSR()
        self._sparse_mat = csr_matrix((data, indices, indptr), shape=mat.size)

        return SparseDiscreteBoundaryOperator(self._sparse_mat)


# pylint: disable=too-many-locals
def p1_trace(fenics_space):
    """
    Return the P1 trace operator.

    This function returns a pair (space, trace_matrix),
    where space is a Bempp space object and trace_matrix is the corresponding
    matrix that maps the coefficients of a FEniCS function to its boundary
    trace coefficients in the corresponding Bempp space.

    """

    import dolfin
    import bempp.api
    from scipy.sparse import coo_matrix
    import numpy as np

    family, degree = fenics_space_info(fenics_space)
    if not (family == "Lagrange" and degree == 1):
        raise ValueError("fenics_space must be a p1 Lagrange space")

    mesh = fenics_space.mesh()

    boundary_mesh = dolfin.BoundaryMesh(mesh, "exterior", False)
    bm_nodes = boundary_mesh.entity_map(0).array().astype(np.int64)
    bm_coords = boundary_mesh.coordinates()
    bm_cells = boundary_mesh.cells()
    bempp_boundary_grid = bempp.api.Grid(bm_coords.transpose(), bm_cells.transpose())

    # First get trace space
    space = bempp.api.function_space(bempp_boundary_grid, "P", 1)

    # Now FEniCS vertices to boundary dofs
    b_vertices_from_vertices = coo_matrix(
        (np.ones(len(bm_nodes)), (np.arange(len(bm_nodes)), bm_nodes)),
        shape=(len(bm_nodes), mesh.num_vertices()),
        dtype="float64",
    ).tocsc()

    # Finally FEniCS dofs to vertices.
    vertices_from_fenics_dofs = coo_matrix(
        (
            np.ones(mesh.num_vertices()),
            (dolfin.dof_to_vertex_map(fenics_space), np.arange(mesh.num_vertices())),
        ),
        shape=(mesh.num_vertices(), mesh.num_vertices()),
        dtype="float64",
    ).tocsc()

    # Get trace matrix by multiplication
    trace_matrix = b_vertices_from_vertices @ vertices_from_fenics_dofs

    # Now return everything
    return space, trace_matrix
