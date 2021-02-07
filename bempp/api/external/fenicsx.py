"""Interface to DOLFINX for FEM-BEM coupling."""


def boundary_grid_from_fenics_mesh(fenics_mesh):
    """
    Create a Bempp boundary grid from a FEniCS Mesh.

    Return the Bempp grid and a map from the node numberings of the FEniCS
    mesh to the node numbers of the boundary grid.
    """
    import bempp.api
    import numpy as np
    from dolfinx.cpp.mesh import entities_to_geometry, exterior_facet_indices

    boundary = entities_to_geometry(
        fenics_mesh,
        fenics_mesh.topology.dim - 1,
        exterior_facet_indices(fenics_mesh),
        True,
    )

    bm_nodes = set()
    for tri in boundary:
        for node in tri:
            bm_nodes.add(node)
    bm_nodes = list(bm_nodes)
    bm_cells = np.array([[bm_nodes.index(i) for i in tri] for tri in boundary])
    bm_coords = fenics_mesh.geometry.x[bm_nodes]

    bempp_boundary_grid = bempp.api.Grid(bm_coords.transpose(), bm_cells.transpose())

    return bempp_boundary_grid, bm_nodes


def fenics_to_bempp_trace_data(fenics_space):
    """Return tuple (space,trace_matrix)."""
    family, degree = fenics_space_info(fenics_space)

    if family == "Lagrange":
        if degree == 1:
            return p1_trace(fenics_space)
    else:
        raise NotImplementedError()


def fenics_space_info(fenics_space):
    """Return tuple (family,degree) containing information about a FEniCS space."""
    element = fenics_space.ufl_element()
    family = element.family()
    degree = element.degree()
    return (family, degree)


# pylint: disable=too-many-locals
def p1_trace(fenics_space):
    """
    Return the P1 trace operator.

    This function returns a pair (space, trace_matrix),
    where space is a Bempp space object and trace_matrix is the corresponding
    matrix that maps the coefficients of a FEniCS function to its boundary
    trace coefficients in the corresponding Bempp space.
    """
    import bempp.api
    from scipy.sparse import coo_matrix
    import numpy as np

    family, degree = fenics_space_info(fenics_space)
    if not (family == "Lagrange" and degree == 1):
        raise ValueError("fenics_space must be a p1 Lagrange space")

    fenics_mesh = fenics_space.mesh
    bempp_boundary_grid, bm_nodes = boundary_grid_from_fenics_mesh(fenics_mesh)

    # First get trace space
    space = bempp.api.function_space(bempp_boundary_grid, "P", 1)

    num_fenics_vertices = fenics_mesh.topology.connectivity(0, 0).num_nodes

    # FEniCS vertices to bempp dofs
    b_vertices_from_vertices = coo_matrix(
        (np.ones(len(bm_nodes)), (np.arange(len(bm_nodes)), bm_nodes)),
        shape=(len(bm_nodes), num_fenics_vertices),
        dtype="float64",
    ).tocsc()

    # Finally FEniCS dofs to vertices.
    dof_to_vertex_map = np.zeros(num_fenics_vertices, dtype=np.int64)
    tets = fenics_mesh.geometry.dofmap
    for tet in range(tets.num_nodes):
        cell_dofs = fenics_space.dofmap.cell_dofs(tet)
        cell_verts = tets.links(tet)
        for v in range(4):
            vertex_n = cell_verts[v]
            dof = cell_dofs[fenics_space.dofmap.dof_layout.entity_dofs(0, v)[0]]
            dof_to_vertex_map[dof] = vertex_n

    vertices_from_fenics_dofs = coo_matrix(
        (
            np.ones(num_fenics_vertices),
            (dof_to_vertex_map, np.arange(num_fenics_vertices)),
        ),
        shape=(num_fenics_vertices, num_fenics_vertices),
        dtype="float64",
    ).tocsc()

    # Get trace matrix by multiplication
    trace_matrix = b_vertices_from_vertices @ vertices_from_fenics_dofs

    # Now return everything
    return space, trace_matrix


class FenicsOperator(object):
    """Wrap a FEniCS-X Operator into a Bempp operator."""

    def __init__(self, fenics_weak_form):
        """Construct an operator from a weak form in FEniCS."""
        self._fenics_weak_form = fenics_weak_form
        self._sparse_mat = None

    def weak_form(self):
        """Return the weak form."""
        from bempp.api.assembly.discrete_boundary_operator import (
            SparseDiscreteBoundaryOperator,
        )
        from dolfinx.fem import assemble_matrix
        from scipy.sparse import csr_matrix

        if self._sparse_mat is None:
            mat = assemble_matrix(self._fenics_weak_form)
            mat.assemble()
            (indptr, indices, data) = mat.getValuesCSR()
            self._sparse_mat = csr_matrix((data, indices, indptr), shape=mat.size)

        return SparseDiscreteBoundaryOperator(self._sparse_mat)
