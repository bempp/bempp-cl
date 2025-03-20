"""Interface to DOLFINX for FEM-BEM coupling."""


def boundary_grid_from_fenics_mesh(fenics_mesh):
    """
    Create a Bempp boundary grid from a FEniCS Mesh.

    Return the Bempp grid and a map from the node numberings of the FEniCS
    mesh to the node numbers of the boundary grid.
    """
    import bempp_cl.api
    import numpy as np
    from dolfinx.mesh import entities_to_geometry, exterior_facet_indices

    fenics_mesh.topology.create_entities(2)
    fenics_mesh.topology.create_connectivity(2, 3)
    fenics_mesh.topology.create_entity_permutations()

    facets = exterior_facet_indices(fenics_mesh.topology)
    boundary = entities_to_geometry(
        fenics_mesh,
        fenics_mesh.topology.dim - 1,
        facets,
        True,
    )

    c23 = fenics_mesh.topology.connectivity(2, 3)

    tet_indices = np.array([c23.links(facet)[0] for facet in facets])
    tet_vertices = entities_to_geometry(
        fenics_mesh,
        fenics_mesh.topology.dim,
        tet_indices,
        True,
    )

    bm_nodes = list(set(node for tri in boundary for node in tri))
    bm_cells = np.zeros([3, len(boundary)])
    for i, (tri, tet) in enumerate(zip(boundary, tet_vertices)):
        for j in range(3):
            bm_cells[j, i] = bm_nodes.index(tri[j])
        v0 = fenics_mesh.geometry.x[tri[0]]
        v1 = fenics_mesh.geometry.x[tri[1]]
        v2 = fenics_mesh.geometry.x[tri[2]]
        v3 = fenics_mesh.geometry.x[[i for i in tet if i not in tri][0]]
        to_other_vertex = v3 - v0
        normal = np.cross(v1 - v0, v2 - v0)
        if np.dot(normal, to_other_vertex) > 0:
            bm_cells[1, i], bm_cells[2, i] = bm_cells[2, i], bm_cells[1, i]

    bm_coords = fenics_mesh.geometry.x[bm_nodes].transpose()

    bempp_boundary_grid = bempp_cl.api.Grid(bm_coords, bm_cells)

    return bempp_boundary_grid, bm_nodes


def fenics_to_bempp_trace_data(fenics_space):
    """Return tuple (space,trace_matrix)."""
    family, degree = fenics_space_info(fenics_space)

    if family in ["Lagrange", "P"]:
        if degree == 1:
            return p1_trace(fenics_space)
    else:
        raise NotImplementedError()


def fenics_space_info(fenics_space):
    """Return tuple (family,degree) containing information about a FEniCS space."""
    element = fenics_space.ufl_element()
    family = element.basix_element.family.name
    degree = element.degree
    return family, degree


# pylint: disable=too-many-locals
def p1_trace(fenics_space):
    """
    Return the P1 trace operator.

    This function returns a pair (space, trace_matrix),
    where space is a Bempp space object and trace_matrix is the corresponding
    matrix that maps the coefficients of a FEniCS function to its boundary
    trace coefficients in the corresponding Bempp space.
    """
    import bempp_cl.api
    from scipy.sparse import coo_matrix
    import numpy as np

    family, degree = fenics_space_info(fenics_space)
    if not (family in ["Lagrange", "P"] and degree == 1):
        raise ValueError("fenics_space must be a p1 Lagrange space")

    fenics_mesh = fenics_space.mesh
    bempp_boundary_grid, bm_nodes = boundary_grid_from_fenics_mesh(fenics_mesh)

    # First get trace space
    space = bempp_cl.api.function_space(bempp_boundary_grid, "P", 1)

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

    # Support older versions of FEniCSx
    if hasattr(tets, "get_links"):
        ntets = tets.num_nodes
        tets = [tets.get_links(i) for i in range(ntets)]
    if hasattr(tets, "links"):
        ntets = tets.num_nodes
        tets = [tets.links(i) for i in range(ntets)]

    for tet, cell_verts in enumerate(tets):
        cell_dofs = fenics_space.dofmap.cell_dofs(tet)
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
        from bempp_cl.api.assembly.discrete_boundary_operator import (
            SparseDiscreteBoundaryOperator,
        )
        from dolfinx.fem import assemble_matrix, form
        from scipy.sparse import csr_matrix

        if self._sparse_mat is None:
            mat = assemble_matrix(form(self._fenics_weak_form))
            try:
                mat.scatter_reverse()
            except AttributeError:
                # Support for older FEniCSx
                mat.finalize()
            shape = tuple(
                i._ufl_function_space.dofmap.index_map.size_global for i in self._fenics_weak_form.arguments()
            )
            self._sparse_mat = csr_matrix((mat.data, mat.indices, mat.indptr), shape=shape)

        return SparseDiscreteBoundaryOperator(self._sparse_mat)
