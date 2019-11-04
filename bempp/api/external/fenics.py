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
        bm_coords.transpose(), bm_cells.transpose())
    return bempp_boundary_grid


def fenics_to_bempp_trace_data(fenics_space):
    """
    Returns tuple (space,trace_matrix)
    """
    family, degree = fenics_space_info(fenics_space)

    if family == "Lagrange":
        if degree == 1:
            import bempp.api.fenics_interface.p1_coupling as p1_coupling
            return p1_coupling.p1_trace(fenics_space)
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
        from bempp.api.assembly.discrete_boundary_operator import \
            SparseDiscreteBoundaryOperator
	from dolfin import as_backend_type, assemble, parameters
        from scipy.sparse import csr_matrix

        if self._sparse_mat is not None:
            return SparseDiscreteBoundaryOperator(self._sparse_mat)

        backend = parameters['linear_algebra_backend']
	if backend != 'PETSc':
            raise ValueError('Only the PETSc linear algebra backend is supported.')
        mat = as_backend_type(assemble(self._fenics_weak_form)).mat()
        (indptr, indices, data) = mat.getValuesCSR()
        self._sparse_mat = csr_matrix(
            (data, indices, indptr), shape=mat.size)

        return SparseDiscreteBoundaryOperator(self._sparse_mat)
