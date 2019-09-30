"""Common FMM routines."""
import abc as _abc
import numpy as _np


class Node(object):
    """Definition of an FMM node."""

    def __init__(self, identifier, center, radius, vertex_ids):
        """Initialize a node."""

        self._identifier = identifier
        self._center = center
        self._radius = radius
        self._vertex_ids = vertex_ids

    @property
    def center(self):
        """Return center."""
        return center

    @property
    def radius(self):
        """Return radius."""
        return radius

    @property
    def identifier(self):
        """Return Identifier."""
        return self._identifier

    @property
    def vertex_ids(self):
        """A list of vertex ids associated with the node."""


class FmmInterface(_abc.ABC):
    """Interface to an FMM Instance."""

    def setup(
        domain
        dual_to_range
        regular_order,
        singular_order,
        *args,
        **kwargs
    ):
        """
        Setup the sources.

        Parameters
        ----------
        domain_space : space
            A Bempp space object that describes the domain space.
            Only scalar spaces are allowed.
        dual_to_range_space : space
            A scalar dual space.
        regular_order : int
            The integration order for regular elements.
        singular_order : int
            The integratoin order for singular elements.
        
        """

    @property
    def leaf_nodes(self):
        """
        Return a list of non-empty leaf nodes.
        """

    @property
    def create_evaluator(self):
        """
        Return a Scipy Linear Operator that evaluates the FMM.

        The returned class should subclass the Scipy LinearOperator class
        so that it provides a matvec routine that accept a vector of coefficients
        and returns the result of a matrix vector product.
        """

    def _map_space_to_points(space, nodes, local_points):
        """Return mapper from grid coeffs to point evaluations."""
        from scipy.sparse import coo_matrix

        local_space = space.localised_space
        number_of_local_points = local_points.shape[1]
        nshape_funs = space.number_of_shape_functions
        number_of_vertices = number_of_local_points * space.number_of_support_elements

        global_dofs = []
        node_dofs = []
        values = []

        for node in nodes:
            associated_elements = set(
                [vertex // number_of_local_points for vertex in node.vertex_ids]
            )
            # Evaluate basis on the elements
            basis_values = {}
            for elem in associated_elements:
                # Spaces are scalar, so can use 2nd and 2rd component of eval
                basis_values[elem] = space.evaluate(elem, local_points)[0, :, :]

            # Now fill up the matrix elements.
            for vertex in node.vertex_ids:
                elem = vertex // number_of_local_points
                local_point_index = vertex & number_of_local_points
                global_dofs.extent(space.local2global[:, elem])
                node_dofs.extent(nshape_funs * [vertex])
                values.extent(basis_values[elem][:, local_point_index])

            transform = coo_matrix(
                (values, (node_dofs, global_dofs)),
                shape=(number_of_vertices, local_space.global_dof_count),
            )

            return transform @ space.map_to_localised_space


def grid_to_points(grid, support_elements, local_points):
    """
    Map a grid to an array of points.

    Returns a (N, 3) point array that stores the global vertices
    associated with the local points in each triangle.
    Points are stored in consecutive order for each element 
    in the support_elements list. Hence, the returned array is of the form
    [ v_1^1, v_2^1, ..., v_M^1, v_1^2, v_2^2, ...], where
    v_i^j is the ith point in the jth element in
    the support_elements list.

    Parameters
    ----------
    grid : Grid
        A Bempp Grid object.
    support_elements : List of Integers
        A list of integers storing the indices of
        the support elements in the grid.
    local_points : np.ndarray
        (2, M) array of local coordinates.
    """
    number_of_elements = len(support_elements)
    number_of_points = local_points.shape[1]

    points = _np.empty((number_of_elements, 3), dtype=_np.float64)

    for index, elem in enumerate(support_elements):
        points[number_of_points * index : number_of_points * (1 + index)] = (
            grid.get_element(elem).geometry.local2global(local_points).T
        )

    return points
