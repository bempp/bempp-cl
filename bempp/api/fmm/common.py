"""Common FMM routines."""
import abc as _abc
import numpy as _np
import numba as _numba


class Node(object):
    """Definition of an FMM node."""

    def __init__(
        self,
        identifier,
        center,
        radius,
        source_ids,
        target_ids,
        colleagues,
        is_leaf,
        level,
        parent,
    ):
        """Initialize a node."""

        self._identifier = identifier
        self._center = center
        self._radius = radius
        self._source_ids = _np.array(source_ids, dtype=_np.int64)
        self._target_ids = _np.array(target_ids, dtype=_np.int64)
        self._colleagues = _np.array(colleagues, dtype=_np.int64)
        self._is_leaf = is_leaf
        self._parent = parent
        self._level = level

    @property
    def center(self):
        """Return center."""
        return self._center

    @property
    def radius(self):
        """Return radius."""
        return self._radius

    @property
    def identifier(self):
        """Return Identifier."""
        return self._identifier

    @property
    def source_ids(self):
        """A list of source ids associated with the node."""
        return self._source_ids

    @property
    def target_ids(self):
        """A list of target ids associated with the node."""
        return self._target_ids

    @property
    def colleagues(self):
        """Return the colleagues of the node."""
        return self._colleagues

    @property
    def level(self):
        """Return the level."""
        return self._level

    @property
    def is_leaf(self):
        """Return true if leaf node, otherwise false."""
        return self._is_leaf

    @property
    def parent(self):
        """Return id of parent node."""
        return self._parent


class FmmInterface(_abc.ABC):
    """Interface to an FMM Instance."""

    def setup(domain, dual_to_range, regular_order, singular_order, *args, **kwargs):
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
    def leaf_node_keys(self):
        """
        Return a list of keys of non-empty leaf nodes.
        """
        raise NotImplementedError

    @property
    def nodes(self):
        """Return a (key, node) dictionary of all nodes."""
        raise NotImplementedError

    @property
    def source_transform(self):
        """Return source transformation matrix."""
        raise NotImplementedError

    @property
    def target_transform(self):
        """Return target transformation matrix."""
        raise NotImplementedError

    @property
    def source_grid(self):
        """Return source grid."""
        raise NotImplementedError

    @property
    def target_grid(self):
        """Return target grid."""
        raise NotImplementedError

    @property
    def sources(self):
        """Return sources."""
        raise NotImplementedError

    @property
    def targets(self):
        """Return target."""
        raise NotImplementedError

    def create_evaluator(self):
        """
        Return a Scipy Linear Operator that evaluates the FMM.

        The returned class should subclass the Scipy LinearOperator class
        so that it provides a matvec routine that accept a vector of coefficients
        and returns the result of a matrix vector product.
        """

    def _map_space_to_points(self, space, local_points, weights, mode):
        """Return mapper from grid coeffs to point evaluations."""
        from scipy.sparse import coo_matrix

        local_space = space.localised_space
        grid = local_space.grid
        number_of_local_points = local_points.shape[1]
        nshape_funs = space.number_of_shape_functions
        number_of_vertices = number_of_local_points * grid.number_of_elements

        global_dofs = []
        node_dofs = []
        values = []

        if mode == "source":
            attr = "source_ids"
        elif mode == "target":
            attr = "target_ids"
        else:
            raise ValueError("'mode' must be one of 'source' or 'target'.")

        for key in self.leaf_node_keys:
            vertex_ids = getattr(self.nodes[key], attr)
            associated_elements = set(
                [vertex // number_of_local_points for vertex in vertex_ids]
            )
            # Evaluate basis on the elements
            basis_values = {}
            for elem in associated_elements:
                # Spaces are scalar, so can use 2nd and 2rd component of eval
                basis_values[elem] = (
                    space.evaluate(elem, local_points)[0, :, :]
                    * weights
                    * grid.integration_elements[elem]
                )

            # Now fill up the matrix elements.
            for vertex in vertex_ids:
                elem = vertex // number_of_local_points
                local_point_index = vertex % number_of_local_points
                global_dofs.extend(space.local2global[elem, :])
                node_dofs.extend(nshape_funs * [vertex])
                values.extend(basis_values[elem][:, local_point_index])

        transform = coo_matrix(
            (values, (node_dofs, global_dofs)),
            shape=(number_of_vertices, local_space.global_dof_count),
        )

        return transform @ space.map_to_localised_space

    def _collect_near_field_indices(self, number_of_local_points):
        """Collect all indices of near-field points."""

        # Could also be target grid.
        # Only used if target and source grid identical
        elements = self.source_grid.elements

        grid_identical = self.source_grid == self.target_grid

        source_vertex_ids_dict = _numba.typed.Dict.empty(
            key_type=_numba.types.int64, value_type=_numba.types.int64[:]
        )

        target_vertex_ids_dict = _numba.typed.Dict.empty(
            key_type=_numba.types.int64, value_type=_numba.types.int64[:]
        )
        target_colleagues_dict = _numba.typed.Dict.empty(
            key_type=_numba.types.int64, value_type=_numba.types.int64[:]
        )

        for key in self.leaf_node_keys:
            target_vertex_ids_dict[key] = self.nodes[key].target_ids
            source_vertex_ids_dict[key] = self.nodes[key].source_ids
            target_colleagues_dict[key] = self.nodes[key].colleagues

        return _numba_collect_near_field_indices(
            number_of_local_points,
            elements,
            grid_identical,
            self.leaf_node_keys,
            source_vertex_ids_dict,
            target_vertex_ids_dict,
            target_colleagues_dict,
        )


# def _numba_collect_near_field_indices(
# number_of_local_points,
# elements,
# grid_identical,
# leaf_node_keys,
# source_vertex_ids_dict,
# target_vertex_ids_dict,
# target_colleagues_dict,
# ):
# source_indices = []
# target_indices = []

# if self.source_grid == self.target_grid:
# grid_identical = True
# grid = self.source_grid
# else:
# grid_identical = False

# for key in self.leaf_node_keys:
# target = self.nodes[key]
# if len(target.target_ids) == 0:
# continue
# for colleague in target.colleagues:
# if colleague == -1:
# continue
# source = self.nodes[colleague]
# if len(source.source_ids) == 0:
# continue
# for target_vertex in target.target_ids:
# target_elem = target_vertex // number_of_local_points
# for source_vertex in source.source_ids:
# source_elem = source_vertex // number_of_local_points
# if not grid_identical or source_elem == target_elem:
# continue
# if not source_elem in grid.element_neighbors[target_elem]:
# source_indices.append(source_vertex)
# target_indices.append(target_vertex)
# return target_indices, source_indices


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

    points = _np.empty((number_of_points * number_of_elements, 3), dtype=_np.float64)

    for index, elem in enumerate(support_elements):
        points[number_of_points * index : number_of_points * (1 + index)] = (
            grid.get_element(elem).geometry.local2global(local_points).T
        )

    return points


@_numba.njit(cache=True)
def _check_elements_adjacent(elem1, elem2):
    """Check if two elements are adjacent."""
    for e1 in elem1:
        for e2 in elem2:
            if e1 == e2:
                return True
    return False


# @_numba.njit(cache=True, locals={'elements_adjacent': _numba.types.boolean})
@_numba.njit(cache=True)
def _numba_collect_near_field_indices(
    number_of_local_points,
    elements,
    grid_identical,
    leaf_node_keys,
    source_vertex_ids_dict,
    target_vertex_ids_dict,
    target_colleagues_dict,
):
    """Collect all indices of near-field points."""

    source_indices = []
    target_indices = []

    for key in leaf_node_keys:
        target_vertices = target_vertex_ids_dict[key]
        if len(target_vertices) == 0:
            continue
        for colleague in target_colleagues_dict[key]:
            if colleague == -1:
                continue
            source_vertices = source_vertex_ids_dict[colleague]
            if len(source_vertices) == 0:
                continue
            for target_vertex in target_vertices:
                previous_source_elem = -1
                elements_adjacent = False
                target_elem = target_vertex // number_of_local_points
                for source_vertex in source_vertices:
                    source_elem = source_vertex // number_of_local_points
                    if not grid_identical or source_elem == target_elem:
                        continue
                    if source_elem != previous_source_elem:
                        elements_adjacent = _check_elements_adjacent(
                            elements[:, target_elem], elements[:, source_elem]
                        )
                        previous_source_elem = source_elem
                    if not elements_adjacent:
                        source_indices.append(source_vertex)
                        target_indices.append(target_vertex)
    return (
        _np.array(target_indices, dtype=_np.int64),
        _np.array(source_indices, dtype=_np.int64),
    )
