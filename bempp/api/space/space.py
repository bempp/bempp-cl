"""Generic access to spaces."""

import abc as _abc
from collections import namedtuple as _namedtuple

import numpy as _np
import numba as _numba


def function_space(
    grid,
    kind,
    degree,
    support_elements=None,
    segments=None,
    swapped_normals=None,
    **kwargs
):
    """Initialize a function space."""

    if _np.count_nonzero([support_elements, segments]) > 1:
        raise ValueError(
            "Only one of 'support_elements' and 'segments' must be nonzero."
        )

    if kind == "DP":
        if degree == 0:
            from .p0_discontinuous_space import P0DiscontinuousFunctionSpace

            return P0DiscontinuousFunctionSpace(
                grid, support_elements, segments, swapped_normals
            )
        if degree == 1:
            from .p1_discontinuous_space import P1DiscontinuousFunctionSpace

            return P1DiscontinuousFunctionSpace(
                grid, support_elements, segments, swapped_normals
            )

    if kind == "P":
        if degree == 1:
            from .p1_continuous_space import P1ContinuousFunctionSpace

            return P1ContinuousFunctionSpace(
                grid, support_elements, segments, swapped_normals, **kwargs
            )

    if kind == "RWG":
        if degree == 0:
            from .rwg0_space import Rwg0FunctionSpace

            return Rwg0FunctionSpace(
                grid, support_elements, segments, swapped_normals, **kwargs
            )

    if kind == "SNC":
        if degree == 0:
            from .snc0_space import Snc0FunctionSpace

            return Snc0FunctionSpace(
                grid, support_elements, segments, swapped_normals, **kwargs
            )

    raise ValueError("Requested space not implemented.")


_SpaceData = _namedtuple(
    "SpaceData",  # pylint: disable=C0103
    [
        "grid",
        "codomain_dimension",
        "order",
        "shapeset",
        "local2global_map",
        "local_multipliers",
        "identifier",
        "support",
        "localised_space",
        "normal_multipliers",
        "dof_transformation",
        "requires_dof_transformation",
        "is_barycentric",
        "barycentric_representation",
    ],
)


class _FunctionSpace(_abc.ABC):
    """Base class for function spaces."""

    def __init__(self, space_data):
        """Initialisation of base class."""

        from .shapesets import Shapeset
        from scipy.sparse import coo_matrix
        from scipy.sparse import identity

        self._grid = space_data.grid
        self._grid_id = self._grid.id
        self._id_string = None
        self._codomain_dimension = space_data.codomain_dimension
        self._order = space_data.order
        self._shapeset = Shapeset(space_data.shapeset)
        self._local2global_map = space_data.local2global_map
        self._local_multipliers = space_data.local_multipliers
        self._identifier = space_data.identifier
        self._support = space_data.support
        self._localised_space = space_data.localised_space
        self._color_map = None
        self._requires_dof_transformation = space_data.requires_dof_transformation
        self._is_barycentric = space_data.is_barycentric
        self._barycentric_representation = space_data.barycentric_representation
        self._dof_transformation = space_data.dof_transformation
        self._global2local_map = self._invert_local2global_map(
            self._local2global_map,
            self._grid.number_of_elements,
            self.dof_transformation.shape[0],
        )

        self._normal_multipliers = space_data.normal_multipliers

        self._number_of_support_elements = _np.count_nonzero(self._support)
        self._support_elements = _np.flatnonzero(self._support).astype("uint32")

        self._mass_matrix = None
        self._inverse_mass_matrix = None

        self._sorted_indices = None
        self._indexptr = None

        nshape_fun = self.number_of_shape_functions

        self._map_to_localised_space = coo_matrix(
            (
                self._local_multipliers[self._support].ravel(),
                (
                    _np.arange(nshape_fun * self._number_of_support_elements),
                    self._local2global_map[self._support].ravel(),
                ),
            ),
            shape=(
                nshape_fun * self._number_of_support_elements,
                self.global_dof_count,
            ),
            dtype="float64",
        ).tocsr()

        self._map_to_full_grid = coo_matrix(
            (
                self._local_multipliers[self._support].ravel(),
                (
                    nshape_fun * _np.repeat(self._support_elements, nshape_fun)
                    + _np.tile(
                        _np.arange(nshape_fun), self._number_of_support_elements
                    ),
                    self._local2global_map[self._support].ravel(),
                ),
            ),
            shape=(
                nshape_fun * self._grid.number_of_elements,
                self.dof_transformation.shape[0],
            ),
            dtype="float64",
        ).tocsr()

    @property
    def grid(self):
        """Return the grid."""
        return self._grid

    @property
    def grid_id(self):
        """Return id of base grid."""
        return self._grid_id

    @property
    def codomain_dimension(self):
        """Return the codomain dimension."""
        return self._codomain_dimension

    @property
    def global_dof_count(self):
        """Return the global dof count."""
        return self._dof_transformation.shape[1]

    @property
    def order(self):
        """Return order of the space."""
        return self._order

    @property
    def local2global(self):
        """Return local to global map."""
        return self._local2global_map

    @property
    def local_multipliers(self):
        """Return the multipliers for each local dof."""
        return self._local_multipliers

    @property
    def normal_multipliers(self):
        """Return the normal multipliers for each grid element."""
        return self._normal_multipliers

    @property
    def global2local(self):
        """Return global to local map."""
        return self._global2local_map

    @property
    def number_of_shape_functions(self):
        """Return the number of shape functions on each element."""
        return self._shapeset.number_of_shape_functions

    @property
    def number_of_support_elements(self):
        """The number of elements that form the support."""
        return self._number_of_support_elements

    @property
    def support_elements(self):
        """Return the list of elements on which space is supported."""
        return self._support_elements

    @property
    def identifier(self):
        """Return the identifier."""
        return self._identifier

    @property
    def support(self):
        """Return support of the space."""
        return self._support

    @property
    def shapeset(self):
        """Return the shapeset."""
        return self._shapeset

    @property
    def localised_space(self):
        """Return the elementwise defined space."""
        return self._localised_space

    @property
    def is_localised(self):
        """Return true if space is localised."""
        return self.localised_space == self

    @property
    def color_map(self):
        """
        Return a coloring of the grid associated with the space.

        The coloring is defined such that if two elements have
        the same color then their associated global degrees of
        freedom do not intersect. This is important for the
        efficient summation of global dofs during the assembly.
        """
        if self._color_map is None:
            self._compute_color_map()
        return self._color_map

    @property
    def map_to_localised_space(self):
        """Return a sparse matrix that maps dofs to localised space."""

        return self._map_to_localised_space

    @property
    def map_to_full_grid(self):
        """Return a sparse matrix that maps dofs to localised space on full grid."""

        return self._map_to_full_grid

    @property
    def dof_transformation(self):
        """
        Transformation from global dofs to space dofs.
        """
        return self._dof_transformation

    @property
    def requires_dof_transformation(self):
        """True if the dof transformation matrix is not the identity."""
        return self._requires_dof_transformation

    @property
    def is_barycentric(self):
        """Return true if space is defined over barycentric grid."""
        return self._is_barycentric

    @property
    def id_string(self):
        """Return an id string for space comparison."""
        if self._id_string is None:
            self._id_string = self._generate_hash()

        return self._id_string

    def barycentric_representation(self):
        """Return barycentric_representation if it exists."""
        import collections

        # Lazy evaluation. By default only a function object is created.
        # On first call this is executed to produce the space.
        if isinstance(self._barycentric_representation, collections.abc.Callable):
            self._barycentric_representation = self._barycentric_representation()
        return self._barycentric_representation

    def get_elements_by_color(self):
        """
        Returns color sorted elements and their index positions.

        This method returns a tuple (sorted_indices, indexptr) so that
        all element indices with color i are contained in
        sorted_indices[indexptr[i]:indexptr[i+1]].

        """
        if self._sorted_indices is None:
            self._sort_elements_by_color()
        return (self._sorted_indices, self._indexptr)

    def evaluate(self, element_index, local_coordinates):
        """Evaluate the basis on an element."""
        return self.numba_evaluate(
            element_index,
            self.shapeset.evaluate,
            local_coordinates,
            self.grid.data,
            self.local_multipliers,
            self.normal_multipliers,
        )

    def surface_gradient(self, element_index, local_coordinates):
        """Return the surface gradient."""
        return self.numba_surface_gradient(
            element_index,
            self.shapeset.gradient,
            local_coordinates,
            self.grid.data,
            self.local_multipliers,
            self.normal_multipliers,
        )

    def mass_matrix(self):
        """Return the mass matrix associated with this space."""

        if self._mass_matrix is None:
            from bempp.api.operators.boundary.sparse import identity

            self._mass_matrix = identity(self, self, self).weak_form()

        return self._mass_matrix

    def inverse_mass_matrix(self):
        """Return the inverse mass matrix for this space."""

        from bempp.api.assembly.discrete_boundary_operator import (
            InverseSparseDiscreteBoundaryOperator,
        )

        if self._inverse_mass_matrix is None:
            self._inverse_mass_matrix = InverseSparseDiscreteBoundaryOperator(
                self.mass_matrix()
            )
        return self._inverse_mass_matrix

    def _generate_hash(self):
        """Generate a hash for the space object."""
        from hashlib import md5

        dof_transform_csr = self.dof_transformation.tocsr().sorted_indices()

        md5_gen = md5()

        md5_gen.update(self.local2global.tobytes())
        md5_gen.update(self.support_elements.tobytes())
        md5_gen.update(self.normal_multipliers.tobytes())
        md5_gen.update(self.local_multipliers.tobytes())
        md5_gen.update(dof_transform_csr.indices)
        md5_gen.update(dof_transform_csr.indptr)
        md5_gen.update(dof_transform_csr.data)

        return self.identifier + "_" + self.grid_id + "_" + md5_gen.hexdigest()

    def is_compatible(self, other):
        """Check if space is compatible with other space."""
        return self == other

    def vertex_on_boundary(self):
        """Return true if vertex is on boundary of segment."""

    def _invert_local2global_map(
        self, local2global_map, number_of_elements, global_dof_count
    ):
        """Obtain the global to local dof map from the local to global map."""

        global2local_map = [[] for _ in range(global_dof_count)]

        for elem_index in range(number_of_elements):
            for local_index, dof in enumerate(local2global_map[elem_index]):
                if self.local_multipliers[elem_index, local_index] != 0:
                    global2local_map[dof].append((elem_index, local_index))

        for index, elem in enumerate(global2local_map):
            global2local_map[index] = tuple(elem)

        return global2local_map

    def _compute_color_map(self):
        """Compute the color map."""

        self._color_map = -_np.ones(self.grid.number_of_elements, dtype=_np.int32)
        for element_index in self.support_elements:
            neighbors = {element_index}
            global_dofs = self.local2global[element_index]
            for dof in global_dofs:
                for elem, _ in self.global2local[dof]:
                    neighbors.add(elem)
            neighbors.remove(element_index)

            neighbor_colors = self._color_map[list(neighbors)]
            self._color_map[element_index] = next(
                color
                for color in range(self.number_of_support_elements)
                if color not in neighbor_colors
            )

    def _sort_elements_by_color(self):
        """Implement elements by color computation."""
        sorted_indices = _np.empty(self.number_of_support_elements, dtype="uint32")
        ncolors = 1 + max(self.color_map)
        indexptr = _np.zeros(1 + ncolors, dtype="uint32")

        count = 0
        for index, color in enumerate(_np.arange(ncolors, dtype="uint32")):
            colors = _np.where(self.color_map == color)[0]
            colors_length = len(colors)
            sorted_indices[count : count + colors_length] = colors
            count += colors_length
            indexptr[index + 1] = count
        self._sorted_indices, self._indexptr = sorted_indices, indexptr

    def __eq__(self, other):
        """Check if spaces are compatible."""
        return check_if_compatible(self, other)


def _process_segments(grid, support_elements, segments, swapped_normals):
    """Pocess information from support_elements and segments vars."""

    if support_elements is not None and segments is not None:
        raise ValueError(
            "Only one of 'support_elements' and 'segments' must be nonzero."
        )

    if swapped_normals is None:
        swapped_normals = {}

    number_of_elements = grid.number_of_elements
    normal_multipliers = _np.zeros(number_of_elements, dtype=_np.int32)

    for element_index in range(number_of_elements):
        if grid.domain_indices[element_index] in swapped_normals:
            normal_multipliers[element_index] = -1
        else:
            normal_multipliers[element_index] = 1

    if support_elements is not None:
        support = _np.full(number_of_elements, False, dtype=bool)
        support[support_elements] = True
    elif segments is not None:
        support = _np.full(number_of_elements, False, dtype=bool)
        for element_index in range(number_of_elements):
            if grid.domain_indices[element_index] in segments:
                support[element_index] = True
    else:
        support = _np.full(number_of_elements, True, dtype=bool)

    return support, normal_multipliers


def return_compatible_representation(*args):
    """Return representation of spaces on same grid."""

    # Check if at least one space is barycentric.

    is_barycentric = any([space.is_barycentric for space in args])

    if not is_barycentric:
        return args
    else:
        # Convert spaces
        converted = [space.barycentric_representation() for space in args]
        if not all(converted):
            raise ValueError("Not all spaces have a valid barycentric representation.")
        return converted


def check_if_compatible(space1, space2):
    """Return true if two spaces are compatible."""

    if id(space1) == id(space2):
        return True

    try:
        new_space1, new_space2 = return_compatible_representation(space1, space2)
        return new_space1.id_string == new_space2.id_string
    except:
        return False
