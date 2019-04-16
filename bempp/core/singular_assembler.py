"""Assembly of singular integrals of a boundary operator."""
import collections as _collections

import numpy as _np

from bempp.api.assembly import assembler as _assembler
from bempp.api.integration import duffy as _duffy

WORKGROUP_SIZE = 16


class SingularAssembler(_assembler.AssemblerBase):
    """Assembler for the singular part of boundary integral operators."""

    # pylint: disable=useless-super-delegation
    def __init__(self, domain, dual_to_range, parameters=None):
        """Instantiate the assembler."""
        super().__init__(domain, dual_to_range, parameters)

    def assemble(
        self, operator_descriptor, device_interface, precision, *args, **kwargs
    ):
        """Assemble the singular part."""
        from bempp.api.assembly.discrete_boundary_operator import (
            SparseDiscreteBoundaryOperator,
        )
        from bempp.api.utils.helpers import promote_to_double_precision
        from scipy.sparse import coo_matrix, csr_matrix
        from .dense_assembly_helpers import choose_source_name

        nrows = self.dual_to_range.global_dof_count
        ncols = self.domain.global_dof_count

        if self.domain.grid != self.dual_to_range.grid:
            # There are no singular elements if the grids
            # are different.
            return SparseDiscreteBoundaryOperator(
                csr_matrix((nrows, ncols), dtype="float64")
            )

        source_name = choose_source_name(operator_descriptor.compute_kernel)

        trial_local2global = self.domain.local2global.ravel()
        test_local2global = self.dual_to_range.local2global.ravel()
        trial_multipliers = self.domain.local_multipliers.ravel()
        test_multipliers = self.dual_to_range.local_multipliers.ravel()

        singular_rows, singular_cols, singular_values = assemble_singular_part(
            self.domain.localised_space,
            self.dual_to_range.localised_space,
            self.parameters,
            operator_descriptor,
            source_name,
            device_interface,
            precision,
        )

        rows = test_local2global[singular_rows]
        cols = trial_local2global[singular_cols]
        values = (
            singular_values
            * trial_multipliers[singular_cols]
            * test_multipliers[singular_rows]
        )

        if self.parameters.assembly.always_promote_to_double:
            values = promote_to_double_precision(values)

        return SparseDiscreteBoundaryOperator(
            coo_matrix((values, (rows, cols)), shape=(nrows, ncols)).tocsr()
        )


def assemble_singular_part(
    domain,
    dual_to_range,
    parameters,
    operator_descriptor,
    source_name,
    device_interface,
    precision,
):
    """
    Really assemble the singular part.

    Returns three arrays i, j, data, which contain the i-indices,
    j-indices, and computed data values for the singular part.

    """
    import bempp.core.cl_helpers as cl_helpers
    import bempp.api

    if domain.grid != dual_to_range.grid:
        raise ValueError("domain and dual_to_range must live on the same grid.")

    options = operator_descriptor.options.copy()
    options["WORKGROUP_SIZE"] = WORKGROUP_SIZE

    options["TEST"] = dual_to_range.shapeset.identifier
    options["TRIAL"] = domain.shapeset.identifier

    number_of_test_shape_functions = dual_to_range.number_of_shape_functions
    number_of_trial_shape_functions = domain.number_of_shape_functions

    options["NUMBER_OF_TEST_SHAPE_FUNCTIONS"] = number_of_test_shape_functions

    options["NUMBER_OF_TRIAL_SHAPE_FUNCTIONS"] = number_of_trial_shape_functions

    if "COMPLEX_KERNEL" in options:
        result_type = cl_helpers.get_type(precision).complex
    else:
        result_type = cl_helpers.get_type(precision).real

    order = parameters.quadrature.singular
    grid = domain.grid

    kernel_source = cl_helpers.kernel_source_from_identifier(
        source_name + "_singular", options
    )

    kernel = cl_helpers.Kernel(kernel_source, device_interface.context, precision)

    rule = _SingularQuadratureRuleInterface(grid, order, parameters)

    number_of_singular_indices = rule.index_count["all"]

    number_of_singular_values = (
        number_of_singular_indices
        * dual_to_range.number_of_shape_functions
        * domain.number_of_shape_functions
    )
    shape = (number_of_singular_values,)

    result_buffer = cl_helpers.DeviceBuffer(
        shape,
        result_type,
        device_interface.context,
        access_mode="write_only",
        order="C",
    )

    grid_buffer = grid.push_to_device(device_interface, precision).buffer

    quadrature_buffers = rule.push_to_device(
        device_interface, precision, WORKGROUP_SIZE
    )

    all_buffers = [grid_buffer, *quadrature_buffers, result_buffer]

    event = kernel.run(
        device_interface,
        (number_of_singular_indices,),
        (WORKGROUP_SIZE,),
        *all_buffers,
        g_times_l=True
    )

    event.wait()

    bempp.api.log("Singular kernel runtime [ms]: {0}".format(event.runtime()))

    irange = _np.arange(number_of_test_shape_functions)
    jrange = _np.arange(number_of_trial_shape_functions)

    i_ind = _np.tile(
        _np.repeat(irange, number_of_trial_shape_functions), len(rule.trial_indices)
    ) + _np.repeat(
        rule.test_indices * number_of_test_shape_functions,
        number_of_test_shape_functions * number_of_trial_shape_functions,
    )

    j_ind = _np.tile(
        _np.tile(jrange, number_of_test_shape_functions), len(rule.trial_indices)
    ) + _np.repeat(
        rule.trial_indices * number_of_trial_shape_functions,
        number_of_test_shape_functions * number_of_trial_shape_functions,
    )

    return (i_ind, j_ind, result_buffer.get_host_copy(device_interface))


_SingularQuadratureRule = _collections.namedtuple(
    "_QuadratureRule", "test_points trial_points weights"
)

_SingularityRuleDeviceBuffers = _collections.namedtuple(
    "SingularityRuleDeviceBuffers",
    "test_points trial_points weights test_indices trial_indices"
    + " test_offsets trial_offsets"
    + " weights_offsets number_of_local_quad_points",
)


class _SingularQuadratureRuleInterface(object):
    """Interface for a singular quadrature rule."""

    def __init__(self, grid, order, parameters):
        """Initialize singular quadrature rule."""

        self._grid = grid
        self._order = order
        self._parameters = parameters
        self._test_indices = None
        self._trial_indices = None

        self._coincident_rule = _SingularQuadratureRule(
            *_duffy.rule(order, "coincident")
        )

        self._edge_adjacent_rule = _SingularQuadratureRule(
            *_duffy.rule(order, "edge_adjacent")
        )

        self._vertex_adjacent_rule = _SingularQuadratureRule(
            *_duffy.rule(order, "vertex_adjacent")
        )

        self._index_count = {}

        self._index_count["coincident"] = self.number_of_elements
        self._index_count["edge_adjacent"] = self.edge_adjacency.shape[1]
        self._index_count["vertex_adjacent"] = self.vertex_adjacency.shape[1]

        self._index_count["all"] = (
            self._index_count["coincident"]
            + self._index_count["edge_adjacent"]
            + self._index_count["vertex_adjacent"]
        )

    @property
    def order(self):
        """Return the order."""
        return self._order

    @property
    def parameters(self):
        """Return parameters."""
        return self._parameters

    @property
    def coincident_rule(self):
        """Return coincident rule."""
        return self._coincident_rule

    @property
    def edge_adjacent_rule(self):
        """Return edge adjacent rule."""
        return self._edge_adjacent_rule

    @property
    def vertex_adjacent_rule(self):
        """Return vertex adjacent rule."""
        return self._vertex_adjacent_rule

    @property
    def grid(self):
        """Return the grid."""
        return self._grid

    @property
    def edge_adjacency(self):
        """Return the grid edge adjacency information."""
        return self.grid.edge_adjacency

    @property
    def vertex_adjacency(self):
        """Return vertex adjacency."""
        return self.grid.vertex_adjacency

    @property
    def number_of_elements(self):
        """Return the number of elements of the underlying grid."""
        return self.grid.number_of_elements

    @property
    def index_count(self):
        """Return the index count."""
        return self._index_count

    @property
    def test_indices(self):
        """Return the test indicies of all singular contributions."""
        return self._test_indices

    @property
    def trial_indices(self):
        """Return the trial indicies of all singular contributions."""
        return self._trial_indices

    def number_of_points(self, adjacency):
        """Return the number of quadrature points for given adjacency."""
        return _duffy.number_of_quadrature_points(self.order, adjacency)

    def push_to_device(self, device_interface, precision, workgroup_size):
        """Push quadrature rule to a given device."""
        from bempp.core.cl_helpers import DeviceBuffer
        from bempp.core.cl_helpers import get_type

        types = get_type(precision)

        test_indices, trial_indices = self._vectorize_indices()
        test_points, trial_points = self._vectorize_points()
        weights = self._vectorize_weights()
        test_offsets, trial_offsets, weights_offsets = self._vectorize_offsets()

        number_of_local_quad_points = self._vectorized_local_number_of_integration_points(
            workgroup_size
        )

        self._test_indices = test_indices
        self._trial_indices = trial_indices

        arrays = [
            test_points,
            trial_points,
            weights,
            test_indices,
            trial_indices,
            test_offsets,
            trial_offsets,
            weights_offsets,
            number_of_local_quad_points,
        ]

        dtypes = [
            types.real,
            types.real,
            types.real,
            "uint32",
            "uint32",
            "uint32",
            "uint32",
            "uint32",
            "uint32",
        ]

        buffers = [
            DeviceBuffer.from_array(
                array,
                device_interface,
                dtype=dtype,
                access_mode="read_write",
                order="F",
            )
            for array, dtype in zip(arrays, dtypes)
        ]

        device_buffers = _SingularityRuleDeviceBuffers(*buffers)

        return device_buffers

    def _collect_remapped_quad_points_for_edge_adjacent_rule(self, quad_points):
        """
        Remap quad points for edge adjacent quadrature rules.

        Given a 2xN array of quadrature points, return all possible
        combinations of remapped rules, according to the following
        order

        0: edge (index 0, 1)
        1: edge (index 1, 0)
        2: edge (index 1, 2)
        3: edge (index 2, 1)
        4: edge (index 0, 2)
        5: edge (index 2, 0)
        """
        return _np.hstack(
            [
                _duffy.remap_points_shared_edge(quad_points, 0, 1),
                _duffy.remap_points_shared_edge(quad_points, 1, 0),
                _duffy.remap_points_shared_edge(quad_points, 1, 2),
                _duffy.remap_points_shared_edge(quad_points, 2, 1),
                _duffy.remap_points_shared_edge(quad_points, 0, 2),
                _duffy.remap_points_shared_edge(quad_points, 2, 0),
            ]
        )

    def _collect_remapped_quad_points_for_vertex_adjacent_rule(self, quad_points):
        """
        Remap quad points for vertex adjacent quadrature rules.

        Given a 2xN array of quadrature points, return all possible
        combinations of remapped rules, according to the following
        order

        0: vertex (index 0)
        1: vertex (index 1)
        2: vertex (index 2)
        """
        return _np.hstack(
            [
                _duffy.remap_points_shared_vertex(quad_points, 0),
                _duffy.remap_points_shared_vertex(quad_points, 1),
                _duffy.remap_points_shared_vertex(quad_points, 2),
            ]
        )

    def _compute_edge_offsets(self):
        """Compute offsets for the edge based rule."""

        ncoincident = self.number_of_points("coincident")
        nedge_adjacent = self.number_of_points("edge_adjacent")

        # Offset values is a 3 x 3 matrix such that
        # the value (i, j) is the offset of the rule
        # associted with the (i, j) remap case,
        # where i and j are 0, 1, or 2. The diagonal
        # elements are not needed, so just set to -1.

        offset_values = _np.array([[-1, 0, 4], [1, -1, 2], [5, 3, -1]])

        edge_offsets = ncoincident + nedge_adjacent * offset_values

        return edge_offsets

    def _compute_vertex_offsets(self):
        """Compute offsets for the vertex based rules."""
        ncoincident = self.number_of_points("coincident")
        nedge_adjacent = self.number_of_points("edge_adjacent")
        nvertex_adjacent = self.number_of_points("vertex_adjacent")

        vertex_offsets = (
            ncoincident
            + 6 * nedge_adjacent
            + nvertex_adjacent * _np.arange(3, dtype="uint32")
        )

        return vertex_offsets

    def _vectorize_indices(self):
        """Return vector of test and trial indices for sing. integration."""
        test_indices = _np.empty(self.index_count["all"], dtype="uint32")
        trial_indices = _np.empty(self.index_count["all"], dtype="uint32")

        for array, index in zip([test_indices, trial_indices], [0, 1]):
            array[: self.index_count["coincident"]] = _np.arange(
                self.index_count["coincident"]
            )

            array[
                self.index_count["coincident"] : (
                    self.index_count["coincident"] + self.index_count["edge_adjacent"]
                )
            ] = self.edge_adjacency[index, :]

            array[-self.index_count["vertex_adjacent"] :] = self.vertex_adjacency[
                index, :
            ]

        return test_indices, trial_indices

    def _vectorized_local_number_of_integration_points(self, workgroup_size):
        """Compute an array of local numbers of integration points."""
        number_of_local_quad_points = _np.empty(self.index_count["all"], dtype="uint32")

        number_of_local_quad_points[: self.index_count["coincident"]] = (
            self.number_of_points("coincident") // workgroup_size
        )
        number_of_local_quad_points[
            self.index_count["coincident"] : (
                self.index_count["coincident"] + self.index_count["edge_adjacent"]
            )
        ] = (self.number_of_points("edge_adjacent") // workgroup_size)
        number_of_local_quad_points[-self.index_count["vertex_adjacent"] :] = (
            self.number_of_points("vertex_adjacent") // workgroup_size
        )

        return number_of_local_quad_points

    def _vectorize_points(self):
        """Return an array of all quadrature points for all different rules."""

        test_points = _np.hstack(
            [
                self.coincident_rule.test_points,
                self._collect_remapped_quad_points_for_edge_adjacent_rule(
                    self.edge_adjacent_rule.test_points
                ),
                self._collect_remapped_quad_points_for_vertex_adjacent_rule(
                    self.vertex_adjacent_rule.test_points
                ),
            ]
        )

        trial_points = _np.hstack(
            [
                self.coincident_rule.trial_points,
                self._collect_remapped_quad_points_for_edge_adjacent_rule(
                    self.edge_adjacent_rule.trial_points
                ),
                self._collect_remapped_quad_points_for_vertex_adjacent_rule(
                    self.vertex_adjacent_rule.trial_points
                ),
            ]
        )

        return test_points, trial_points

    def _vectorize_weights(self):
        """Vectorize the quadrature weights."""
        weights = _np.hstack(
            [
                self.coincident_rule.weights,
                self.edge_adjacent_rule.weights,
                self.vertex_adjacent_rule.weights,
            ]
        )

        return weights

    def _vectorize_offsets(self):
        """Vectorize the offsets."""
        edge_offsets = self._compute_edge_offsets()
        vertex_offsets = self._compute_vertex_offsets()

        test_offsets = _np.empty(self.index_count["all"], dtype="uint32")
        trial_offsets = _np.empty(self.index_count["all"], dtype="uint32")
        weights_offsets = _np.empty(self.index_count["all"], dtype="uint32")

        test_offsets[: self.index_count["coincident"]] = 0

        test_offsets[
            self.index_count["coincident"] : (
                self.index_count["coincident"] + self.index_count["edge_adjacent"]
            )
        ] = edge_offsets[self.edge_adjacency[2, :], self.edge_adjacency[3, :]]

        test_offsets[-self.index_count["vertex_adjacent"] :] = vertex_offsets[
            self.vertex_adjacency[2, :]
        ]

        trial_offsets[: self.index_count["coincident"]] = _np.zeros(
            self.index_count["coincident"]
        )

        trial_offsets[
            self.index_count["coincident"] : (
                self.index_count["coincident"] + self.index_count["edge_adjacent"]
            )
        ] = edge_offsets[self.edge_adjacency[4, :], self.edge_adjacency[5, :]]

        trial_offsets[-self.index_count["vertex_adjacent"] :] = vertex_offsets[
            self.vertex_adjacency[3, :]
        ]

        weights_offsets[: self.index_count["coincident"]] = 0
        weights_offsets[
            self.index_count["coincident"] : (
                self.index_count["coincident"] + self.index_count["edge_adjacent"]
            )
        ] = self.number_of_points("coincident")
        weights_offsets[-self.index_count["vertex_adjacent"] :] = self.number_of_points(
            "coincident"
        ) + self.number_of_points("edge_adjacent")

        return test_offsets, trial_offsets, weights_offsets
