"""Unit tests for the singular assembler."""

# pylint: disable=redefined-outer-name
# pylint: disable=C0103

import numpy as _np
import pytest

WORKGROUP_SIZE = 16
ORDER = 4

pytestmark = pytest.mark.usefixtures(
    "default_parameters", "small_sphere", "small_piecewise_const_space", "helpers"
)


@pytest.fixture()
def singular_rule_interface(small_sphere, default_parameters):
    """Create a singular quadrature rule interface object."""
    from bempp.core.singular_assembler import _SingularQuadratureRuleInterface

    return _SingularQuadratureRuleInterface(small_sphere, ORDER, default_parameters)


@pytest.fixture(params=["single", "double"])
def expected_vectorized_rules(small_sphere, request):
    """Comparison fixture for vectorized quadrature rules."""
    from bempp.api.integration import duffy

    workgroup_size = WORKGROUP_SIZE
    order = ORDER
    grid = small_sphere

    dtype = {"single": "float32", "double": "float64"}[request.param]

    xc, yc, wc = duffy.rule(order, "coincident")
    xe, ye, we = duffy.rule(order, "edge_adjacent")
    xv, yv, wv = duffy.rule(order, "vertex_adjacent")

    nc = len(wc)
    ne = len(we)
    nv = len(wv)

    edge_adjacency = grid.edge_adjacency
    vertex_adjacency = grid.vertex_adjacency

    # We have the following types of quadrature rules
    # 0: coincident
    # 1: edge (index 0, 1)
    # 2: edge (index 1, 0)
    # 3: edge (index 1, 2)
    # 4: edge (index 2, 1)
    # 5: edge (index 0, 2)
    # 6: edge (index 2, 0)
    # 7: vertex (index 0)
    # 8: vertex (index 1)
    # 9: vertex (index 2)

    n_coincident = grid.entity_count(0)
    n_edge_indices = edge_adjacency.shape[1]
    n_vertex_indices = vertex_adjacency.shape[1]
    n_total = n_coincident + n_edge_indices + n_vertex_indices

    test_points = _np.hstack(
        [
            xc,
            duffy.remap_points_shared_edge(xe, 0, 1),
            duffy.remap_points_shared_edge(xe, 1, 0),
            duffy.remap_points_shared_edge(xe, 1, 2),
            duffy.remap_points_shared_edge(xe, 2, 1),
            duffy.remap_points_shared_edge(xe, 0, 2),
            duffy.remap_points_shared_edge(xe, 2, 0),
            duffy.remap_points_shared_vertex(xv, 0),
            duffy.remap_points_shared_vertex(xv, 1),
            duffy.remap_points_shared_vertex(xv, 2),
        ]
    )

    trial_points = _np.hstack(
        [
            yc,
            duffy.remap_points_shared_edge(ye, 0, 1),
            duffy.remap_points_shared_edge(ye, 1, 0),
            duffy.remap_points_shared_edge(ye, 1, 2),
            duffy.remap_points_shared_edge(ye, 2, 1),
            duffy.remap_points_shared_edge(ye, 0, 2),
            duffy.remap_points_shared_edge(ye, 2, 0),
            duffy.remap_points_shared_vertex(yv, 0),
            duffy.remap_points_shared_vertex(yv, 1),
            duffy.remap_points_shared_vertex(yv, 2),
        ]
    )

    weights = _np.hstack([wc, we, wv])

    edge_offsets = nc + ne * _np.array(
        [[-1, 0, 4], [1, -1, 2], [5, 3, -1]], dtype="uint32"
    )
    vertex_offsets = nc + 6 * ne + nv * _np.arange(3, dtype="uint32")

    test_indices = _np.zeros(n_total, dtype="uint32")
    trial_indices = _np.zeros(n_total, dtype="uint32")
    test_offsets = _np.zeros(n_total, dtype="uint32")
    trial_offsets = _np.zeros(n_total, dtype="uint32")
    weights_offsets = _np.zeros(n_total, dtype="uint32")
    number_of_local_quad_points = _np.zeros(n_total, dtype="uint32")

    test_indices[:n_coincident] = _np.arange(n_coincident)
    trial_indices[:n_coincident] = _np.arange(n_coincident)
    test_indices[n_coincident : (n_coincident + n_edge_indices)] = edge_adjacency[0, :]
    trial_indices[n_coincident : (n_coincident + n_edge_indices)] = edge_adjacency[1, :]
    test_indices[(n_coincident + n_edge_indices) :] = vertex_adjacency[0, :]
    trial_indices[(n_coincident + n_edge_indices) :] = vertex_adjacency[1, :]

    test_offsets[:n_coincident] = _np.zeros(n_coincident)
    trial_offsets[:n_coincident] = _np.zeros(n_coincident)
    test_offsets[n_coincident : (n_coincident + n_edge_indices)] = edge_offsets[
        edge_adjacency[2, :], edge_adjacency[3, :]
    ]
    trial_offsets[n_coincident : (n_coincident + n_edge_indices)] = edge_offsets[
        edge_adjacency[4, :], edge_adjacency[5, :]
    ]
    test_offsets[(n_coincident + n_edge_indices) :] = vertex_offsets[
        vertex_adjacency[2, :]
    ]
    trial_offsets[(n_coincident + n_edge_indices) :] = vertex_offsets[
        vertex_adjacency[3, :]
    ]

    weights_offsets[:n_coincident] = 0
    weights_offsets[n_coincident : (n_coincident + n_edge_indices)] = nc
    weights_offsets[(n_coincident + n_edge_indices) :] = nc + ne

    number_of_local_quad_points[:n_coincident] = nc // workgroup_size
    number_of_local_quad_points[n_coincident : (n_coincident + n_edge_indices)] = (
        ne // workgroup_size
    )
    number_of_local_quad_points[(n_coincident + n_edge_indices) :] = (
        nv // workgroup_size
    )

    fix_types = lambda a: _np.require(a, dtype=dtype, requirements="F")

    test_points = fix_types(test_points)
    trial_points = fix_types(trial_points)
    weights = fix_types(weights)

    return (
        test_points,
        trial_points,
        weights,
        test_indices,
        trial_indices,
        test_offsets,
        trial_offsets,
        weights_offsets,
        number_of_local_quad_points,
    )


def test_vectorized_arrays_of_singular_quadrature_rules(
    expected_vectorized_rules, singular_rule_interface, device_interface, precision
):
    """Test vectorization of singular quadrature rules."""
    from bempp.core import cl_helpers

    buffers = singular_rule_interface.push_to_device(
        device_interface, precision, WORKGROUP_SIZE
    )

    # for buffer, actual in zip(buffers, actual_vectorized_rules):

    buf_descriptor = {
        0: "test_points",
        1: "trial_points",
        2: "weights",
        3: "test_indices",
        4: "trial_indices",
        5: "test_offsets",
        6: "trial_offsets",
        7: "weights_offsets",
        8: "number_of_local_quad_points",
    }

    for index, (buffer, expected) in enumerate(zip(buffers, expected_vectorized_rules)):
        _np.testing.assert_allclose(
            buffer.get_host_copy(device_interface),
            expected,
            err_msg="Error in comparing actual and expected "
            + buf_descriptor[index]
            + " arrays.",
        )


def test_laplace_singular_assembler(
    small_piecewise_const_space,
    default_parameters,
    helpers,
    device_interface,
    precision,
):
    """Test singular assembler for the Laplace slp on a small sphere."""
    from bempp.api.operators.boundary.laplace import single_layer

    default_parameters.quadrature.singular = 6

    discrete_op = single_layer(
        small_piecewise_const_space,
        small_piecewise_const_space,
        small_piecewise_const_space,
        assembler="only_singular_part",
        parameters=default_parameters,
        device_interface=device_interface,
        precision=precision,
    ).assemble()

    coo_mat = discrete_op.A.tocoo()
    rows = coo_mat.row
    cols = coo_mat.col
    data = coo_mat.data

    mat = helpers.load_npy_data("laplace_small_sphere_p0_disc")
    expected = mat[rows, cols]
    _np.testing.assert_allclose(data, expected, rtol=1e-5)
