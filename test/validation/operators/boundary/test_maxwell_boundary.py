"""Unit tests for modified Helmholtz operators."""

# pylint: disable=redefined-outer-name
# pylint: disable=C0103

import numpy as _np
import pytest

pytestmark = pytest.mark.usefixtures("default_parameters", "helpers")


def test_maxwell_electric_field_sphere(
    default_parameters, helpers, device_interface, precision
):
    """Test Maxwell electric field on sphere."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.maxwell import electric_field

    grid = helpers.load_grid("sphere")

    space1 = function_space(grid, "RWG", 0)
    space2 = function_space(grid, "SNC", 0)

    discrete_op = electric_field(
        space1,
        space1,
        space2,
        2.5,
        assembler="dense",
        device_interface=device_interface,
        precision=precision,
        parameters=default_parameters,
    ).weak_form()

    if precision == "single":
        rtol = 1e-5
        atol = 1e-7
    else:
        rtol = 1e-10
        atol = 1e-14

    expected = helpers.load_npy_data("maxwell_electric_field_boundary")
    _np.testing.assert_allclose(discrete_op.A, expected, rtol=rtol, atol=atol)


def test_maxwell_electric_field_rbc_bc_sphere(
    default_parameters, helpers, device_interface, precision
):
    """Test Maxwell electric field on sphere with RBC/BC basis."""
    import bempp.api
    from bempp.api import function_space
    from bempp.api.operators.boundary.maxwell import electric_field

    grid = helpers.load_grid("sphere")

    space1 = function_space(grid, "BC", 0)
    space2 = function_space(grid, "RBC", 0)

    rand = _np.random.RandomState(0)
    vec = rand.rand(space1.global_dof_count)

    bempp.api.GLOBAL_PARAMETERS.fmm.dense_evaluation = True

    discrete_op = electric_field(
        space1,
        space1,
        space2,
        2.5,
        assembler="fmm",
        device_interface=device_interface,
        precision=precision,
        parameters=default_parameters,
    ).weak_form()

    actual = discrete_op @ vec

    bempp.api.GLOBAL_PARAMETERS.fmm.dense_evaluation = False

    if precision == "single":
        rtol = 5e-5
        atol = 5e-6
    else:
        rtol = 1e-10
        atol = 1e-14

    mat = helpers.load_npy_data("maxwell_electric_field_boundary_rbc_bc")

    expected = mat @ vec

    _np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)

    bempp.api.clear_fmm_cache()


def test_maxwell_electric_field_bc_sphere(
    default_parameters, helpers, device_interface, precision
):
    """Test Maxwell electric field on sphere with BC basis."""
    import bempp.api
    from bempp.api import function_space
    from bempp.api.operators.boundary.maxwell import electric_field

    grid = helpers.load_grid("sphere")

    space1 = function_space(grid, "BC", 0)
    space2 = function_space(grid, "SNC", 0)

    rand = _np.random.RandomState(0)
    vec = rand.rand(space1.global_dof_count)

    bempp.api.GLOBAL_PARAMETERS.fmm.dense_evaluation = True

    discrete_op = electric_field(
        space1,
        space1,
        space2,
        2.5,
        assembler="fmm",
        device_interface=device_interface,
        precision=precision,
        parameters=default_parameters,
    ).weak_form()

    actual = discrete_op @ vec

    bempp.api.GLOBAL_PARAMETERS.fmm.dense_evaluation = False

    if precision == "single":
        rtol = 1e-5
        atol = 1e-6
    else:
        rtol = 1e-10
        atol = 1e-14

    mat = helpers.load_npy_data("maxwell_electric_field_boundary_bc")
    expected = mat @ vec
    _np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)

    bempp.api.clear_fmm_cache()


# def test_maxwell_electric_field_complex_sphere(
#     default_parameters, helpers, device_interface, precision
# ):
#     """Test Maxwell electric field on sphere with complex wavenumber."""
#     from bempp.api import get_precision
#     from bempp.api import function_space
#     from bempp.api.operators.boundary.maxwell import electric_field

#     grid = helpers.load_grid("sphere")

#     space1 = function_space(grid, "RWG", 0)
#     space2 = function_space(grid, "SNC", 0)

#     discrete_op = electric_field(
#         space1,
#         space1,
#         space2,
#         2.5 + 1j,
#         assembler="dense",
#         device_interface=device_interface,
#         precision=precision,
#         parameters=default_parameters,
#     ).weak_form()

#     if precision == "single":
#         rtol = 1e-5
#         atol = 1e-7
#     else:
#         rtol = 1e-10
#         atol = 1e-14

#     expected = helpers.load_npy_data("maxwell_electric_field_complex_boundary")
#     _np.testing.assert_allclose(discrete_op.A, expected, rtol=rtol, atol=atol)


# def test_maxwell_electric_field_screen(
#     default_parameters, helpers, device_interface, precision
# ):
#     """Test Maxwell electric field on sphere."""
#     from bempp.api import get_precision
#     from bempp.api import function_space
#     from bempp.api.operators.boundary.maxwell import electric_field

#     grid = helpers.load_grid("structured_grid")

#     space1 = function_space(grid, "RWG", 0)
#     space2 = function_space(grid, "SNC", 0)

#     discrete_op = electric_field(
#         space1,
#         space1,
#         space2,
#         2.5,
#         assembler="dense",
#         device_interface=device_interface,
#         precision=precision,
#         parameters=default_parameters,
#     ).weak_form()

#     if precision == "single":
#         rtol = 1e-5
#         atol = 1e-7
#     else:
#         rtol = 1e-10
#         atol = 1e-14

#     expected = helpers.load_npy_data("maxwell_electric_field_structured_boundary")
#     _np.testing.assert_allclose(discrete_op.A, expected, rtol=rtol, atol=atol)


def test_maxwell_magnetic_field_sphere(
    default_parameters, helpers, device_interface, precision
):
    """Test Maxwell magnetic field on sphere."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.maxwell import magnetic_field

    grid = helpers.load_grid("sphere")

    space1 = function_space(grid, "RWG", 0)
    space2 = function_space(grid, "SNC", 0)

    discrete_op = magnetic_field(
        space1,
        space1,
        space2,
        2.5,
        assembler="dense",
        device_interface=device_interface,
        precision=precision,
        parameters=default_parameters,
    ).weak_form()

    if precision == "single":
        rtol = 1e-5
        atol = 1e-7
    else:
        rtol = 1e-10
        atol = 1e-14

    expected = helpers.load_npy_data("maxwell_magnetic_field_boundary")

    _np.testing.assert_allclose(discrete_op.A, expected, rtol=rtol, atol=atol)


# def test_maxwell_magnetic_field_complex_sphere(
#     default_parameters, helpers, device_interface, precision
# ):
#     """Test Maxwell magnetic field on sphere with complex wavenumber."""
#     from bempp.api import get_precision
#     from bempp.api import function_space
#     from bempp.api.operators.boundary.maxwell import magnetic_field

#     grid = helpers.load_grid("sphere")

#     space1 = function_space(grid, "RWG", 0)
#     space2 = function_space(grid, "SNC", 0)

#     discrete_op = magnetic_field(
#         space1,
#         space1,
#         space2,
#         2.5 + 1j,
#         assembler="dense",
#         device_interface=device_interface,
#         precision=precision,
#         parameters=default_parameters,
#     ).weak_form()

#     if precision == "single":
#         rtol = 1e-5
#         atol = 1e-7
#     else:
#         rtol = 1e-10
#         atol = 1e-14

#     expected = helpers.load_npy_data("maxwell_magnetic_field_complex_boundary")
#     _np.testing.assert_allclose(discrete_op.A, expected, rtol=rtol, atol=atol)


# def test_maxwell_electric_field_sphere_evaluator(
#     default_parameters, helpers, device_interface, precision
# ):
#     """Test Maxwell electric field evaluator on sphere."""
#     from bempp.api import get_precision
#     from bempp.api import function_space
#     from bempp.api.operators.boundary.maxwell import electric_field

#     grid = helpers.load_grid("sphere")

#     space1 = function_space(grid, "RWG", 0)
#     space2 = function_space(grid, "SNC", 0)

#     discrete_op = electric_field(
#         space1,
#         space1,
#         space2,
#         2.5,
#         assembler="dense_evaluator",
#         device_interface=device_interface,
#         precision=precision,
#         parameters=default_parameters,
#     ).weak_form()

#     mat = electric_field(
#         space1,
#         space1,
#         space2,
#         2.5,
#         assembler="dense",
#         device_interface=device_interface,
#         precision=precision,
#         parameters=default_parameters,
#     ).weak_form()

#     x = _np.random.RandomState(0).randn(space1.global_dof_count)

#     actual = discrete_op @ x
#     expected = mat @ x

#     if precision == "single":
#         tol = 1e-4
#     else:
#         tol = 1e-12

#     _np.testing.assert_allclose(actual, expected, rtol=tol)


# def test_maxwell_magnetic_field_sphere_evaluator(
#     default_parameters, helpers, device_interface, precision
# ):
#     """Test Maxwell magnetic field evaluator on sphere."""
#     from bempp.api import get_precision
#     from bempp.api import function_space
#     from bempp.api.operators.boundary.maxwell import magnetic_field

#     grid = helpers.load_grid("sphere")

#     space1 = function_space(grid, "RWG", 0)
#     space2 = function_space(grid, "SNC", 0)

#     discrete_op = magnetic_field(
#         space1,
#         space1,
#         space2,
#         2.5,
#         assembler="dense_evaluator",
#         device_interface=device_interface,
#         precision=precision,
#         parameters=default_parameters,
#     ).weak_form()

#     mat = magnetic_field(
#         space1,
#         space1,
#         space2,
#         2.5,
#         assembler="dense",
#         device_interface=device_interface,
#         precision=precision,
#         parameters=default_parameters,
#     ).weak_form()

#     x = _np.random.RandomState(0).randn(space1.global_dof_count)

#     actual = discrete_op @ x
#     expected = mat @ x

#     if precision == "single":
#         tol = 1e-4
#     else:
#         tol = 1e-12

#     _np.testing.assert_allclose(actual, expected, rtol=tol)


# def test_maxwell_electric_field_complex_sphere_evaluator(
#     default_parameters, helpers, device_interface, precision
# ):
#     """Test Maxwell electric field evaluator on sphere with complex wavenumber."""
#     from bempp.api import get_precision
#     from bempp.api import function_space
#     from bempp.api.operators.boundary.maxwell import electric_field

#     grid = helpers.load_grid("sphere")

#     space1 = function_space(grid, "RWG", 0)
#     space2 = function_space(grid, "SNC", 0)

#     discrete_op = electric_field(
#         space1,
#         space1,
#         space2,
#         2.5 + 1j,
#         assembler="dense_evaluator",
#         device_interface=device_interface,
#         precision=precision,
#         parameters=default_parameters,
#     ).weak_form()

#     mat = electric_field(
#         space1,
#         space1,
#         space2,
#         2.5 + 1j,
#         assembler="dense",
#         device_interface=device_interface,
#         precision=precision,
#         parameters=default_parameters,
#     ).weak_form()

#     x = _np.random.RandomState(0).randn(space1.global_dof_count)

#     actual = discrete_op @ x
#     expected = mat @ x

#     if precision == "single":
#         tol = 1e-4
#     else:
#         tol = 1e-12

#     _np.testing.assert_allclose(actual, expected, rtol=tol)


# def test_maxwell_magnetic_field_complex_sphere_evaluator(
#     default_parameters, helpers, device_interface, precision
# ):
#     """Test Maxwell magnetic field evaluator on sphere with complex wavenumber."""
#     from bempp.api import get_precision
#     from bempp.api import function_space
#     from bempp.api.operators.boundary.maxwell import magnetic_field

#     grid = helpers.load_grid("sphere")

#     space1 = function_space(grid, "RWG", 0)
#     space2 = function_space(grid, "SNC", 0)

#     discrete_op = magnetic_field(
#         space1,
#         space1,
#         space2,
#         2.5 + 1j,
#         assembler="dense_evaluator",
#         device_interface=device_interface,
#         precision=precision,
#         parameters=default_parameters,
#     ).weak_form()

#     mat = magnetic_field(
#         space1,
#         space1,
#         space2,
#         2.5 + 1j,
#         assembler="dense",
#         device_interface=device_interface,
#         precision=precision,
#         parameters=default_parameters,
#     ).weak_form()

#     x = _np.random.RandomState(0).randn(space1.global_dof_count)

#     actual = discrete_op @ x
#     expected = mat @ x

#     if precision == "single":
#         tol = 1e-4
#     else:
#         tol = 1e-12

#     _np.testing.assert_allclose(actual, expected, rtol=tol)


# def test_maxwell_multitrace_sphere(
# default_parameters, helpers, device_interface, precision
# ):
# """Test Maxwell magnetic field on sphere."""
# from bempp.api import function_space
# from bempp.api.shapes import regular_sphere
# from bempp.api.operators.boundary.maxwell import electric_field, magnetic_field
# from bempp.api.assembly.blocked_operator import BlockedDiscreteOperator

# # if precision == 'single':
# #    pytest.skip("Test runs only in double precision mode.")

# grid = helpers.load_grid("sphere")

# op = bempp.api.operators.boundary.maxwell.multitrace_operator(
# grid,
# 2.5,
# parameters=default_parameters,
# assembler="multitrace_evaluator",
# device_interface=device_interface,
# precision=precision,
# )

# efield = electric_field(
# op.domain_spaces[0],
# op.range_spaces[0],
# op.dual_to_range_spaces[0],
# 2.5,
# parameters=default_parameters,
# assembler="dense",
# device_interface=device_interface,
# precision=precision,
# ).weak_form()

# mfield = magnetic_field(
# op.domain_spaces[0],
# op.range_spaces[0],
# op.dual_to_range_spaces[0],
# 2.5,
# parameters=default_parameters,
# assembler="dense",
# device_interface=device_interface,
# precision=precision,
# ).weak_form()

# expected = BlockedDiscreteOperator(_np.array([[mfield, efield], [-efield, mfield]]))

# rand = _np.random.RandomState(0)
# x = rand.randn(expected.shape[1])

# y_expected = expected @ x
# y_actual = op.weak_form() @ x

# _np.testing.assert_allclose(
# y_actual, y_expected, rtol=helpers.default_tolerance(precision)
# )


# def test_maxwell_transmission_sphere(
# default_parameters, helpers, device_interface, precision
# ):
# """Test Maxwell magnetic field on sphere."""
# from bempp.api import function_space
# from bempp.api.shapes import regular_sphere
# from bempp.api.operators.boundary.maxwell import electric_field, magnetic_field
# from bempp.api.assembly.blocked_operator import BlockedDiscreteOperator

# grid = helpers.load_grid("sphere")

# eps_rel = 1.3
# mu_rel = 1.5

# wavenumber = 2.5

# op = bempp.api.operators.boundary.maxwell.transmission_operator(
# grid,
# wavenumber,
# eps_rel,
# mu_rel,
# parameters=default_parameters,
# assembler="multitrace_evaluator",
# device_interface=device_interface,
# precision=precision,
# )

# domain = op.domain_spaces[0]
# range_ = op.domain_spaces[0]
# dual_to_range = op.dual_to_range_spaces[0]

# sqrt_eps_rel = _np.sqrt(eps_rel)
# sqrt_mu_rel = _np.sqrt(mu_rel)

# wavenumber_int = wavenumber * sqrt_eps_rel * sqrt_mu_rel

# magnetic_ext = magnetic_field(
# domain,
# range_,
# dual_to_range,
# wavenumber,
# default_parameters,
# "dense",
# device_interface,
# precision,
# ).weak_form()

# magnetic_int = magnetic_field(
# domain,
# range_,
# dual_to_range,
# wavenumber_int,
# default_parameters,
# "dense",
# device_interface,
# precision,
# ).weak_form()

# electric_ext = electric_field(
# domain,
# range_,
# dual_to_range,
# wavenumber,
# default_parameters,
# "dense",
# device_interface,
# precision,
# ).weak_form()

# electric_int = electric_field(
# domain,
# range_,
# dual_to_range,
# wavenumber_int,
# default_parameters,
# "dense",
# device_interface,
# precision,
# ).weak_form()

# fac = sqrt_mu_rel / sqrt_eps_rel

# expected = BlockedDiscreteOperator(
# _np.array(
# [
# [magnetic_ext + magnetic_int, electric_ext + fac * electric_int],
# [
# (-1.0 / fac) * electric_int - electric_ext,
# magnetic_int + magnetic_ext,
# ],
# ],
# dtype=_np.object,
# )
# )

# rand = _np.random.RandomState(0)
# x = rand.randn(expected.shape[1])

# y_expected = expected @ x
# y_actual = op.weak_form() @ x

# _np.testing.assert_allclose(
# y_actual, y_expected, rtol=helpers.default_tolerance(precision)
# )


# def test_maxwell_multitrace_complex_sphere(
# default_parameters, helpers, device_interface, precision
# ):
# """Test Maxwell multitrace on sphere with complex wavenumber."""
# from bempp.api import function_space
# from bempp.api.shapes import regular_sphere
# from bempp.api.operators.boundary.maxwell import electric_field, magnetic_field
# from bempp.api.assembly.blocked_operator import BlockedDiscreteOperator

# # if precision == 'single':
# #    pytest.skip("Test runs only in double precision mode.")

# grid = helpers.load_grid("sphere")

# op = bempp.api.operators.boundary.maxwell.multitrace_operator(
# grid,
# 2.5 + 1j,
# parameters=default_parameters,
# assembler="multitrace_evaluator",
# device_interface=device_interface,
# precision=precision,
# )

# efield = electric_field(
# op.domain_spaces[0],
# op.range_spaces[0],
# op.dual_to_range_spaces[0],
# 2.5 + 1j,
# parameters=default_parameters,
# assembler="dense",
# device_interface=device_interface,
# precision=precision,
# ).weak_form()

# mfield = magnetic_field(
# op.domain_spaces[0],
# op.range_spaces[0],
# op.dual_to_range_spaces[0],
# 2.5 + 1j,
# parameters=default_parameters,
# assembler="dense",
# device_interface=device_interface,
# precision=precision,
# ).weak_form()

# expected = BlockedDiscreteOperator(_np.array([[mfield, efield], [-efield, mfield]]))

# rand = _np.random.RandomState(0)
# x = rand.randn(expected.shape[1])

# y_expected = expected @ x
# y_actual = op.weak_form() @ x

# _np.testing.assert_allclose(
# y_actual, y_expected, rtol=helpers.default_tolerance(precision)
# )


# def test_maxwell_transmission_complex_sphere(
# default_parameters, helpers, device_interface, precision
# ):
# """Test Maxwell transmission operator on sphere with complex wavenumber.."""
# from bempp.api import function_space
# from bempp.api.shapes import regular_sphere
# from bempp.api.operators.boundary.maxwell import electric_field, magnetic_field
# from bempp.api.assembly.blocked_operator import BlockedDiscreteOperator

# grid = helpers.load_grid("sphere")

# eps_rel = 1.3
# mu_rel = 1.5

# wavenumber = 2.5 + 1j

# op = bempp.api.operators.boundary.maxwell.transmission_operator(
# grid,
# wavenumber,
# eps_rel,
# mu_rel,
# parameters=default_parameters,
# assembler="multitrace_evaluator",
# device_interface=device_interface,
# precision=precision,
# )

# domain = op.domain_spaces[0]
# range_ = op.domain_spaces[0]
# dual_to_range = op.dual_to_range_spaces[0]

# sqrt_eps_rel = _np.sqrt(eps_rel)
# sqrt_mu_rel = _np.sqrt(mu_rel)

# wavenumber_int = wavenumber * sqrt_eps_rel * sqrt_mu_rel

# magnetic_ext = magnetic_field(
# domain,
# range_,
# dual_to_range,
# wavenumber,
# default_parameters,
# "dense",
# device_interface,
# precision,
# ).weak_form()

# magnetic_int = magnetic_field(
# domain,
# range_,
# dual_to_range,
# wavenumber_int,
# default_parameters,
# "dense",
# device_interface,
# precision,
# ).weak_form()

# electric_ext = electric_field(
# domain,
# range_,
# dual_to_range,
# wavenumber,
# default_parameters,
# "dense",
# device_interface,
# precision,
# ).weak_form()

# electric_int = electric_field(
# domain,
# range_,
# dual_to_range,
# wavenumber_int,
# default_parameters,
# "dense",
# device_interface,
# precision,
# ).weak_form()

# fac = sqrt_mu_rel / sqrt_eps_rel

# expected = BlockedDiscreteOperator(
# _np.array(
# [
# [magnetic_ext + magnetic_int, electric_ext + fac * electric_int],
# [
# (-1.0 / fac) * electric_int - electric_ext,
# magnetic_int + magnetic_ext,
# ],
# ],
# dtype=_np.object,
# )
# )

# rand = _np.random.RandomState(0)
# x = rand.randn(expected.shape[1])

# y_expected = expected @ x
# y_actual = op.weak_form() @ x

# _np.testing.assert_allclose(
# y_actual, y_expected, rtol=helpers.default_tolerance(precision)
# )


# def test_maxwell_multitrace_subgrid(
# default_parameters, helpers, device_interface, precision
# ):
# """Test Maxwell multitrace operator on a subgrid."""
# from bempp.api import function_space
# from bempp.api.shapes import regular_sphere
# from bempp.api.operators.boundary.maxwell import electric_field, magnetic_field
# from bempp.api.assembly.blocked_operator import BlockedDiscreteOperator
# from bempp.api.grid.grid import grid_from_segments
# from bempp.api.shapes.shapes import multitrace_cube

# grid = bempp.api.shapes.multitrace_cube()

# seglists = [[1, 2, 3, 4, 5, 6], [6, 7, 8, 9, 10, 11]]
# swapped_normal_lists = [{}, {6}]

# rand = _np.random.RandomState(0)

# for seglist, swapped_normals in zip(seglists, swapped_normal_lists):
# new_grid = grid_from_segments(grid, seglist)
# rwg = bempp.api.function_space(
# new_grid,
# "RWG",
# 0,
# swapped_normals=swapped_normals,
# include_boundary_dofs=True,
# )
# snc = bempp.api.function_space(
# new_grid,
# "SNC",
# 0,
# swapped_normals=swapped_normals,
# include_boundary_dofs=True,
# )

# op = bempp.api.operators.boundary.maxwell.multitrace_operator(
# grid,
# 2.5,
# parameters=default_parameters,
# segments=seglist,
# assembler="multitrace_evaluator",
# swapped_normals=swapped_normals,
# device_interface=device_interface,
# precision=precision,
# )

# efield = electric_field(
# op.domain_spaces[0],
# op.range_spaces[0],
# op.dual_to_range_spaces[0],
# 2.5,
# parameters=default_parameters,
# assembler="dense_evaluator",
# device_interface=device_interface,
# precision=precision,
# ).weak_form()

# mfield = magnetic_field(
# op.domain_spaces[0],
# op.range_spaces[0],
# op.dual_to_range_spaces[0],
# 2.5,
# parameters=default_parameters,
# assembler="dense_evaluator",
# device_interface=device_interface,
# precision=precision,
# ).weak_form()

# expected = BlockedDiscreteOperator(
# _np.array([[mfield, efield], [-efield, mfield]])
# )

# rand = _np.random.RandomState(0)
# x = rand.randn(expected.shape[1])

# y_expected = expected @ x
# y_actual = op.weak_form() @ x

# _np.testing.assert_allclose(
# y_actual, y_expected, rtol=helpers.default_tolerance(precision)
# )


# def test_maxwell_operators_subgrid_evaluator(
# default_parameters, helpers, device_interface, precision
# ):
# """Test standard evaluators on subgrids of a skeleton with junctions."""
# import bempp.api
# from bempp.api import function_space
# from bempp.api.shapes import multitrace_cube
# from bempp.api.grid.grid import grid_from_segments

# grid = bempp.api.shapes.multitrace_cube()

# seglists = [[1, 2, 3, 4, 5, 6], [6, 7, 8, 9, 10, 11]]
# swapped_normal_lists = [{}, {6}]

# rand = _np.random.RandomState(0)

# operators = [
# bempp.api.operators.boundary.maxwell.electric_field,
# bempp.api.operators.boundary.maxwell.magnetic_field,
# ]

# for seglist, swapped_normals in zip(seglists, swapped_normal_lists):
# new_grid = grid_from_segments(grid, seglist)
# rwg1 = bempp.api.function_space(
# grid,
# "RWG",
# 0,
# segments=seglist,
# swapped_normals=swapped_normals,
# include_boundary_dofs=True,
# )
# snc1 = bempp.api.function_space(
# grid,
# "SNC",
# 0,
# segments=seglist,
# swapped_normals=swapped_normals,
# include_boundary_dofs=True,
# )
# rwg2 = bempp.api.function_space(
# new_grid,
# "RWG",
# 0,
# )
# snc2 = bempp.api.function_space(
# new_grid,
# "SNC",
# 0,
# swapped_normals=swapped_normals,
# )

# coeffs = rand.rand(rwg1.global_dof_count)

# for op in operators:
# res_actual = op(rwg1, rwg1, snc1, 2.5, assembler='dense_evaluator').weak_form() @ coeffs
# res_expected = op(rwg2, rwg2, snc2, 2.5, assembler='dense_evaluator').weak_form() @ coeffs
# _np.testing.assert_equal(res_actual, res_expected)
