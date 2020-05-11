import numpy as _np

from bempp.api.assembly import assembler as _assembler
from bempp.helpers import timeit as _timeit


class DiagonalAssembler(_assembler.AssemblerBase):
    """Implementation of a diagonal assembler for integral operators."""

    # pylint: disable=useless-super-delegation
    def __init__(self, domain, dual_to_range, parameters=None):
        """Create a diagonal assembler instance."""
        super().__init__(domain, dual_to_range, parameters)

    def assemble(
        self, operator_descriptor, device_interface, precision, *args, **kwargs
    ):
        """Diagonal assembly of the integral operator."""
        from bempp.api.assembly.discrete_boundary_operator import DiagonalOperator
        from bempp.api.utils.helpers import promote_to_double_precision

        if (
            self.domain.requires_dof_transformation
            or self.dual_to_range.requires_dof_transformation
        ):
            raise ValueError(
                "Spaces that require dof transformations not supported for diagonal assembly."
            )

        if self.domain != self.dual_to_range:
            raise ValueError(
                "Only identical spaces currently supported for diagonal assembly."
            )

        if (
            self.domain.identifier == self.dual_to_range.identifier
            and self.dual_to_range.identifier == "p1_continuous"
        ) or (
            self.domain.identifier == self.dual_to_range.identifier
            and self.dual_to_range.identifier == "p0_discontinuous"
        ):
            values = operator_descriptor.singular_part.weak_form().A.diagonal()
        else:
            raise ValueError(
                "Only spaces of type 'p0_discontinuous' or 'p1_continuous' supported for diagonal assembly."
            )

        return DiagonalOperator(
            values,
            shape=(self.dual_to_range.global_dof_count, self.domain.global_dof_count),
        )

        # values = assemble_diagonal(
        # self.domain,
        # self.dual_to_range,
        # self.parameters,
        # operator_descriptor,
        # device_interface,
        # )

        # if self.parameters.assembly.always_promote_to_double:
        # values = promote_to_double_precision(values)

        # return DiagonalOperator(
        # values,
        # shape=(self.dual_to_range.global_dof_count, self.domain.global_dof_count),
        # )


# def assemble_diagonal(
# domain, dual_to_range, parameters, operator_descriptor, device_interface
# ):
# """Assembles the diagonal of the operator."""
# import bempp.api
# from bempp.api.utils.helpers import get_type
# from bempp.core.singular_assembler import assemble_singular_part
# from bempp.api.integration.triangle_gauss import rule
# from bempp.core.numba_kernels import select_numba_kernels

# precision = operator_descriptor.precision

# rows = dual_to_range.global_dof_count
# cols = domain.global_dof_count

# nvalues = _np.min([rows, cols])

# if operator_descriptor.is_complex:
# result_type = "complex128"
# else:
# result_type = "float64"

# result = _np.zeros((1, nvalues), dtype=result_type)

# quad_points, quad_weights = rule(parameters.quadrature.regular)

# (numba_assembly_function, numba_kernel_function) = select_numba_kernels(
# operator_descriptor, mode="regular"
# )

# test_grid_data = dual_to_range.grid.data("double")
# trial_grid_data = domain.grid.data("double")
# kernel_parameters = _np.array(operator_descriptor.options, dtype="float64")

# grids_identical = dual_to_range.grid == domain.grid

# with bempp.api.Timer(
# message=f"Diagonal assembler:{operator_descriptor.identifier}:{device_interface}"
# ):
# # We need to copy them as we adapt the local2global arrays in each step.
# local2global_test = _np.zeros_like(dual_to_range.local2global)
# local2global_trial = _np.zeros_like(domain.local2global)
# multipliers_test = _np.zeros_like(dual_to_range.local_multipliers)
# multipliers_trial = _np.zeros_like(domain.local_multipliers)

# for global_dof_index in range(nvalues):
# # Go through each global dof and assemble the associated local dofs.
# test_support = []
# trial_support = []
# dofs_test = dual_to_range.global2local[global_dof_index]
# dofs_trial = domain.global2local[global_dof_index]
# # The following shifts the global dofs to sum into (0, global_dof_index)
# for test_elem, dof in dofs_test:
# test_support.append(test_elem)
# local2global_test[test_elem] = 0
# multipliers_test[test_elem][:] = 0
# multipliers_test[test_elem][dof] = dual_to_range.local2global[
# test_elem
# ][dof]
# for trial_elem, dof in dofs_trial:
# trial_support.append(trial_elem)
# local2global_trial[trial_elem][:] = 0
# local2global_trial[trial_elem][dof] = global_dof_index
# multipliers_trial[trial_elem][:] = 0
# multipliers_trial[trial_elem][dof] = domain.local2global[
# trial_elem
# ][dof]
# test_support = _np.array(test_support, dtype="uint32")
# trial_support = _np.array(trial_support, dtype="uint32")

# numba_assembly_function(
# test_grid_data,
# trial_grid_data,
# dual_to_range.number_of_shape_functions,
# domain.number_of_shape_functions,
# test_support,
# trial_support,
# multipliers_test,
# multipliers_trial,
# local2global_test,
# local2global_trial,
# dual_to_range.normal_multipliers,
# domain.normal_multipliers,
# quad_points,
# quad_weights,
# numba_kernel_function,
# kernel_parameters,
# grids_identical,
# dual_to_range.shapeset.evaluate,
# domain.shapeset.evaluate,
# result,
# )

# if grids_identical:
# singular_part = (
# operator_descriptor.singular_part.weak_form().A.diagonal()
# )
# result += singular_part

# return result


# @_timeit
# def assemble_dense(
# domain,
# dual_to_range,
# parameters,
# operator_descriptor,
# numba_assembly_function_regular,
# numba_kernel_function_regular,
# numba_assembly_function_singular,
# numba_kernel_function_singular,
# ):
# """
# Really assemble the operator.
# Assembles the complete operator (near-field and far-field)
# Returns a dense matrix.
# """
# from bempp.api.integration.triangle_gauss import rule as regular_rule
# from bempp.api import log
# from bempp.core.numba.singular_assembler import assemble_singular_part
# from bempp.api.utils.helpers import get_type
# import bempp.api

# order = parameters.quadrature.regular
# quad_points, quad_weights = regular_rule(order)

# test_indices, test_color_indexptr = dual_to_range.get_elements_by_color()
# trial_indices, trial_color_indexptr = domain.get_elements_by_color()
# number_of_test_colors = len(test_color_indexptr) - 1
# number_of_trial_colors = len(trial_color_indexptr) - 1

# rows = dual_to_range.global_dof_count
# cols = domain.global_dof_count

# nshape_test = dual_to_range.number_of_shape_functions
# nshape_trial = domain.number_of_shape_functions

# precision = operator_descriptor.precision

# data_type = get_type(precision).real
# if operator_descriptor.is_complex:
# result_type = get_type(precision).complex
# else:
# result_type = get_type(precision).real

# result = _np.zeros((rows, cols), dtype=result_type)

# grids_identical = domain.grid == dual_to_range.grid

# dense_assembler_dispatcher(
# device_interface,
# operator_descriptor,
# domain,
# dual_to_range,


# with bempp.api.Timer(message="Start Numba dense assembler."):
# for test_color_index in range(number_of_test_colors):
# numba_assembly_function_regular(
# dual_to_range.grid.data(precision),
# domain.grid.data(precision),
# nshape_test,
# nshape_trial,
# test_indices[
# test_color_indexptr[test_color_index] : test_color_indexptr[
# 1 + test_color_index
# ]
# ],
# trial_indices,
# dual_to_range.local_multipliers.astype(data_type),
# domain.local_multipliers.astype(data_type),
# dual_to_range.local2global,
# domain.local2global,
# dual_to_range.normal_multipliers,
# domain.normal_multipliers,
# quad_points.astype(data_type),
# quad_weights.astype(data_type),
# numba_kernel_function_regular,
# _np.array(operator_descriptor.options, dtype=data_type),
# grids_identical,
# dual_to_range.shapeset.evaluate,
# domain.shapeset.evaluate,
# result,
# )

# if grids_identical:
# # Need to treat singular contribution

# trial_local2global = domain.local2global.ravel()
# test_local2global = dual_to_range.local2global.ravel()
# trial_multipliers = domain.local_multipliers.ravel()
# test_multipliers = dual_to_range.local_multipliers.ravel()

# singular_rows, singular_cols, singular_values = assemble_singular_part(
# domain.localised_space,
# dual_to_range.localised_space,
# parameters,
# operator_descriptor,
# numba_assembly_function_singular,
# numba_kernel_function_singular,
# )

# rows = test_local2global[singular_rows]
# cols = trial_local2global[singular_cols]
# values = (
# singular_values
# * trial_multipliers[singular_cols]
# * test_multipliers[singular_rows]
# )

# _np.add.at(result, (rows, cols), values)

# return result
