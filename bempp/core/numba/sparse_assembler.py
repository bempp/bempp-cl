import numpy as _np

from bempp.api.assembly import assembler as _assembler
from bempp.helpers import timeit as _timeit


class SparseAssembler(_assembler.AssemblerBase):
    """Implementation of a sparse assembler."""

    # pylint: disable=useless-super-delegation
    def __init__(self, domain, dual_to_range, parameters=None):
        """Create a dense assembler instance."""
        super().__init__(domain, dual_to_range, parameters)

    def assemble(
        self, operator_descriptor, device_interface, precision, *args, **kwargs
    ):
        """Sparse assembly of operators."""
        from bempp.api.utils.helpers import promote_to_double_precision
        from scipy.sparse import coo_matrix, csr_matrix
        from bempp.api.space.space import return_compatible_representation
        from .kernels import select_numba_kernels

        domain, dual_to_range = return_compatible_representation(
            self.domain, self.dual_to_range
        )
        row_dof_count = dual_to_range.global_dof_count
        col_dof_count = domain.global_dof_count
        row_grid_dofs = dual_to_range.grid_dof_count
        col_grid_dofs = domain.grid_dof_count

        if domain.grid != dual_to_range.grid:
            raise ValueError("For sparse operators the domain and dual_to_range grids must be identical.")

        trial_local2global = domain.local2global.ravel()
        test_local2global = dual_to_range.local2global.ravel()
        trial_multipliers = domain.local_multipliers.ravel()
        test_multipliers = dual_to_range.local_multipliers.ravel()

        numba_assembly_function, numba_kernel_function = select_numba_kernels(
            operator_descriptor, mode="sparse"
        )

        rows, cols, values = assemble_sparse(
            domain.localised_space,
            dual_to_range.localised_space,
            self.parameters,
            numba_assembly_function,
            numba_kernel_function,
            precision,
            operator_descriptor.options,
        )
        global_rows = test_local2global[rows]
        global_cols = trial_local2global[cols]
        global_values = values * trial_multipliers[cols] * test_multipliers[rows]

        if self.parameters.assembly.always_promote_to_double:
            values = promote_to_double_precision(values)

        mat = coo_matrix(
            (global_values, (global_rows, global_cols)),
            shape=(row_grid_dofs, col_grid_dofs),
        ).tocsr()

        if domain.requires_dof_transformation:
            mat = mat @ domain.dof_transformation

        if dual_to_range.requires_dof_transformation:
            mat = dual_to_range.dof_transformation.T @ mat

        return SparseDiscreteBoundaryOperator(mat)


@_timeit
def assemble_sparse(
    domain,
    dual_to_range,
    parameters,
    operator_descriptor,
    numba_assembly_function_regular,
    numba_kernel_function_regular,
    numba_assembly_function_singular,
    numba_kernel_function_singular,
):
    """
    Really assemble the operator.
    """
    import bempp.api
    from bempp.api.integration.triangle_gauss import rule as regular_rule
    from bempp.api import log
    from bempp.api.utils.helpers import get_type

    order = parameters.quadrature.regular
    quad_points, quad_weights = regular_rule(order)

    test_indices, test_color_indexptr = dual_to_range.get_elements_by_color()
    trial_indices, trial_color_indexptr = domain.get_elements_by_color()
    number_of_test_colors = len(test_color_indexptr) - 1
    number_of_trial_colors = len(trial_color_indexptr) - 1

    rows = dual_to_range.global_dof_count
    cols = domain.global_dof_count

    nshape_test = dual_to_range.number_of_shape_functions
    nshape_trial = domain.number_of_shape_functions

    precision = operator_descriptor.precision

    data_type = get_type(precision).real
    if operator_descriptor.is_complex:
        result_type = get_type(precision).complex
    else:
        result_type = get_type(precision).real

    result = _np.zeros((rows, cols), dtype=result_type)

    grids_identical = domain.grid == dual_to_range.grid

    with bempp.api.Timer() as t:
        for test_color_index in range(number_of_test_colors):
            numba_assembly_function_regular(
                dual_to_range.grid.data(precision),
                domain.grid.data(precision),
                nshape_test,
                nshape_trial,
                test_indices[
                    test_color_indexptr[test_color_index] : test_color_indexptr[
                        1 + test_color_index
                    ]
                ],
                trial_indices,
                dual_to_range.local_multipliers.astype(data_type),
                domain.local_multipliers.astype(data_type),
                dual_to_range.local2global,
                domain.local2global,
                dual_to_range.normal_multipliers,
                domain.normal_multipliers,
                quad_points.astype(data_type),
                quad_weights.astype(data_type),
                numba_kernel_function_regular,
                _np.array(operator_descriptor.options, dtype=data_type),
                grids_identical,
                dual_to_range.shapeset.evaluate,
                domain.shapeset.evaluate,
                result,
            )
    print(f"Numba kernel time: {t.interval}")


    # with bempp.api.Timer() as t:
        # for test_color_index in range(number_of_test_colors):
            # for trial_color_index in range(number_of_trial_colors):
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
                    # trial_indices[
                        # trial_color_indexptr[trial_color_index] : trial_color_indexptr[
                            # 1 + trial_color_index
                        # ]
                    # ],
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
    # print(f"Numba kernel time: {t.interval}")

    if grids_identical:
        # Need to treat singular contribution

        trial_local2global = domain.local2global.ravel()
        test_local2global = dual_to_range.local2global.ravel()
        trial_multipliers = domain.local_multipliers.ravel()
        test_multipliers = dual_to_range.local_multipliers.ravel()

        singular_rows, singular_cols, singular_values = assemble_singular_part(
            domain.localised_space,
            dual_to_range.localised_space,
            parameters,
            numba_assembly_function_singular,
            numba_kernel_function_singular,
            precision,
            operator_descriptor.options,
        )

        rows = test_local2global[singular_rows]
        cols = trial_local2global[singular_cols]
        values = (
            singular_values
            * trial_multipliers[singular_cols]
            * test_multipliers[singular_rows]
        )

        _np.add.at(result, (rows, cols), values)

    return result
