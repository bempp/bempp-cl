"""Main interface class to ExaFMM."""
import numpy as _np


class Exafmm(object):
    """Evaluate integral operators via FMM."""

    def __init__(
        self, domain, dual_to_range, parameters, expansion_order=5, ncrit=400, depth=4
    ):
        from bempp.api.integration.triangle_gauss import rule
        from bempp.api.space.space import return_compatible_representation
        import bempp.api
        from .helpers import grid_to_points
        from .helpers import map_space_to_points
        from .helpers import get_local_interaction_matrix
        from .helpers import laplace_kernel
        import exafmm as fmm
        import exafmm.laplace as laplace

        order = parameters.quadrature.regular
        self._quad_points, self._quad_weights = rule(order)

        self._domain = domain
        self._dual_to_range = dual_to_range

        actual_domain, actual_dual_to_range = return_compatible_representation(
            self._domain, self._dual_to_range
        )

        source_points = grid_to_points(
            actual_domain.grid.data("double"), self._quad_points
        )
        target_points = grid_to_points(
            self._dual_to_range.grid.data("double"), self._quad_points
        )

        sources = laplace.init_sources(
            source_points, _np.zeros(len(source_points), dtype=_np.float64)
        )
        targets = laplace.init_targets(target_points)

        self._fmm = laplace.LaplaceFMM(expansion_order, ncrit, depth)

        skip_P2P = False
        self._tree = laplace.setup(sources, targets, self._fmm, skip_P2P)

        self._source_transform = map_space_to_points(
            actual_domain, self._quad_points, self._quad_weights
        )
        self._target_transform = map_space_to_points(
            actual_dual_to_range,
            self._quad_points,
            self._quad_weights,
            return_transpose=True,
        )

        self._singular_correction = get_local_interaction_matrix(
            actual_domain.grid, self._quad_points, laplace_kernel, [], "double", False
        )

        self._singular_part = (
            bempp.api.operators.boundary.laplace.single_layer(
                self._domain,
                self._dual_to_range,
                self._dual_to_range,
                assembler="only_singular_part",
            )
            .weak_form()
            .A
        )

    def _evaluate_exafmm(self, vec):
        """Apply ExaFMM."""
        import exafmm.laplace as laplace

        laplace.update_charges(self._tree, vec)
        laplace.clear_values(self._tree)
        return laplace.evaluate(self._tree, self._fmm)

    def _evaluate(self, vec):
        """Evaluate the operator."""
        import exafmm.laplace as laplace

        transformed_vec = self._source_transform @ vec
        potentials = self._evaluate_exafmm(transformed_vec)
        correction = (self._singular_correction @ transformed_vec).reshape([-1, 4])

        return (
            self._target_transform @ (potentials - correction)[:, 0]
            + self._singular_part @ vec
        )

    def fmm_as_matrix(self):
        """Return the matrix representation of the FMM."""

        npoints = len(self._quad_weights)

        shape = (npoints * self._dual_to_range.grid.number_of_elements,
                 npoints * self._domain.grid.number_of_elements)

        mat = _np.empty(shape, dtype="float64")
        ident = _np.eye(shape[1], dtype='float64')
    
        for index in range(shape[1]):
            mat[:, index] = self._evaluate_exafmm(ident[:, index])[:, 0]

        return mat

    def aslinearoperator(self):
        """
        Return a Scipy Linear Operator that evaluates the FMM.

        The returned class should subclass the Scipy LinearOperator class
        so that it provides a matvec routine that accept a vector of coefficients
        and returns the result of a matrix vector product.
        """
        from scipy.sparse.linalg import LinearOperator

        return LinearOperator(
            (self._dual_to_range.global_dof_count, self._domain.global_dof_count),
            matvec=self._evaluate,
            dtype=_np.float64,
        )
