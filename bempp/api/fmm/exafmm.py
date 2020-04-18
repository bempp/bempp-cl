"""Main interface class to ExaFMM."""
import numpy as _np


class ExafmmInstance(object):
    """Evaluate an Fmm instance."""

    def __init__(
        self,
        source_points,
        target_points,
        quadrature_order,
        mode,
        kernel_parameters,
        depth=4,
        expansion_order=5,
        ncrit=400,
        precision="double",
    ):
        """Instantiate an Exafmm session."""

        with bempp.api.Timer(message="Initialising Exafmm."):

            if mode == "laplace":
                import exafmm.laplace

                self._module = exafmm.laplace

                sources = exafmm.laplace.init_sources(
                    source_points, _np.zeros(len(source_points), dtype=_np.float64)
                )

                targets = exafmm.laplace.init_targets(target_points)

                self._fmm = exafmm.laplace.LaplaceFMM(expansion_order, ncrit, depth)
                self._tree = exafmm.laplace.setup(sources, targets, self._fmm, False)

            elif mode == "helmholtz":
                import exafmm.helmholtz

                self._module = exafmm.laplace

                sources = exafmm.helmholtz.init_sources(
                    source_points, _np.zeros(len(source_points), dtype=_np.float64)
                )

                targets = exafmm.helmholtz.init_targets(target_points)

                self._fmm = exafmm.helmholtz.HelmholtzFMM(
                    expansion_order, ncrit, depth, kernel_parameters[0]
                )
                self._tree = exafmm.helmholtz.setup(sources, targets, self._fmm, False)

            elif mode == "modified_helmholtz":
                import exafmm.modified_helmholtz

                self._module = exafmm.modified_helmholtz

                sources = exafmm.modified_helmholtz.init_sources(
                    source_points, _np.zeros(len(source_points), dtype=_np.float64)
                )

                targets = exafmm.modified_helmholtz.init_targets(target_points)

                self._fmm = exafmm.modified_helmholtz.ModifiedHelmholtzFMM(
                    expansion_order, ncrit, depth, kernel_parameters[0]
                )
                self._tree = exafmm.modified_helmholtz.setup(
                    sources, targets, self._fmm, False
                )


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

        with bempp.api.Timer(message="Generating FMM points."):
            source_points = grid_to_points(
                actual_domain.grid.data("double"), self._quad_points
            )
            target_points = grid_to_points(
                self._dual_to_range.grid.data("double"), self._quad_points
            )

        with bempp.api.Timer(message="Initialising Exafmm."):
            sources = laplace.init_sources(
                source_points, _np.zeros(len(source_points), dtype=_np.float64)
            )

            targets = laplace.init_targets(target_points)

            self._fmm = laplace.LaplaceFMM(expansion_order, ncrit, depth)

            skip_P2P = False
            self._tree = laplace.setup(sources, targets, self._fmm, skip_P2P)

        with bempp.api.Timer(message="Generate map from space to point evaluations."):
            self._source_transform = map_space_to_points(
                actual_domain, self._quad_points, self._quad_weights
            )
            self._target_transform = map_space_to_points(
                actual_dual_to_range,
                self._quad_points,
                self._quad_weights,
                return_transpose=True,
            )

        with bempp.api.Timer(message="Computing singular correction."):
            self._singular_correction = get_local_interaction_matrix(
                actual_domain.grid,
                self._quad_points,
                laplace_kernel,
                [],
                "double",
                False,
            )

        with bempp.api.Timer(message="Computing singular integral contributions."):
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
        import bempp.api

        with bempp.api.Timer(message="Performing Fmm matvec."):
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

        shape = (
            npoints * self._dual_to_range.grid.number_of_elements,
            npoints * self._domain.grid.number_of_elements,
        )

        mat = _np.empty(shape, dtype="float64")
        ident = _np.eye(shape[1], dtype="float64")

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
