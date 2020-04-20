"""Main interface class to ExaFMM."""
import numpy as _np


class ExafmmInterface(object):
    """Interface to Exafmm."""

    def __init__(
        self,
        source_points,
        target_points,
        mode,
        wavenumber=None,
        depth=4,
        expansion_order=5,
        ncrit=400,
        precision="double",
        singular_correction=None,
        source_normals=None,
        target_normals=None,
    ):
        """Instantiate an Exafmm session."""
        import bempp.api

        self._singular_correction = singular_correction
        self._source_normals = source_normals
        self._target_normals = target_normals

        self._source_points = source_points
        self._target_points = target_points

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
                    expansion_order, ncrit, depth, wavenumber
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
                    expansion_order, ncrit, depth, wavenumber
                )
                self._tree = exafmm.modified_helmholtz.setup(
                    sources, targets, self._fmm, False
                )

    @property
    def number_of_source_points(self):
        """Return number of source points."""
        return len(self._source_points)

    @property
    def number_of_target_points(self):
        """Return number of target points."""
        return len(self._target_points)

    def evaluate(
        self, vec, return_mode="function_values", apply_singular_correction=True
    ):
        """Evalute the Fmm."""
        import bempp.api

        with bempp.api.Timer(message="Evaluating Fmm."):
            self._module.update_charges(self._tree, vec)
            self._module.clear_values(self._tree)

            result = self._module.evaluate(self._tree, self._fmm)

            if apply_singular_correction and self._singular_correction is not None:
                result -= (self._singular_correction @ vec).reshape([-1, 4])

            if return_mode == "function_values":
                return result[:, 0]
            elif return_mode == "target_gradient":
                return result[:, 1:]
            elif return_mode == "source_gradient":
                return -result[:, 1:]
            elif return_mode == "target_normal_derivative":
                return np.sum(self._target_normals * result[:, 1:], axis=1)
            elif return_mode == "source_normal_derivative":
                return np.sum(-self._source_normals * result[:, 1:], axis=1)

    @classmethod
    def from_grid(
        cls,
        source_grid,
        mode,
        wavenumber=None,
        quadrature_order=None,
        target_grid=None,
        depth=4,
        expansion_order=5,
        ncrit=400,
        precision="double",
    ):
        """
        Initialise an Exafmm instance from a given source and target grid.

        Parameters
        ----------
        source_grid : Grid object
            Grid for the source points.
        mode: string
            Fmm mode. One of 'laplace', 'helmholtz', or 'modified_helmholtz'
        wavenumber : real number
            For Helmholtz or modified Helmholtz the wavenumber.
        quadrature_order : integer
            Quadrature order for converting grid to Fmm evaluation points.
            If None is specified, use the value provided by
            bempp.api.GLOBAL_PARAMETERS.quadrature.regular
        target_grid : Grid object
            An optional target grid. If not provided the source and target
            grid are assumed to be identical.
        depth: integer
            Depth of the Fmm tree.
        expansion_order : integer
            Expansion order for the Fmm.
        ncrit : integer
            Maximum number of leaf points per box (currently not used).
        precision : string
            Either 'single' or 'double'. Currently, the Fmm is always
            executed in double precision.
        """
        import bempp.api
        from bempp.api.integration.triangle_gauss import rule
        from bempp.api.fmm.helpers import get_local_interaction_matrix
        import numpy as np

        if quadrature_order is None:
            quadrature_order = bempp.api.GLOBAL_PARAMETERS.quadrature.regular

        local_points, weights = rule(quadrature_order)
        npoints = len(weights)

        if target_grid is None:
            target_grid = source_grid

        source_points = source_grid.map_to_point_cloud(
            quadrature_order, precision=precision
        )

        # Compute source normals

        source_normals = np.empty(
            (npoints * source_grid.number_of_elements, 3), dtype="float64"
        )
        for element in range(source_grid.number_of_elements):
            for n in range(npoints):
                source_normals[npoints * element + n, :] = source_grid.normals[element]

        if target_grid != source_grid:
            target_points = target_grid.map_to_point_cloud(
                quadrature_order, precision=precision
            )
            target_normals = np.empty(
                (npoints * target_grid.number_of_elements, 3), dtype="float64"
            )
            for element in range(target_grid.number_of_elements):
                for n in range(npoints):
                    target_normals[npoints * element + n, :] = target_grid.normals[
                        element
                    ]
        else:
            target_points = source_points
            target_normals = source_normals

        singular_correction = None

        if target_grid == source_grid:
            # Require singular correction terms.

            if mode == "laplace":
                from bempp.api.fmm.helpers import laplace_kernel

                singular_correction = get_local_interaction_matrix(
                    source_grid,
                    local_points,
                    laplace_kernel,
                    np.array([], dtype="float64"),
                    precision,
                    False,
                )
            elif mode == "helmholtz":
                from bempp.api.fmm.helpers import helmholtz_kernel

                singular_correction = get_local_interaction_matrix(
                    source_grid,
                    local_points,
                    helmholtz_kernel,
                    np.array([wavenumber], dtype="float64"),
                    precision,
                    False,
                )
            elif mode == "modified_helmholtz":
                from bempp.api.fmm.helpers import modified_helmholtz_kernel

                singular_correction = get_local_interaction_matrix(
                    source_grid,
                    local_points,
                    modified_helmholtz_kernel,
                    np.array([wavenumber], dtype="float64"),
                    precision,
                    False,
                )
        return cls(
            source_points,
            target_points,
            mode,
            wavenumber=wavenumber,
            depth=depth,
            expansion_order=expansion_order,
            ncrit=ncrit,
            precision=precision,
            singular_correction=singular_correction,
            source_normals=source_normals,
            target_normals=target_normals,
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
