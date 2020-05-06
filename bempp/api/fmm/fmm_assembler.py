"""Implementation of an FMM Assembler."""
from bempp.api.assembly import assembler as _assembler
import numba as _numba
import numpy as _np

_FMM_CACHE = {}
_FMM_POTENTIAL_CACHE = {}


def get_mode_from_operator_identifier(identifier):
    """Get the Fmm mode from the operator identifier."""

    descriptor = identifier.split("_")[0]

    if descriptor == "laplace":
        return "laplace"
    elif descriptor == "helmholtz":
        return "helmholtz"
    elif descriptor == "modified":
        return "modified_helmholtz"
    else:
        raise ValueError("Unknown identifier string.")


def get_fmm_interface(domain, dual_to_range, mode, wavenumber):
    """Get an Fmm instance."""
    import bempp.api

    global _FMM_CACHE

    key = (domain.grid.id, dual_to_range.grid.id, mode, wavenumber)

    interface = _FMM_CACHE.get(key, None)

    if interface is None:
        from bempp.api.fmm.exafmm import ExafmmInterface

        interface = ExafmmInterface.from_grid(
            domain.grid, mode, wavenumber=wavenumber, target_grid=dual_to_range.grid
        )
        _FMM_CACHE[key] = interface
    else:
        bempp.api.log("Using cached Fmm Interface.", level="debug")

    return interface


def get_fmm_potential_interface(space, points, mode, wavenumber):
    import bempp.api

    global _FMM_POTENTIAL_CACHE

    points_hash = hash(points.data.tobytes())

    key = (space.grid.id, points_hash, mode, wavenumber)

    interface = _FMM_POTENTIAL_CACHE.get(key, None)

    if interface is None:
        from bempp.api.fmm.exafmm import ExafmmInterface

        quadrature_order = bempp.api.GLOBAL_PARAMETERS.quadrature.regular

        interface = ExafmmInterface(
            space.grid.map_to_point_cloud(quadrature_order, precision="double"),
            points.T,
            mode,
            wavenumber,
            bempp.api.GLOBAL_PARAMETERS.fmm.depth,
            bempp.api.GLOBAL_PARAMETERS.fmm.expansion_order,
            bempp.api.GLOBAL_PARAMETERS.fmm.ncrit,
        )
        _FMM_POTENTIAL_CACHE[key] = interface
    else:
        bempp.api.log("Using cached Fmm Interface.", level="debug")

    return interface


def create_evaluator(
    operator_descriptor, fmm_interface, domain, dual_to_range, parameters
):
    """Return an Fmm evaluator for the requested kernel."""

    if operator_descriptor.assembly_type == "default_scalar":
        return make_default_scalar(
            operator_descriptor, fmm_interface, domain, dual_to_range
        )
    if operator_descriptor.assembly_type.split("_")[-1] == "hypersingular":
        return make_scalar_hypersingular(
            operator_descriptor, fmm_interface, domain, dual_to_range
        )


def create_potential_evaluator(operator_descriptor, fmm_interface, space, parameters):
    """Select an Fmm Potential Evaluator."""

    if operator_descriptor.assembly_type == "default_scalar":
        return make_default_scalar_potential(operator_descriptor, fmm_interface, space)
    else:
        raise ValueError("Unknown descriptor.")


class FmmPotentialAssembler(object):
    """Potential assembler for FMM."""

    def __init__(
        self, space, operator_descriptor, points, device_interface, parameters
    ):
        """Initialise FMM Potential Assembler."""

        mode = get_mode_from_operator_identifier(operator_descriptor.identifier)

        if mode == "laplace":
            wavenumber = None
        else:
            wavenumber = operator_descriptor.options[0]

        fmm_potential_interface = get_fmm_potential_interface(
            space, points, mode, wavenumber
        )
        self._evaluator = create_potential_evaluator(
            operator_descriptor, fmm_potential_interface, space, parameters
        )

    def evaluate(self, x):
        """Actually evaluate the potential."""
        return self._evaluator(x)


class FmmAssembler(_assembler.AssemblerBase):
    """Assembler for Fmm."""

    # pylint: disable=useless-super-delegation
    def __init__(self, domain, dual_to_range, parameters):
        """Create an Fmm assembler instance."""
        super().__init__(domain, dual_to_range, parameters)

        # self._source_map = domain.map_to_points(parameters.quadrature.regular)
        # self._target_map = dual_to_range.map_to_points(
        # parameters.quadrature.regular, return_transpose=True
        # )

        self.dtype = None
        self._evaluator = None
        self.shape = (dual_to_range.global_dof_count, domain.global_dof_count)

    def assemble(
        self, operator_descriptor, device_interface, precision, *args, **kwargs
    ):
        """Actually assemble."""
        from bempp.api.space.space import return_compatible_representation
        from bempp.api.assembly.discrete_boundary_operator import (
            GenericDiscreteBoundaryOperator,
        )

        actual_domain, actual_dual_to_range = return_compatible_representation(
            self.domain, self.dual_to_range
        )

        mode = get_mode_from_operator_identifier(operator_descriptor.identifier)
        if mode == "laplace":
            wavenumber = None
        else:
            wavenumber = operator_descriptor.options[0]

        fmm_interface = get_fmm_interface(
            actual_domain, actual_dual_to_range, mode, wavenumber
        )

        self._evaluator = create_evaluator(
            operator_descriptor,
            fmm_interface,
            actual_domain,
            actual_dual_to_range,
            self.parameters,
        )

        if operator_descriptor.is_complex:
            self.dtype = "complex128"
        else:
            self.dtype = "float64"

        return GenericDiscreteBoundaryOperator(self)

    def matvec(self, x):
        """Perform a matvec."""

        ndim = len(x.shape)

        if ndim > 2:
            raise ValueError(
                "x must have shape (N, ) or (N, 1), where N is number of elements."
            )

        result = self._evaluator(x.ravel())

        if ndim == 1:
            return result
        else:
            return result.reshape([-1, 1])


def make_scalar_hypersingular(
    operator_descriptor, fmm_interface, domain, dual_to_range
):
    """Create an evaluator for scalar hypersingular operators."""
    import bempp.api
    from bempp.api.integration.triangle_gauss import get_number_of_quad_points

    npoints = get_number_of_quad_points(bempp.api.GLOBAL_PARAMETERS.quadrature.regular)

    source_map = domain.map_to_points(bempp.api.GLOBAL_PARAMETERS.quadrature.regular)
    target_map = dual_to_range.map_to_points(
        bempp.api.GLOBAL_PARAMETERS.quadrature.regular, return_transpose=True
    )

    singular_part = operator_descriptor.singular_part.weak_form().A

    source_normals = get_normals(domain, npoints)
    target_normals = get_normals(dual_to_range, npoints)

    source_curls, source_curls_trans = compute_p1_curl_transformation(
        domain, bempp.api.GLOBAL_PARAMETERS.quadrature.regular
    )

    if dual_to_range == domain:
        target_curls, target_curls_trans = source_curls, source_curls_trans
    else:
        target_curls, target_curls_trans = compute_p1_curl_transformation(
            dual_to_range.bempp.api.GLOBAL_PARAMETERS.quadrature_order
        )

    def evaluate_laplace_hypersingular(x):
        """Evaluate the Laplace hypersingular kernel."""

        fmm_res0 = (
            target_curls_trans[0] @ fmm_interface.evaluate(source_curls[0] @ x)[:, 0]
        )
        fmm_res1 = (
            target_curls_trans[1] @ fmm_interface.evaluate(source_curls[1] @ x)[:, 0]
        )
        fmm_res2 = (
            target_curls_trans[2] @ fmm_interface.evaluate(source_curls[2] @ x)[:, 0]
        )

        return fmm_res0 + fmm_res1 + fmm_res2 + singular_part @ x

    def evaluate_helmholtz_hypersingular(x):
        """Evaluate the Helmholtz hypersingular kernel."""

        wavenumber = operator_descriptor.options[0]
        x_transformed = source_map @ x

        fmm_res0 = (
            target_curls_trans[0] @ fmm_interface.evaluate(source_curls[0] @ x)[:, 0]
        )
        fmm_res1 = (
            target_curls_trans[1] @ fmm_interface.evaluate(source_curls[1] @ x)[:, 0]
        )
        fmm_res2 = (
            target_curls_trans[2] @ fmm_interface.evaluate(source_curls[2] @ x)[:, 0]
        )

        first_part = fmm_res0 + fmm_res1 + fmm_res2

        fmm_n1 = (
            target_normals[:, 0]
            * fmm_interface.evaluate(source_normals[:, 0] * x_transformed)[:, 0]
        )
        fmm_n2 = (
            target_normals[:, 1]
            * fmm_interface.evaluate(source_normals[:, 1] * x_transformed)[:, 0]
        )
        fmm_n3 = (
            target_normals[:, 2]
            * fmm_interface.evaluate(source_normals[:, 2] * x_transformed)[:, 0]
        )

        second_part = target_map @ (fmm_n1 + fmm_n2 + fmm_n3)

        return first_part - wavenumber * wavenumber * second_part + singular_part @ x

    def evaluate_modified_helmholtz_hypersingular(x):
        """Evaluate the modified Helmholtz hypersingular kernel."""

        wavenumber = operator_descriptor.options[0]
        x_transformed = source_map @ x

        fmm_res0 = (
            target_curls_trans[0] @ fmm_interface.evaluate(source_curls[0] @ x)[:, 0]
        )
        fmm_res1 = (
            target_curls_trans[1] @ fmm_interface.evaluate(source_curls[1] @ x)[:, 0]
        )
        fmm_res2 = (
            target_curls_trans[2] @ fmm_interface.evaluate(source_curls[2] @ x)[:, 0]
        )

        first_part = fmm_res0 + fmm_res1 + fmm_res2

        fmm_n1 = (
            target_normals[:, 0]
            * fmm_interface.evaluate(source_normals[:, 0] * x_transformed)[:, 0]
        )
        fmm_n2 = (
            target_normals[:, 1]
            * fmm_interface.evaluate(source_normals[:, 1] * x_transformed)[:, 0]
        )
        fmm_n3 = (
            target_normals[:, 2]
            * fmm_interface.evaluate(source_normals[:, 2] * x_transformed)[:, 0]
        )

        second_part = target_map @ (fmm_n1 + fmm_n2 + fmm_n3)

        return first_part + wavenumber * wavenumber * second_part + singular_part @ x

    if operator_descriptor.identifier == "laplace_hypersingular_boundary":
        return evaluate_laplace_hypersingular

    if operator_descriptor.identifier == "helmholtz_hypersingular_boundary":
        return evaluate_helmholtz_hypersingular

    if operator_descriptor.identifier == "modified_helmholtz_hypersingular_boundary":
        return evaluate_modified_helmholtz_hypersingular


def make_default_scalar(operator_descriptor, fmm_interface, domain, dual_to_range):
    """Create an evaluator for scalar operators."""
    import bempp.api
    from bempp.api.integration.triangle_gauss import get_number_of_quad_points

    npoints = get_number_of_quad_points(bempp.api.GLOBAL_PARAMETERS.quadrature.regular)

    source_map = domain.map_to_points(bempp.api.GLOBAL_PARAMETERS.quadrature.regular)
    target_map = dual_to_range.map_to_points(
        bempp.api.GLOBAL_PARAMETERS.quadrature.regular, return_transpose=True
    )

    singular_part = operator_descriptor.singular_part.weak_form().A

    source_normals = get_normals(domain, npoints)
    target_normals = get_normals(dual_to_range, npoints)

    def evaluate_single_layer(x):
        """Actually evaluate single layer."""
        x_transformed = source_map @ x
        fmm_res = fmm_interface.evaluate(x_transformed)[:, 0]
        return target_map @ fmm_res + singular_part @ x

    def evaluate_adjoint_double_layer(x):
        """Actually evaluate adjoint double layer."""
        import numpy as np

        x_transformed = source_map @ x
        fmm_res = np.sum(
            fmm_interface.evaluate(x_transformed)[:, 1:] * target_normals, axis=1
        )
        return target_map @ fmm_res + singular_part @ x

    def evaluate_double_layer(x):
        """Actually evaluate double layer."""

        x_transformed = source_map @ x

        fmm_res1 = fmm_interface.evaluate(source_normals[:, 0] * x_transformed)[:, 1]
        fmm_res2 = fmm_interface.evaluate(source_normals[:, 1] * x_transformed)[:, 2]
        fmm_res3 = fmm_interface.evaluate(source_normals[:, 2] * x_transformed)[:, 3]

        fmm_res = -(fmm_res1 + fmm_res2 + fmm_res3)

        return target_map @ fmm_res + singular_part @ x

    if "single" in operator_descriptor.identifier:
        return evaluate_single_layer
    elif "adjoint_double" in operator_descriptor.identifier:
        return evaluate_adjoint_double_layer
    elif "double" in operator_descriptor.identifier:
        return evaluate_double_layer
    else:
        raise ValueError("Could not recognise identifier string.")


def get_normals(space, npoints):
    """Get the normal vectors on the quadrature points."""
    import numpy as np

    grid = space.grid
    number_of_elements = grid.number_of_elements

    normals = np.empty((npoints * number_of_elements, 3), dtype="float64")
    for element in range(number_of_elements):
        for n in range(npoints):
            normals[npoints * element + n, :] = (
                grid.normals[element] * space.normal_multipliers[element]
            )
    return normals


def compute_p1_curl_transformation(space, quadrature_order):
    """
    Compute the transformation of P1 space coefficients to surface curl values.

    Returns two lists, curl_transforms and curl_transforms_transpose. The jth matrix
    in curl_transforms is the map from P1 function space coefficients (or extended space
    built upon P1 type spaces) to the jth component of the surface curl evaluated at the
    quadrature points, multiplied with the quadrature weights and integration element. The
    list curl_transforms_transpose contains the transpose of these matrices.
    """
    from bempp.api.integration.triangle_gauss import rule
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import aslinearoperator

    grid_data = space.grid.data("double")
    number_of_elements = space.grid.number_of_elements
    quad_points, weights = rule(quadrature_order)
    npoints = len(weights)
    dof_count = space.localised_space.grid_dof_count

    data, iind, jind = compute_p1_curl_transformation_impl(
        grid_data,
        space.support_elements,
        space.normal_multipliers,
        quad_points,
        weights,
    )

    curl_transforms = []
    curl_transforms_transpose = []

    for index in range(3):
        curl_transforms.append(
            aslinearoperator(
                coo_matrix(
                    (data[index, :], (iind, jind)),
                    shape=(npoints * number_of_elements, dof_count),
                ).tocsr()
            )
            @ aslinearoperator(space.map_to_localised_space)
            @ aslinearoperator(space.dof_transformation)
        )
        curl_transforms_transpose.append(
            aslinearoperator(space.dof_transformation.T)
            @ aslinearoperator(space.map_to_localised_space.T)
            @ aslinearoperator(
                coo_matrix(
                    (data[index, :], (jind, iind)),
                    shape=(dof_count, npoints * number_of_elements),
                ).tocsr()
            )
        )

    return curl_transforms, curl_transforms_transpose


@_numba.njit
def compute_p1_curl_transformation_impl(
    grid_data, support_elements, normal_multipliers, quad_points, weights
):
    """Implement the curl transformation."""

    number_of_quad_points = quad_points.shape[1]
    number_of_support_elements = len(support_elements)

    nlocal = 3 * number_of_quad_points

    data = _np.empty((3, nlocal * number_of_support_elements), dtype=_np.float64)
    jind = _np.empty(nlocal * number_of_support_elements, dtype=_np.int64)
    iind = _np.empty(nlocal * number_of_support_elements, dtype=_np.int64)

    reference_values = _np.array([[-1, 1, 0], [-1, 0, 1]], dtype=_np.float64)

    for element_index, element in enumerate(support_elements):
        surface_gradient = grid_data.jac_inv_trans[element] @ reference_values
        for function_index in range(3):
            surface_curl = normal_multipliers[element] * _np.cross(
                grid_data.normals[element], surface_gradient[:, function_index]
            )

            for point_index in range(number_of_quad_points):
                index = (
                    nlocal * element_index
                    + function_index * number_of_quad_points
                    + point_index
                )
                data[:, index] = surface_curl * (
                    weights[point_index] * grid_data.integration_elements[element]
                )
                iind[index] = number_of_quad_points * element_index + point_index
                jind[index] = 3 * element_index + function_index

    return (data, iind, jind)


def make_default_scalar_potential(operator_descriptor, fmm_interface, space):
    """Make a scalar potential operator."""
    import bempp.api
    from bempp.api.integration.triangle_gauss import get_number_of_quad_points

    npoints = get_number_of_quad_points(bempp.api.GLOBAL_PARAMETERS.quadrature.regular)
    source_map = space.map_to_points(bempp.api.GLOBAL_PARAMETERS.quadrature.regular)
    source_normals = get_normals(space, npoints)

    def evaluate_single_layer(x):
        """Evaluate the single-layer operator."""
        return fmm_interface.evaluate(source_map @ x)[:, 0].reshape([1, -1])

    def evaluate_double_layer(x):
        """Evaluate the double-layer operator."""
        x_transformed = source_map @ x
        
        fmm0 = fmm_interface.evaluate(source_normals[:, 0] * x_transformed)[:, 1]
        fmm1 = fmm_interface.evaluate(source_normals[:, 1] * x_transformed)[:, 2]
        fmm2 = fmm_interface.evaluate(source_normals[:, 2] * x_transformed)[:, 3]

        return -(fmm0 + fmm1 + fmm2)

    if "single" in operator_descriptor.identifier:
        return evaluate_single_layer
    elif "double" in operator_descriptor.identifier:
        return evaluate_double_layer

