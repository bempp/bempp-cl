"""Implementation of an FMM Assembler."""

from bempp.api.assembly import assembler as _assembler


def create_evaluator(
    operator_descriptor, fmm_interface, domain, dual_to_range, parameters
):
    """Return an Fmm evaluator for the requested kernel."""

    if operator_descriptor.assembly_type == "default_scalar":
        return make_default_scalar(
            operator_descriptor, fmm_interface, domain, dual_to_range
        )


class FmmAssembler(_assembler.AssemblerBase):
    """Assembler for Fmm."""

    # pylint: disable=useless-super-delegation
    def __init__(self, domain, dual_to_range, parameters, fmm_interface):
        """Create an Fmm assembler instance."""
        super().__init__(domain, dual_to_range, parameters)

        self._fmm_interface = fmm_interface

        self._source_map = domain.map_to_points(parameters.quadrature.regular)
        self._target_map = dual_to_range.map_to_points(
            parameters.quadrature.regular, return_transpose=True
        )

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

        self._evaluator = create_evaluator(
            operator_descriptor,
            self._fmm_interface,
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

def make_default_scalar(operator_descriptor, fmm_interface, domain, dual_to_range):
    """Create an evaluator for scalar operators."""
    import bempp.api
    from bempp.api.integration.triangle_gauss import get_number_of_quad_points

    npoints = get_number_of_quad_points(
            bempp.api.GLOBAL_PARAMETERS.quadrature.regular)

    source_map = domain.map_to_points(bempp.api.GLOBAL_PARAMETERS.quadrature.regular)
    target_map = dual_to_range.map_to_points(
        bempp.api.GLOBAL_PARAMETERS.quadrature.regular, return_transpose=True
    )

    singular_part = operator_descriptor.singular_part.weak_form().A

    source_normals = get_normals(domain, npoints)
    target_normals = get_normals(dual_to_range, npoints)


    if "single" in operator_descriptor.identifier:
        mode = "single"
    elif "adjoint_double" in operator_descriptor.identifier:
        mode = "adjoint_double"
    elif "double" in operator_descriptor.identifier:
        mode = "double"
    else:
        raise ValueError("Could not recognise identifier string.")


    def evaluate(x):
        """Actually evaluate."""
        x_transformed = source_map @ x
        fmm_res = fmm_interface.evaluate(x_transformed, kernel_mode=kernel_mode)
        return target_map @ fmm_res + singular_part @ x

    return evaluate

def get_normals(space, npoints):
    """Get the normal vectors on the quadrature points."""
    grid = space.grid
    number_of_elements = grid.number_of_elements

    normals = np.empty(
        (npoints * number_of_elements, 3), dtype="float64"
    )
    for element in range(number_of_elements):
        for n in range(npoints):
            normals[npoints * element + n, :] = grid.normals[element] * space.normal_multipliers[element]
    return normals

