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
            parameters.quadrature.regular, return_tranpose=True
        )

    def assemble(
        self, operator_descriptor, device_interface, precision, *args, **kwargs
    ):
        """Actually assemble."""
        from bempp.api.space.space import return_compatible_representation
        from bempp.api.assembly.discrete_boundary_operator import (
            BoundaryOperatorWithAssembler,
        )

        actual_domain, actual_dual_to_range = return_compatible_representation(
            self.domain, self.dual_to_range
        )

        self._evaluator = select_evaluator(
            operator_descriptor,
            self._fmm_interface,
            actual_domain,
            actual_dual_to_range,
            self.parameters,
        )

    def matvec(x):
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