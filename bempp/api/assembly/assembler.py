"""Various assemblers to discretize boundary operators."""


def _create_assembler(
    domain, dual_to_range, identifier, parameters, device_interface=None,
):
    """Create assembler based on string."""
    from bempp.core.singular_assembler import SingularAssembler
    from bempp.core.dense_assembler import DenseAssembler
    from bempp.api.fmm.fmm_assembler import FmmAssembler
    #from bempp.core.numba.dense_assembler import DenseAssembler
    from bempp.core.sparse_assembler import SparseAssembler

    # from bempp.core.dense_assembler import DenseAssembler
    # from bempp.core.sparse_assembler import SparseAssembler
    # from bempp.core.dense_evaluator import DenseEvaluatorAssembler
    # from bempp.core.dense_multitrace_evaluator import DenseMultitraceEvaluatorAssembler



    if identifier == "only_singular_part":
        return SingularAssembler(domain, dual_to_range, parameters)
    if identifier == "dense":
        return DenseAssembler(domain, dual_to_range, parameters)
    if identifier == "default_nonlocal":
        return DenseAssembler(domain, dual_to_range, parameters)
    if identifier == "sparse":
        return SparseAssembler(domain, dual_to_range, parameters)
    if identifier == "fmm":
        return FmmAssembler(domain, dual_to_range, parameters)
    else:
        raise ValueError("Unknown assembler type.")
    # if identifier == "dense_evaluator":
        # return DenseEvaluatorAssembler(domain, dual_to_range, parameters)
    # if identifier == "multitrace_evaluator":
        # return DenseMultitraceEvaluatorAssembler(domain, dual_to_range, parameters)


class AssemblerInterface(object):
    """Default Assembler interface object."""

    def __init__(
        self,
        domain,
        dual_to_range,
        assembler,
        device_interface,
        precision,
        parameters=None,
    ):
        """Initialize assembler based on assembler_type string."""
        import bempp.api
        import bempp.api as _api

        self._domain = domain
        self._dual_to_range = dual_to_range
        self._parameters = _api.assign_parameters(parameters)
        self._device_interface = device_interface
        self._precision = precision

        if self._device_interface is None:
            self._device_interface = bempp.api.DEFAULT_DEVICE_INTERFACE

        if not isinstance(assembler, str):
            self._implementation = assembler
        else:
            self._implementation = _create_assembler(
                domain, dual_to_range, assembler, self.parameters, device_interface
            )

    @property
    def domain(self):
        """Return domain space."""
        return self._domain

    @property
    def dual_to_range(self):
        """Return dual to range space."""
        return self._dual_to_range

    @property
    def parameters(self):
        """Return parameters."""
        return self._parameters

    def assemble(self, operator_descriptor, *args, **kwargs):
        """Assemble the operator."""
        return self._implementation.assemble(
            operator_descriptor,
            self._device_interface,
            self._precision,
            *args,
            **kwargs
        )


class AssemblerBase(object):
    """Base class for assemblers."""

    def __init__(self, domain, dual_to_range, parameters=None):
        """Instantiate the base class."""
        import bempp.api as api

        self._domain = domain
        self._dual_to_range = dual_to_range
        self._parameters = api.assign_parameters(parameters)

    @property
    def domain(self):
        """Return domain."""
        return self._domain

    @property
    def dual_to_range(self):
        """Return dual to range."""
        return self._dual_to_range

    @property
    def parameters(self):
        """Return parameters."""
        return self._parameters

    def assemble(self, operator_descriptor, *args, **kwargs):
        """Assemble the operator."""
        raise NotImplementedError("Needs to be implemented by derived class.")
