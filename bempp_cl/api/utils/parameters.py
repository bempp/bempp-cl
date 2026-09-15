"""Define the default parameters object."""

# pylint: disable=too-few-public-methods


class _Verbosity:
    """Verbosity Parameters."""

    def __init__(self):
        """Initialize verbosity parameters."""
        self.extended_verbosity = False


class _Output:
    """Output parameters."""

    def __init__(self):
        """Initialize output parameters."""
        self.gmsh_use_binary = True


class _Quadrature:
    """Quadrature orders."""

    def __init__(self):
        """Iniitalize quadrature parameters."""
        self.regular = 4
        self.singular = 4


class _Fmm:
    """Fmm options."""

    def __init__(self):
        """Iniitalize FMM parameters."""
        self.expansion_order = 5
        self.depth = 4
        self.ncrit = 400
        self.near_field_representation = "evaluate"
        self.debug = False
        self.dense_evaluation = False


class _DenseAssembly:
    """Dense assembly options."""

    def __init__(self):
        """Iniitalize dense assembly parameters."""
        self.workgroup_size_multiple = 2


class _Assembly:
    """Assembly options."""

    def __init__(self):
        """Iniitalize assembly parameters."""
        self.dense = _DenseAssembly()
        self.always_promote_to_double = False
        self.discretization_type = "galerkin"


class DefaultParameters:
    """Default parameters for Bempp."""

    def __init__(self):
        """Initialize parameters."""
        self.verbosity = _Verbosity()
        self.output = _Output()
        self.quadrature = _Quadrature()
        self.assembly = _Assembly()
        self.fmm = _Fmm()
