"""Implementation of boundary operators."""


class BoundaryOperator(object):
    """A base class for boundary operators."""

    def __init__(self, domain, range_, dual_to_range, parameters):
        """Constructor should only be called through derived class."""
        from bempp.api.utils.helpers import assign_parameters

        self._domain = domain
        self._range = range_
        self._dual_to_range = dual_to_range
        self._parameters = parameters
        self._cached = None
        self._range_map = None

    @property
    def domain(self):
        """Return the domain space of the operator."""
        return self._domain

    @property
    def range(self):
        """Return the range space of the operator."""
        return self._range

    @property
    def dual_to_range(self):
        """Return the dual to range space of the operator."""
        return self._dual_to_range

    @property
    def parameters(self):
        """Return the parameters associated with the operator."""
        return self._parameters

    def assemble(self, *args, **kwargs):
        """Assemble the operator."""
        raise NotImplementedError

    def weak_form(self, *args, **kwargs):
        """Return cached weak form (assemble if necessary). """
        if not self._cached:
            self._cached = self.assemble(*args, **kwargs)

        return self._cached

    def strong_form(self, recompute=False):
        """Return a discrete operator  that maps into the range space.

        Parameters
        ----------
        recompute : bool
            Usually the strong form is cached. If this parameter is set to
            `true` the strong form is recomputed.
        """
        if recompute is True:
            self._range_map = None

        if self._range_map is None:

            # This is the most frequent case and we cache the mass
            # matrix from the space object.
            if self.range == self.dual_to_range:
                self._range_map = \
                    self.dual_to_range.inverse_mass_matrix()
            else:
                from bempp.api.assembly.discrete_boundary_operator import \
                    InverseSparseDiscreteBoundaryOperator
                from bempp.api.operators.boundary.sparse import identity

                self._range_map = InverseSparseDiscreteBoundaryOperator(
                    identity(
                        self.range,
                        self.range, self.dual_to_range).weak_form())

        return self._range_map * self.weak_form(recompute)


    def __mul__(self, other):
        """Return product with a scalar, grid function or other operator."""
        import numpy as np
        from bempp.api import GridFunction

        if np.isscalar(other):
            return _ScaledBoundaryOperator(self, other)
        elif isinstance(other, BoundaryOperator):
            return _ProductBoundaryOperator(self, other)
        elif isinstance(other, GridFunction):
            if not self.domain.is_compatible(other.space):
                raise ValueError(
                    "Operator domain space does not match GridFunction space.")
            return GridFunction(self.range,
                                projections=self.weak_form() *
                                other.coefficients,
                                dual_space=self.dual_to_range)
        else:
            return NotImplemented

    def __rmul__(self, other):

        import numpy as np

        if np.isscalar(other):
            return _ScaledBoundaryOperator(self, other)
        else:
            return NotImplemented

    def __add__(self, other):
        """Return sum of two boundary operators."""
        return _SumBoundaryOperator(self, other)

    def __neg__(self):

        return self.__mul__(-1.0)

    def __sub__(self, other):

        return self.__add__(-other)


class BoundaryOperatorWithAssembler(BoundaryOperator):
    """Implements a boundary operator together with an assembler."""

    # pylint: disable=too-many-arguments
    def __init__(self, domain, range_, dual_to_range, assembler, operator_descriptor):
        """Initialize a boundary operator with assembler."""
        super().__init__(domain, range_, dual_to_range, assembler.parameters)

        self._assembler = assembler
        self._operator_descriptor = operator_descriptor

    @property
    def assembler(self):
        """Return the assembler associated with this operator."""
        return self._assembler

    @property
    def descriptor(self):
        """Operator descriptor."""
        return self._operator_descriptor

    def assemble(self, *args, **kwargs):
        """Assemble the operator."""
        return self.assembler.assemble(self.descriptor, *args, **kwargs)


class _SumBoundaryOperator(BoundaryOperator):
    """Return the sum of two boundary operators."""

    def __init__(self, op1, op2):
        """Construct the sum of two boundary operators."""
        if (not op1.domain.is_compatible(op2.domain) or
                not op1.range.is_compatible(op2.range) or
                not op1.dual_to_range.is_compatible(op2.dual_to_range)):
            raise ValueError("Spaces not compatible.")

        super(_SumBoundaryOperator, self).__init__(
            op1.domain, op1.range, op1.dual_to_range, None)

        self._op1 = op1
        self._op2 = op2

    def assemble(self, *args, **kwargs):
        """Implement the weak form."""
        return self._op1.weak_form(*args, **kwargs) + self._op2.weak_form(*args, **kwargs)


class _ScaledBoundaryOperator(BoundaryOperator):
    """Scale a boundary operator."""

    def __init__(self, op, alpha):
        """Construct a scaled boundary operator."""
        super(_ScaledBoundaryOperator, self).__init__(
            op.domain, op.range, op.dual_to_range,
            op.parameters)

        self._op = op
        self._alpha = alpha

    def assemble(self, *args, **kwargs):
        """Implement the weak form."""
        return self._op.weak_form(*args, **kwargs) * self._alpha


class _ProductBoundaryOperator(BoundaryOperator):
    """Multiply two boundary operators."""

    def __init__(self, op1, op2):
        """Construct a product boundary operator."""
        if not op2.range.is_compatible(op1.domain):
            raise ValueError(
                "Range space of second operator must be compatible to " +
                "domain space of first operator.")

        super(_ProductBoundaryOperator, self).__init__(
            op2.domain, op1.range, op1.dual_to_range, None)

        self._op1 = op1
        self._op2 = op2

    def assemble(self):
        """Implement the weak form."""
        return self._op1.weak_form() * self._op2.strong_form()

class ZeroBoundaryOperator(BoundaryOperator):
    """A boundary operator that represents a zero operator.

    Parameters
    ----------
    domain : bempp.api.space.Space
        Domain space of the operator.
    range_ : bempp.api.space.Space
        Range space of the operator.
    dual_to_range : bempp.api.space.Space
        Dual space to the range space.

    """

    def __init__(self, domain, range_, dual_to_range):
        super(ZeroBoundaryOperator, self).__init__(
            domain, range_, dual_to_range)

    def assemble(self, *args, **kwargs):

        from bempp.api.assembly.discrete_boundary_operator \
            import ZeroDiscreteBoundaryOperator

        return ZeroDiscreteBoundaryOperator(self.dual_to_range.global_dof_count,
                                            self.domain.global_dof_count)

    def __iadd__(self, other):
        if (self.domain != other.domain or
                self.range != other.range or
                self.dual_to_range != other.dual_to_range):
            raise ValueError("Spaces not compatible.")

        return other

    def __isub__(self, other):
        if (self.domain != other.domain or
                self.range != other.range or
                self.dual_to_range != other.dual_to_range):
            raise ValueError("Spaces not compatible.")

        return -other