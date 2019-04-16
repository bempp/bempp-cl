"""Definition of blocked operator structures."""

import numpy as _np
from bempp.api.assembly.discrete_boundary_operator import DiscreteBoundaryOperator


def _sum(op1, op2):
    """Sum of two operators, allowing for one of them to be None."""
    if op1 is None:
        return op2
    elif op2 is None:
        return op1
    else:
        return op1 + op2


def _prod(op1, op2):
    """Product of two operators, allowing for one of them to be None."""
    if op1 is None or op2 is None:
        return None
    else:
        return op1 * op2


class BlockedOperatorBase(object):
    """Base class for blocked operators."""

    def __init__(self, m, n):
        """Construct an m x n blocked operator."""
        self._ndims = (m, n)
        self._weak_form = None
        self._range_map = None
        self._cached = None

    def assemble(self, *args, **kwargs):
        """Assemble the blocked operator."""
        raise NotImplementedError

    def weak_form(self, *args, **kwargs):
        """Return cached weak form (assemble if necessary). """
        if not self._cached:
            self._cached = self.assemble(*args, **kwargs)
        return self._cached

    def strong_form(self, recompute=False):
        """Return a discrete operator that maps into the range space.

        Parameters
        ----------
        recompute : bool
            Usually the strong form is cached. If this parameter is set to
            `true` the strong form is recomputed.
        """
        if recompute is True:
            self._range_map = None

        if self._range_map is None:

            _range_ops = _np.empty((self.ndims[0], self.ndims[1]), dtype="O")

            for index in range(self.ndims[0]):

                # This is the most frequent case and we cache the mass
                # matrix from the space object.
                if self.range_spaces[index] == self.dual_to_range_spaces[index]:
                    _range_ops[index, index] = self.dual_to_range_spaces[
                        index
                    ].inverse_mass_matrix()
                else:
                    from bempp.api.operators.boundary.sparse import identity
                    from bempp.api.assembly.discrete_boundary_operator import (
                        InverseSparseDiscreteBoundaryOperator,
                    )

                    _range_ops[index, index] = InverseSparseDiscreteBoundaryOperator(
                        identity(
                            self.range_spaces[index],
                            self.range_spaces[index],
                            self.dual_to_range_spaces[index],
                        ).weak_form()
                    )

            self._range_map = BlockedDiscreteOperator(_range_ops)

        return self._range_map * self.weak_form(recompute)

    def __getitem__(self, key):
        """Return an operator indexed by a key of the form (i, j)."""
        raise NotImplementedError()

    @property
    def ndims(self):
        """Number of block rows and block columns."""
        return self._ndims

    @property
    def range_spaces(self):
        """A list of range spaces of the blocked operator."""
        raise NotImplementedError()

    @property
    def dual_to_range_spaces(self):
        """A list of dual to range spaces of the blocked operator."""
        raise NotImplementedError()

    @property
    def domain_spaces(self):
        """A list of domain spaces of the blocked operator."""
        raise NotImplementedError()

    def __add__(self, other):
        """Add two blocked operators."""
        if not isinstance(other, BlockedOperatorBase):
            return NotImplementedError

        return SumBlockedOperator(self, other)

    def __neg__(self):
        """Multiply the blocked operator with -1."""
        return self.__mul__(-1)

    def __sub__(self, other):
        """Subtract another blocked operator from this blocked operator."""
        return self.__add__(-other)

    def __mul__(self, other):
        """Multiply two blocked operators."""
        import collections

        if _np.isscalar(other):
            # Multiplication with scalar
            return ScaledBlockedOperator(self, other)
        elif isinstance(other, BlockedOperatorBase):
            # Multiplication with another blocked operator.
            return ProductBlockedOperator(self, other)
        elif isinstance(other, collections.Iterable):
            # Multiplication with a list of grid functions.
            from bempp.api.assembly.grid_function import GridFunction

            list_input = list(other)
            if len(list_input) != self.ndims[1]:
                raise ValueError(
                    "Length of input list is {0}.".format(len(list_input))
                    + ". But domain dimension of blocked operator"
                    + "is {0}".format(self.ndims[1])
                )
            for item in list_input:
                if not isinstance(item, GridFunction):
                    raise ValueError(
                        "All items in the input list must be grid functions."
                    )
            weak_op = self.weak_form()
            input_type = list_input[0].coefficients.dtype
            for item in list_input:
                input_type = _np.promote_types(input_type, item.coefficients.dtype)
            x_in = _np.zeros(weak_op.shape[1], dtype=input_type)
            col_pos = _np.hstack([[0], _np.cumsum(weak_op.column_dimensions)])
            row_pos = _np.hstack([[0], _np.cumsum(weak_op.row_dimensions)])
            for index in range(weak_op.ndims[1]):
                x_in[col_pos[index] : col_pos[index + 1]] = list_input[
                    index
                ].coefficients
            res = weak_op * x_in

            # Now assemble the output grid functions back together.
            output_list = []

            for index in range(weak_op.ndims[0]):
                output_list.append(
                    GridFunction(
                        self.range_spaces[index],
                        dual_space=self.dual_to_range_spaces[index],
                        projections=res[row_pos[index] : row_pos[index + 1]],
                    )
                )
            return output_list

        else:
            return NotImplemented

    def __rmul__(self, other):
        """Right multiplication."""
        if _np.isscalar(other):
            return self.__mul__(other)
        else:
            return NotImplemented


class BlockedOperator(BlockedOperatorBase):
    """Generic blocked boundary operator."""

    def __init__(self, m, n):
        """Construct an (m x n) blocked boundary operator."""
        super(BlockedOperator, self).__init__(m, n)

        self._operators = _np.empty((m, n), dtype=_np.object)
        self._rows = m * [False]
        self._cols = n * [False]

        self._dual_to_range_spaces = m * [None]
        self._range_spaces = m * [None]
        self._domain_spaces = n * [None]

    def __getitem__(self, key):
        """Return a component operator with a key of the form (i, j)."""
        import bempp.api

        if self._operators[key] is None:
            return bempp.api.ZeroBoundaryOperator(
                self.domain_spaces[key[1]],
                self.range_spaces[key[0]],
                self.dual_to_range_spaces[key[0]],
            )
        else:
            return self._operators[key]

    def __setitem__(self, key, operator):
        """Set a component operator with a key of the form (i, j)."""
        row = key[0]
        col = key[1]

        if self.range_spaces[row] is not None:
            if operator.range != self.range_spaces[row]:
                raise ValueError(
                    "Range space not compatible with "
                    + "self.range_spaces[{0}]".format(row)
                )

        if self.dual_to_range_spaces[row] is not None:
            if operator.dual_to_range != self.dual_to_range_spaces[row]:
                raise ValueError(
                    "Dual to range space not compatible with "
                    + "self.dual_to_range_spaces[{0}]".format(row)
                )

        if self.domain_spaces[col] is not None:
            if operator.domain != self.domain_spaces[col]:
                raise ValueError(
                    "Domain space not compatible with "
                    + "self.domain_spaces[{0}]".format(col)
                )

        self._range_spaces[row] = operator.range
        self._dual_to_range_spaces[row] = operator.dual_to_range
        self._domain_spaces[col] = operator.domain
        self._operators[key] = operator
        self._rows[row] = True
        self._cols[col] = True

    def _fill_complete(self):
        """Check if each row and  column contain at least one operator."""
        return (False not in self._cols) and (False not in self._rows)

    def assemble(self, *args, **kwargs):
        """Implement the weak form."""
        if not self._fill_complete():
            raise ValueError("Each row and column must have at least one operator")

        ops = _np.empty((self.ndims[0], self.ndims[1]), dtype="O")

        for i in range(self.ndims[0]):
            for j in range(self.ndims[1]):
                if self._operators[i, j] is not None:
                    ops[i, j] = self._operators[i, j].weak_form(*args, **kwargs)

        return BlockedDiscreteOperator(ops)

    def strong_form(self, recompute=False):
        """Return the strong form of the blocked operator."""
        if not self._fill_complete():
            raise ValueError("Each row and column must have at least one operator")

        return super(BlockedOperator, self).strong_form(recompute)

    @property
    def range_spaces(self):
        """Return the list of range spaces."""
        return tuple(self._range_spaces)

    @property
    def dual_to_range_spaces(self):
        """Return the list of dual_to_range spaces."""
        return tuple(self._dual_to_range_spaces)

    @property
    def domain_spaces(self):
        """Return the list of domain spaces."""
        return tuple(self._domain_spaces)


class MultitraceOperatorFromAssembler(BlockedOperatorBase):
    """A multitrace operator from an assembler."""

    def __init__(
        self,
        domain_spaces,
        range_spaces,
        dual_to_range_spaces,
        assembler,
        multitrace_operator_descriptor,
    ):
        """Generate operator from assembler."""
        
        super().__init__(2, 2)
        self._domain_spaces = domain_spaces
        self._range_spaces = range_spaces
        self._dual_to_range_spaces = dual_to_range_spaces
        self._assembler = assembler
        self._descriptor = multitrace_operator_descriptor

    @property
    def assembler(self):
        """Return the assembler associated with this operator."""
        return self._assembler

    @property
    def descriptor(self):
        """Operator descriptor."""
        return self._descriptor

    def assemble(self, *args, **kwargs):
        """Assemble the operator."""
        return self.assembler.assemble(self.descriptor, *args, **kwargs)

    @property
    def range_spaces(self):
        """A list of range spaces of the blocked operator."""
        return self._range_spaces

    @property
    def dual_to_range_spaces(self):
        """A list of dual to range spaces of the blocked operator."""
        return self._dual_to_range_spaces

    @property
    def domain_spaces(self):
        """A list of domain spaces of the blocked operator."""
        return self._domain_spaces



class SumBlockedOperator(BlockedOperatorBase):
    """Represents the sum of two blocked boundary operators."""

    def __init__(self, op1, op2):
        """Construct the object from two blocked operators."""
        if op1.ndims != op2.ndims:
            raise ValueError(
                "Incompatible dimensions: {0} != {1}.".format(op1.ndims, op2.ndims)
            )

        for index in range(op1.ndims[0]):
            if op1.range_spaces[index] != op2.range_spaces[index]:
                raise ValueError(
                    "Range spaces at index {0} are not identical.".format(index)
                )

        for index in range(op1.ndims[0]):
            if op1.dual_to_range_spaces[index] != op2.dual_to_range_spaces[index]:
                raise ValueError(
                    "Dual_to_range spaces at index {0}".format(index)
                    + " are not identical."
                )

        for index in range(op1.ndims[1]):
            if op1.domain_spaces[index] != op2.domain_spaces[index]:
                raise ValueError(
                    "Domain spaces at index {0} are not identical.".format(index)
                )

        self._op1 = op1
        self._op2 = op2

        super(SumBlockedOperator, self).__init__(op1.ndims[0], op1.ndims[1])

    def assemble(self, *args, **kwargs):

        return self._op1.weak_form(*args, **kwargs) + self._op2.weak_form(
            *args, **kwargs
        )

    def __getitem__(self, key):
        """Return a component operator with a key of the form (i, j)."""
        return _sum(self._op1[key], self._op2[key])

    @property
    def range_spaces(self):
        """Return the list of range spaces."""
        return tuple(self._op1.range_spaces)

    @property
    def dual_to_range_spaces(self):
        """Return the list of dual_to_range spaces."""
        return tuple(self._op1.dual_to_range_spaces)

    @property
    def domain_spaces(self):
        """Return the list of domain spaces."""
        return tuple(self._op1.domain_spaces)


class ProductBlockedOperator(BlockedOperatorBase):
    """Represents the Product of two blocked boundary operators."""

    def __init__(self, op1, op2):
        """Construct the blocked operator product op1 * op2."""
        if op1.ndims[1] != op2.ndims[0]:
            raise ValueError(
                "Incompatible dimensions: {0} != {1}.".format(
                    op1.ndims[1], op2.ndims[0]
                )
            )

        for index in range(op1.ndims[1]):
            if op1.domain_spaces[index] != op2.range_spaces[index]:
                raise ValueError(
                    "Range and domain space at index "
                    + " {0} not identical.".format(index)
                )

        self._op1 = op1
        self._op2 = op2

        super(ProductBlockedOperator, self).__init__(op1.ndims[0], op2.ndims[1])

    def assemble(self, *args, **kwargs):

        return self._op1.weak_form(*args, **kwargs) * self._op2.strong_form(
            *args, **kwargs
        )

    def __getitem__(self, key):
        """Return a component operator with a key of the form (i, j)."""
        import bempp.api

        i = key[0]
        j = key[1]

        res = bempp.api.ZeroBoundaryOperator(
            self.domain_spaces[j], self.range_spaces[i], self.dual_to_range_spaces[i]
        )

        for k in range(self._op1.ndims[1]):
            res = _sum(res, _prod(self._op1[i, k], self._op2[k, j]))

        return res

    @property
    def range_spaces(self):
        """Return the list of range spaces."""
        return tuple(self._op1.range_spaces)

    @property
    def dual_to_range_spaces(self):
        """Return the list of dual_to_range spaces."""
        return tuple(self._op1.dual_to_range_spaces)

    @property
    def domain_spaces(self):
        """Return the list of domain spaces."""
        return tuple(self._op2.domain_spaces)


class ScaledBlockedOperator(BlockedOperatorBase):
    """Represents the scalar multiplication of a blocked operator."""

    def __init__(self, op, alpha):
        """Construct the scaled operator alpha * op."""
        self._op = op
        self._alpha = alpha

        super(ScaledBlockedOperator, self).__init__(op.ndims[0], op.ndims[1])

    def _weak_form_impl(self):
        return self._alpha * self._op.weak_form()

    def __getitem__(self, key):
        """Return a component operator with a key of the form (i, j)."""
        import bempp.api

        if self._op[key] is None:
            return bempp.api.ZeroBoundaryOperator(
                self.domain_spaces[key[1]],
                self.range_spaces[key[0]],
                self.dual_to_range_spaces[key[0]],
            )
        else:
            return self._op[key] * self._alpha

    @property
    def range_spaces(self):
        """Return the list of range spaces."""
        return tuple(self._op.range_spaces)

    @property
    def dual_to_range_spaces(self):
        """Return the list of dual_to_range spaces."""
        return tuple(self._op.dual_to_range_spaces)

    @property
    def domain_spaces(self):
        """Return the list of domain spaces."""
        return tuple(self._op.domain_spaces)


class BlockedDiscreteOperatorBase(DiscreteBoundaryOperator):
    """Base class for discrete blocked boundary operators."""

    def __init__(self, m, n, dtype, shape):
        """
        Create an mxn blocked discrete operator.

        Parameters
        ----------
        m : int
            Blocked row dimension
        n : int
            Blocked column dimension
        dtype : np.dtype
            Type of the discrete operator
        shape : tuple
            Shape (number of rows, number of columns) of the
            resulting discrete operator.

        """
        self._ndims = (m, n)
        super(BlockedDiscreteOperatorBase, self).__init__(dtype, shape)

    @property
    def ndims(self):
        """Tuple with the number of row and column dimensions."""
        return self._ndims

    @property
    def row_dimensions(self):
        """Return the list of row dimensions."""
        raise NotImplementedError()

    @property
    def column_dimensions(self):
        """Return the list of column dimensions."""
        raise NotImplementedError()

    def __getitem__(self, key):
        """Return the discrete operator at position (i, j)."""
        raise NotImplementedError()

    def _adjoint(self):
        """Compute the adjoint of the discrete blocked operator."""
        raise NotImplementedError()

    def _transpose(self):
        """Compute the transpose of the discrete blocked operator."""
        raise NotImplementedError()

    def __mul__(self, other):
        """Multiply two discrete blocked operators."""
        return self.dot(other)

    def __add__(self, other):
        """Add a discrete blocked operator with another operator."""
        if isinstance(other, BlockedDiscreteOperatorBase):
            return BlockedDiscreteOperatorSum(self, other)
        else:
            return super(BlockedDiscreteOperatorBase, self).__add__(other)

    def dot(self, other):
        """Multiply a discrete blocked operator with something else."""
        from scipy.sparse.linalg.interface import LinearOperator as _LinearOperator

        if isinstance(other, BlockedDiscreteOperatorBase):
            return BlockedDiscreteOperatorProduct(self, other)
        elif _np.isscalar(other):
            return BlockedScaledDiscreteOperator(self, other)
        elif isinstance(other, _LinearOperator):
            return super(BlockedDiscreteOperatorBase, self).dot(other)
        else:
            x_in = _np.asarray(other)
            if x_in.ndim == 1 or (x_in.ndim == 2 and x_in.shape[1] == 1):
                return self._matvec(x_in)
            elif x_in.ndim == 2:
                return self._matmat(x_in)
            else:
                raise ValueError("Expect a 1d or 2d array or matrix.")

    def __rmul__(self, other):
        """Right multiplication with a scalar."""
        if _np.isscalar(other):
            return self * other
        else:
            raise ValueError(
                "Cannot multiply operand of type {0} from the left.".format(type(other))
            )

    def __call__(self, other):
        """Multiply with something else."""
        return self.dot(other)

    def __matmul__(self, other):
        """Matrix multiplication overload."""
        if _np.isscalar(other):
            raise ValueError("Scalar operands not allowed. Use '*' instead.")
        return self.dot(other)

    def __neg__(self):
        """Negate the operator."""
        return -1 * self

    def __sub__(self, other):
        """Subtract something else from operator."""
        return self.__add__(-other)

    import scipy

    if scipy.__version__ < "0.16.0":

        def adjoint(self):
            """Return the adjoint."""
            return self._adjoint()

        def transpose(self):
            """Return the transpose."""
            return self._transpose()

        H = property(adjoint)
        T = property(transpose)


class BlockedDiscreteOperator(BlockedDiscreteOperatorBase):
    """Implementation of a discrete blocked boundary operator."""

    def __init__(self, ops):
        """
        Construct an operator from a two dimensional Numpy array of operators.

        ops is a list of list containing discrete boundary operators or None.
        A None entry is equivalent to a zero discrete boundary operator.

        """
        # pylint: disable=too-many-branches
        if not isinstance(ops, _np.ndarray):
            ops = _np.array(ops)

        rows = ops.shape[0]
        cols = ops.shape[1]

        self._operators = _np.empty((rows, cols), dtype=_np.object)
        self._rows = -_np.ones(rows, dtype=int)
        self._cols = -_np.ones(cols, dtype=int)

        for i in range(rows):
            for j in range(cols):
                if ops[i, j] is None:
                    continue
                if self._rows[i] != -1:
                    if ops[i, j].shape[0] != self._rows[i]:
                        raise ValueError(
                            "Block row {0} has incompatible ".format(i)
                            + " operator sizes."
                        )
                else:
                    self._rows[i] = ops[i, j].shape[0]

                if self._cols[j] != -1:
                    if ops[i, j].shape[1] != self._cols[j]:
                        raise ValueError(
                            "Block column {0} has incompatible".format(j)
                            + "operator sizes."
                        )
                else:
                    self._cols[j] = ops[i, j].shape[1]
                self._operators[i, j] = ops[i, j]

        if not self._fill_complete():
            raise ValueError("Each row and column must contain at least one operator.")

        from bempp.api.assembly.discrete_boundary_operator import (
            ZeroDiscreteBoundaryOperator,
        )

        for i in range(rows):
            for j in range(cols):
                if self._operators[i, j] is None:
                    self._operators[i, j] = ZeroDiscreteBoundaryOperator(
                        self._rows[i], self._cols[j]
                    )

        shape = (_np.sum(self._rows), _np.sum(self._cols))

        from bempp.api.utils.data_types import combined_type

        dtype = "float32"
        for obj in self._operators.ravel():
            if obj is not None:
                dtype = combined_type(dtype, obj.dtype)

        super(BlockedDiscreteOperator, self).__init__(
            ops.shape[0], ops.shape[1], dtype, shape
        )

    def __getitem__(self, key):
        """Return the object at position (i, j)."""
        return self._operators[key]

    def _fill_complete(self):
        if (-1 in self._rows) or (-1 in self._cols):
            return False
        return True

    def _matvec(self, x):
        from bempp.api.utils.data_types import combined_type

        expand_input_dim = False
        if x.ndim == 1:
            x_new = _np.expand_dims(x, 1)
            expand_input_dim = True
            return self.matvec(x_new).ravel()

        if not self._fill_complete():
            raise ValueError("Not all rows or columns contain operators.")

        row_dim = 0
        res = _np.zeros(
            (self.shape[0], x.shape[1]), dtype=combined_type(self.dtype, x.dtype)
        )

        for i in range(self.ndims[0]):
            col_dim = 0
            local_res = res[row_dim : row_dim + self._rows[i], :]
            for j in range(self.ndims[1]):
                local_x = x[col_dim : col_dim + self._cols[j], :]
                op_is_complex = _np.iscomplexobj(self._operators[i, j].dtype.type(1))
                if _np.iscomplexobj(x) and not op_is_complex:
                    local_res[:] += self._operators[i, j].dot(
                        _np.real(local_x)
                    ) + 1j * self._operators[i, j].dot(_np.imag(local_x))
                else:
                    local_res[:] += self._operators[i, j].dot(local_x)
                col_dim += self._cols[j]
            row_dim += self._rows[i]
        if expand_input_dim:
            return res.ravel()
        else:
            return res

    def _get_row_dimensions(self):
        return self._rows

    def _get_column_dimensions(self):
        return self._cols

    def _as_matrix(self):
        rows = []
        for i in range(self.ndims[0]):
            row = []
            for j in range(self.ndims[1]):
                row.append(self[i, j].as_matrix())
            rows.append(_np.hstack(row))
        return _np.vstack(rows)

    def _transpose(self):
        """Implement the transpose."""
        raise NotImplementedError()

    def _adjoint(self):
        """Implement the adjoint."""
        raise NotImplementedError()

    row_dimensions = property(_get_row_dimensions)
    column_dimensions = property(_get_column_dimensions)


class BlockedDiscreteOperatorSum(BlockedDiscreteOperatorBase):
    """Sum of two blocked discrete operators."""

    def __init__(self, op1, op2):
        """Construct a blocked discrete operator sum."""
        if _np.any(op1.row_dimensions != op2.row_dimensions):
            raise ValueError(
                "Incompatible row dimensions. {0} != {1}".format(
                    op1.row_dimensions, op2.row_dimensions
                )
            )

        if _np.any(op1.column_dimensions != op2.column_dimensions):
            raise ValueError(
                "Incompatible column dimensions. {0} != {1}".format(
                    op1.column_dimensions, op2.column_dimensions
                )
            )

        self._op1 = op1
        self._op2 = op2

        from bempp.api.utils.data_types import combined_type

        super(BlockedDiscreteOperatorSum, self).__init__(
            op1.ndims[0],
            op1.ndims[1],
            combined_type(op1.dtype, op2.dtype),
            (op1.shape[0], op1.shape[1]),
        )

    def _matvec(self, x):
        return self._op1 * x + self._op2 * x

    def __getitem__(self, key):
        """Return item (i, j)."""
        return self._op1[key] + self._op2[key]

    def _transpose(self):
        """Transpose of sum."""
        raise NotImplementedError()

    def _adjoint(self):
        """Adjoint of sum."""
        raise NotImplementedError()

    @property
    def row_dimensions(self):
        """Return the row dimensions."""
        return self._op1.row_dimensions

    @property
    def column_dimensions(self):
        """Return the column dimensions."""
        return self._op1.column_dimensions


class BlockedDiscreteOperatorProduct(BlockedDiscreteOperatorBase):
    """Product of two blocked discrete operators."""

    def __init__(self, op1, op2):
        """Construct a product of the form op1 * op2."""
        if _np.any(op1.column_dimensions != op2.row_dimensions):
            raise ValueError(
                "Incompatible dimensions. {0} != {1}".format(
                    op1.column_dimensions, op2.row_dimensions
                )
            )

        self._op1 = op1
        self._op2 = op2

        from bempp.api.utils.data_types import combined_type

        super(BlockedDiscreteOperatorProduct, self).__init__(
            op1.ndims[0],
            op2.ndims[1],
            combined_type(op1.dtype, op2.dtype),
            (op1.shape[0], op2.shape[1]),
        )

    def _matvec(self, x):
        return self._op1 * (self._op2 * x)

    def __getitem__(self, key):
        """Return the item at position (i, j) of the product."""
        from bempp.api.assembly.discrete_boundary_operator import (
            ZeroDiscreteBoundaryOperator,
        )

        i = key[0]
        j = key[1]

        res = ZeroDiscreteBoundaryOperator(
            self.row_dimensions[i], self.column_dimensions[j]
        )

        for k in range(self._op1.ndims[1]):
            res += self._op1[i, k] * self._op2[k, j]

        return res

    def _adjoint(self):
        """Implement the product adjoint."""
        raise NotImplementedError()

    def _transpose(self):
        """Implement the product transpose."""
        raise NotImplementedError()

    @property
    def row_dimensions(self):
        """Return the row dimensions."""
        return self._op1.row_dimensions

    @property
    def column_dimensions(self):
        """Return the column dimensions."""
        return self._op2.column_dimensions


class BlockedScaledDiscreteOperator(BlockedDiscreteOperatorBase):
    """Scalar multiplication of a discrete blocked operator."""

    def __init__(self, op, alpha):
        """Construct the scaled discrete blocked operator op * alpha."""
        self._op = op
        self._alpha = alpha

        if _np.iscomplex(alpha):
            dtype = _np.dtype("complex128")
        else:
            dtype = op.dtype

        super(BlockedScaledDiscreteOperator, self).__init__(
            op.ndims[0], op.ndims[1], dtype, (op.shape[0], op.shape[1])
        )

    def _matvec(self, x):
        return self._alpha * (self._op * x)

    def __getitem__(self, key):
        """Return the item at position (i, j)."""
        return self._op[key] * self._alpha

    def _adjoint(self):
        """Implement the scaled adjoint."""
        raise NotImplementedError()

    def _transpose(self):
        """Implement the scaled transpose."""
        raise NotImplementedError()

    @property
    def row_dimensions(self):
        """Return the row dimensions."""
        return self._op.row_dimensions

    @property
    def column_dimensions(self):
        """Return the column dimensions."""
        return self._op.column_dimensions


# pylint: disable=invalid-name
def coefficients_of_grid_function_list(grid_funs):
    """
    Return a vector of coefficients of a list of grid functions.

    Given a list [f0, f1, f2, ...] this function returns a
    single Numpy vector containing [f1.coefficients, f2.coefficients, ...].

    Parameters
    ----------
    grid_funs : list of GridFunction objects
        A list containing the grid functions
    """
    vec_len = 0
    input_type = _np.dtype("float64")
    for item in grid_funs:
        input_type = _np.promote_types(input_type, item.coefficients.dtype)
        vec_len += item.space.global_dof_count
    res = _np.zeros(vec_len, dtype=input_type)
    pos = 0
    for item in grid_funs:
        dof_count = item.space.global_dof_count
        res[pos : pos + dof_count] = item.coefficients
        pos += dof_count
    return res


def projections_of_grid_function_list(grid_funs, projection_spaces):
    """
    Return a vector of projections of a list of grid functions.

    Given a list [f0, f1, f2, ...] this function returns a
    single Numpy vector containing
    [f0.projections(projection_spaces[0]),
     f1.projections(projection_spaces[1])],
     ...].

    Parameters
    ----------
    grid_funs : list of GridFunction objects
        A list containing the grid functions
    projection_spaces : list of projection spaces
        A list of projection spaces. Must have the same
        length as grid_funs.
    """
    projections = []
    for item, proj_space in zip(grid_funs, projection_spaces):
        projections.append(item.projections(proj_space))
    vec_len = 0
    input_type = _np.dtype("float64")
    for item in projections:
        input_type = _np.promote_types(input_type, item.dtype)
        vec_len += len(item)
    res = _np.zeros(vec_len, dtype=input_type)
    pos = 0
    for item in projections:
        dof_count = len(item)
        res[pos : pos + dof_count] = item
        pos += dof_count
    return res


# pylint: disable=invalid-name
def grid_function_list_from_coefficients(coefficients, spaces):
    """
    Create a list of grid functions from a long vector of coefficients.

    Parameters
    ----------
    coefficients : np.ndarray
        One-dimensional array of coefficients
    spaces : list of Space objects
        The sum of the global dofs of the spaces must be equal to the
        length of the coefficients vector.
    """
    from bempp.api import GridFunction

    pos = 0
    res_list = []
    for space in spaces:
        dof_count = space.global_dof_count
        res_list.append(
            GridFunction(space, coefficients=coefficients[pos : pos + dof_count])
        )
        pos += dof_count
    return res_list
