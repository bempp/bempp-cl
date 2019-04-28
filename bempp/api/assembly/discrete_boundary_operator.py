"""Data structures for assembled boundary operators."""

import numpy as _np
from scipy.sparse.linalg.interface import LinearOperator as _LinearOperator

# Disable warnings for differing overridden parameters
# pylint: disable=W0221


class DiscreteBoundaryOperator(_LinearOperator):
    """Base class for discrete boundary operators."""

    def __new__(cls, *args, **kwargs):
        """Overwrite new operator."""
        # Overwriting new because LinearOperator calls __init__
        # unnecessarily in its __new__ method causing doubly
        # called constructors (to be fixed in 0.18)
        return object.__new__(cls)

    def __init__(self, dtype, shape):
        """Constructor for discrete boundary operator."""
        import scipy

        if scipy.__version__ < "0.16.0":
            super(DiscreteBoundaryOperator, self).__init__(
                shape,
                self._matvec,
                rmatvec=self._rmatvec,
                matmat=self._matmat,
                dtype=dtype,
            )
        else:
            super(DiscreteBoundaryOperator, self).__init__(dtype, shape)

    def __add__(self, other):
        """Add two discrete boundary operators."""
        if isinstance(other, DiscreteBoundaryOperator):
            return DiscreteBoundaryOperatorSum(self, other)
        else:
            return super(DiscreteBoundaryOperator, self).__add__(other)

    def __mul__(self, other):
        """Multiply operator with something else."""
        return self.dot(other)

    def dot(self, other):
        """Multiply operator with something else."""
        if isinstance(other, DiscreteBoundaryOperator):
            return DiscreteBoundaryOperatorProduct(self, other)
        elif isinstance(other, _LinearOperator):
            return super(DiscreteBoundaryOperator, self).dot(other)
        elif _np.isscalar(other):
            return ScaledDiscreteBoundaryOperator(self, other)
        else:
            x_in = _np.asarray(other)
            if x_in.ndim == 1 or (x_in.ndim == 2 and x_in.shape[1] == 1):
                return self._matvec(x_in)
            elif x_in.ndim == 2:
                return self._matmat(x_in)
            else:
                raise ValueError("Expect a 1d or 2d array or matrix.")

    def __rmul__(self, other):
        """Right multiplication."""
        if _np.isscalar(other):
            return self * other
        else:
            raise ValueError(
                "Cannot multiply operand of type "
                + "{0} from the left.".format(type(other))
            )

    def __call__(self, other):
        """Apply operator."""
        return self.dot(other)

    def __matmul__(self, other):
        """Product with matrix."""
        if _np.isscalar(other):
            raise ValueError("Scalar operands not allowed. Use '*' instead.")

        return self.dot(other)

    def __neg__(self):
        """Negate operator."""
        return -1 * self

    def __sub__(self, other):
        """Subtract operator from something else."""
        return self.__add__(-other)

    def _adjoint(self):
        """Implement the adjoint."""
        raise NotImplementedError()

    def _transpose(self):
        """Implement the transpose."""
        raise NotImplementedError()

    import scipy

    if scipy.__version__ < "0.16.0":

        def adjoint(self):
            """Return the adoint."""
            return self._adjoint()

        def transpose(self):
            """Return the transpose."""
            return self._transpose()

        H = property(adjoint)
        T = property(transpose)

    def elementary_operators(self):
        """Return the elementary operators that form this operator."""
        raise NotImplementedError()

    @property
    def memory(self):
        """Return an estimate of the memory size in kb"""
        ops = self.elementary_operators()
        # pylint: disable=protected-access
        return sum([operator._memory for operator in ops])


class GenericDiscreteBoundaryOperator(DiscreteBoundaryOperator):
    """Discrete boundary operator that implements a matvec routine."""

    def __init__(self, evaluator):
        """Constructor for discrete boundary operator."""

        super(GenericDiscreteBoundaryOperator, self).__init__(
            evaluator.dtype, evaluator.shape
        )
        self._evaluator = evaluator
        self._is_complex = self.dtype == "complex128" or self.dtype == "complex64"

    @property
    def memory(self):
        """Return an estimate of the memory size in kb"""
        return 0.0

    def _matvec(self, x):
        if self._is_complex:
            return self._evaluator.matvec(x)
        if _np.iscomplexobj(x):
            return self._evaluator.matvec(_np.real(x)) + 1j * self._evaluator.matvec(
                _np.imag(x)
            )
        else:
            return self._evaluator.matvec(x)


class DiscreteBoundaryOperatorSum(DiscreteBoundaryOperator):
    """Sum of two discrete boundary operators."""

    def __init__(self, op1, op2):
        """Construct the sum of two discrete boundary operators."""
        if not isinstance(op1, DiscreteBoundaryOperator) or not isinstance(
            op2, DiscreteBoundaryOperator
        ):
            raise ValueError("Both operators must be discrete boundary operators.")

        if op1.shape != op2.shape:
            raise ValueError("Shape mismatch: {0} != {1}.".format(op1.shape, op2.shape))

        self._op1 = op1
        self._op2 = op2

        super(DiscreteBoundaryOperatorSum, self).__init__(
            _np.find_common_type([op1.dtype, op2.dtype], []), op1.shape
        )

    def _matvec(self, x):
        return self._op1.matvec(x) + self._op2.matvec(x)

    def _matmat(self, x):
        return self._op1.matmat(x) + self._op2.matmat(x)

    def _rmatvec(self, x):
        return self._op1.rmatvec(x) + self._op2.rmatvec(x)

    def _adjoint(self):
        return self._op1.adjoint() + self._op2.adjoint()

    def _transpose(self):
        return self._op1.transpose() + self._op2.transpose()

    def elementary_operators(self):
        """Return the elementary operators that make up this operator."""

        return self._op1.elementary_operators() | self._op2.elementary_operators()


class DiscreteBoundaryOperatorProduct(DiscreteBoundaryOperator):
    """Product of two discrete operators."""

    def __init__(self, op1, op2):
        """Construct the product of two discrete operators."""

        if not isinstance(op1, DiscreteBoundaryOperator) or not isinstance(
            op2, DiscreteBoundaryOperator
        ):
            raise ValueError("Both operators must be discrete boundary operators.")

        if op1.shape[1] != op2.shape[0]:
            raise ValueError(
                "Shapes {0} and {1}".format(op1.shape, op2.shape)
                + " not compatible for matrix product."
            )

        self._op1 = op1
        self._op2 = op2

        super(DiscreteBoundaryOperatorProduct, self).__init__(
            _np.find_common_type([op1.dtype, op2.dtype], []),
            (op1.shape[0], op2.shape[1]),
        )

    def _matvec(self, x):

        return self._op1.matvec(self._op2.matvec(x))

    def _matmat(self, x):

        return self._op1.matmat(self._op2.matmat(x))

    def _rmatvec(self, x):
        return self._op2.rmatvec(self._op1.rmatvec(x))

    def _adjoint(self):

        return self._op2.adjoint() * self._op1.adjoint()

    def _transpose(self):

        return self._op2.transpose() * self._op1.transpose()

    def elementary_operators(self):
        """Return the elementary operators that make up this operator."""

        return self._op1.elementary_operators() | self._op2.elementary_operators()


class ScaledDiscreteBoundaryOperator(DiscreteBoundaryOperator):
    """Scaled discrete boundary operator."""

    def __init__(self, op, alpha):
        """Construct a scaled discrete boundary operator."""

        if not isinstance(op, DiscreteBoundaryOperator):
            raise ValueError("Both operators must be discrete boundary operators.")

        self._op = op
        self._alpha = alpha

        super(ScaledDiscreteBoundaryOperator, self).__init__(
            _np.find_common_type([op.dtype, _np.array([alpha]).dtype], []), op.shape
        )

    def _matvec(self, x):
        return self._alpha * self._op.matvec(x)

    def _matmat(self, x):
        return self._alpha * self._op.matmat(x)

    def _rmatvec(self, x):
        return self._alpha * self._op.rmatvec(x)

    def _adjoint(self):
        return self._alpha * self._op.adjoint()

    def _transpose(self):
        return self._alpha * self._op.transpose()

    def elementary_operators(self):
        """Return the elementary operators that make up this operator."""
        return self._op.elementary_operators()


class DenseDiscreteBoundaryOperator(DiscreteBoundaryOperator):
    """
    Main class for the discrete form of dense nonlocal operators.

    This class derives from
    :class:`scipy.sparse.linalg.interface.LinearOperator`
    and thereby implements the SciPy LinearOperator protocol.

    """

    def __init__(self, impl):
        """Constructor. Should not be called by the user."""
        self._impl = impl
        super(DenseDiscreteBoundaryOperator, self).__init__(impl.dtype, impl.shape)

    def _matvec(self, x):

        if _np.iscomplexobj(x) and not _np.iscomplexobj(self.A):
            return self.A.dot(_np.real(x).astype(self.dtype)) + \
                    1j * self.A.dot(_np.imag(x).astype(self.dtype))
        return self.A.dot(x.astype(self.dtype))

    def __add__(self, other):
        if isinstance(other, DenseDiscreteBoundaryOperator):
            return DenseDiscreteBoundaryOperator(self.A + other.A)
        else:
            return super(DenseDiscreteBoundaryOperator, self).__add__(other)

    def __neg__(self):
        return DenseDiscreteBoundaryOperator(-self.A)

    def __mul__(self, other):
        return self.dot(other)

    def dot(self, other):
        """Form the product with another object."""
        if isinstance(other, DenseDiscreteBoundaryOperator):
            return DenseDiscreteBoundaryOperator(self.A.dot(other.A))
        if _np.isscalar(other):
            return DenseDiscreteBoundaryOperator(
                    self.A * _np.dtype(self.dtype).type(other))
        return super(DenseDiscreteBoundaryOperator, self).dot(other)

    def __rmul__(self, other):
        if _np.isscalar(other):
            return DenseDiscreteBoundaryOperator(self.A * other)
        else:
            return NotImplemented

    def _transpose(self):
        """Transpose of the operator."""
        return DenseDiscreteBoundaryOperator(self.A.T)

    def _adjoint(self):
        """Adjoint of the operator."""
        return DenseDiscreteBoundaryOperator(self.A.conjugate().transpose())

    # pylint: disable=invalid-name
    @property
    def A(self):
        """Return the underlying array."""
        return self._impl

    @property
    def _memory(self):
        return self.A.nbytes / 1024.0

    def elementary_operators(self):
        """Return the elementary operators that make up this operator."""
        return {self}


class SparseDiscreteBoundaryOperator(DiscreteBoundaryOperator):
    """
    Main class for the discrete form of sparse operators.

    This class derives from
    :class:`scipy.sparse.linalg.interface.LinearOperator`
    and thereby implements the SciPy LinearOperator protocol.

    """

    def __init__(self, impl):
        """Constructor. Should not e called by the user."""
        super(SparseDiscreteBoundaryOperator, self).__init__(impl.dtype, impl.shape)
        self._impl = impl
        self._adjoint_impl = None

    def _matvec(self, vec):
        """Multiply the operator with a numpy vector or matrix x."""
        if self.dtype == "float64" and _np.iscomplexobj(vec):
            return self.A * _np.real(vec) + 1j * (self.A * _np.imag(vec))
        return self.A * vec

    def _matmat(self, mat):
        """Multiply operator with the dense numpy matrix mat."""
        return self._matvec(mat)

    def _transpose(self):
        """Return the transpose of the discrete operator."""
        return SparseDiscreteBoundaryOperator(self.A.transpose())

    def _rmatvec(self, x):
        if self._adjoint_impl is None:
            self._adjoint_impl = self.A.adjoint()
        return self._adjoint_impl * x

    def _adjoint(self):
        """Return the adjoint of the discrete operator."""
        return SparseDiscreteBoundaryOperator(self.A.transpose().conjugate())

    def __add__(self, other):
        if isinstance(other, SparseDiscreteBoundaryOperator):
            return SparseDiscreteBoundaryOperator(self.A + other.A)
        else:
            return super(SparseDiscreteBoundaryOperator, self).__add__(other)

    def __neg__(self):
        return SparseDiscreteBoundaryOperator(-self.A)

    def __mul__(self, other):
        if isinstance(other, SparseDiscreteBoundaryOperator):
            return SparseDiscreteBoundaryOperator(self.A * other.A)
        else:
            return self.dot(other)

    def dot(self, other):
        if isinstance(other, SparseDiscreteBoundaryOperator):
            return SparseDiscreteBoundaryOperator(self.A * other.A)
        if _np.isscalar(other):
            return SparseDiscreteBoundaryOperator(self.A * other)

        return super(SparseDiscreteBoundaryOperator, self).dot(other)

    def __rmul__(self, other):
        if _np.isscalar(other):
            return SparseDiscreteBoundaryOperator(self.A * other)
        else:
            return NotImplemented

    @property
    def A(self):
        """Return the underlying Scipy sparse matrix."""
        return self._impl

    @property
    def _memory(self):
        mat = self.A
        return (mat.data.nbytes + mat.indices.nbytes + mat.indptr.nbytes) // 1024

    def elementary_operators(self):
        """Return the elementary operators that make up this operator."""
        return {self}


class InverseSparseDiscreteBoundaryOperator(DiscreteBoundaryOperator):
    """
    Apply the (pseudo-)inverse of a sparse operator.

    This class uses a Sparse LU-Decomposition
    (in the case of a square matrix) or a sparse normal
    equation to provide the application of an inverse to
    a sparse operator.

    This class derives from
    :class:`scipy.sparse.linalg.interface.LinearOperator`
    and thereby implements the SciPy LinearOperator protocol.

    Parameters
    ----------
    operator : bempp.api.SparseDiscreteBoundaryOperator
        Sparse operator to be inverted.

    """

    def __init__(self, operator):

        self._solver = _Solver(operator)
        self._adjoint_op = None
        self._operator = operator
        super(InverseSparseDiscreteBoundaryOperator, self).__init__(
            self._solver.dtype, self._solver.shape
        )

    def _matvec(self, vec):
        """Implemententation of matvec."""

        return self._solver.solve(vec)

    def _rmatvec(self, vec):
        """Implemententation of rmatvec."""
        # pylint: disable=protected-access
        if self._adjoint_op is None:
            self._adjoint_op = self.adjoint()
        return self._adjoint_op * vec

    def _transpose(self):
        return InverseSparseDiscreteBoundaryOperator(self._operator.transpose())

    def _adjoint(self):
        return InverseSparseDiscreteBoundaryOperator(self._operator.adjoint())

    def elementary_operators(self):
        """Return the elementary operators that make up this operator."""
        return self._operator.elementary_operators()


class ZeroDiscreteBoundaryOperator(DiscreteBoundaryOperator):
    """A discrete operator that represents a zero operator.

    This class derives from
    :class:`scipy.sparse.linalg.interface.LinearOperator`
    and thereby implements the SciPy LinearOperator protocol.

    Parameters
    ----------
    rows : int
        The number of rows in the operator.
    columns : int
        The number of columns in the operator.

    """

    def __init__(self, rows, columns):
        """Construct a zero operator."""
        super(ZeroDiscreteBoundaryOperator, self).__init__(
            _np.dtype("float64"), (rows, columns)
        )

    def _matvec(self, x):
        if x.ndim > 1:
            return _np.zeros((self.shape[0], x.shape[1]), dtype="float64")
        else:
            return _np.zeros(self.shape[0], dtype="float64")

    def _rmatvec(self, x):
        if x.ndim > 1:
            return _np.zeros((x.shape[0], self.shape[1]), dtype="float64")
        else:
            return _np.zeros(self.shape[1], dtype="float64")

    def _adjoint(self):
        raise NotImplementedError()

    def _transpose(self):
        raise NotImplementedError()

    def elementary_operators(self):
        """Return the elementary operators that make up this operator."""
        return {}


class DiscreteRankOneOperator(DiscreteBoundaryOperator):
    """Creates a discrete rank one operator.

    This class represents a rank one operator given
    by column * row, where column is column is
    interpreted as a (m, 1) array and row as
    a (1, n) array.

    Parameters
    ----------
    column : np.array
        A column vector
    row : np.array
        A row vector

    """

    def __init__(self, column, row):
        """Construct a discrete rank one operator."""
        if row.dtype == "complex128" or column.dtype == "complex128":
            dtype = "complex128"
        else:
            dtype = "float64"

        self._row = row.ravel()
        self._column = column.ravel()

        shape = (len(self._column), len(self._row))
        super(DiscreteRankOneOperator, self).__init__(dtype, shape)

    def _matvec(self, x):
        if x.ndim > 1:
            return _np.outer(self._column, _np.dot(self._row, x))
        else:
            return self._column * _np.dot(self._row, x)

    def _rmatvec(self, x):
        # pylint: disable=protected-access
        return self._adjoint()._matvec(x)

    def _transpoe(self):
        return DiscreteRankOneOperator(self._row, self._column)

    def _adjoint(self):
        return DiscreteRankOneOperator(self._row.conjugate(), self._column.conjugate())

    def elementary_operators(self):
        """Return the elementary operators that make up this operator."""
        return {}


def as_matrix(operator):
    """
    Convert a discrte operator into a dense matrix.

    Parameters
    ----------
    operator : scipy.sparse.linalg.interface.LinearOperator
        The linear operator to be converted into a dense matrix.


    Notes
    -----
    Note that this function may be slow depending on how the original
    discrete operator was stored. In the case of a dense assembly simple
    the underlying NumPy matrix is returned. Otherwise, the operator needs
    to be converted to an array, which can take a long time.

    """
    from numpy import eye

    cols = operator.shape[1]
    if isinstance(operator, DenseDiscreteBoundaryOperator):
        return operator.A
    elif isinstance(operator, SparseDiscreteBoundaryOperator):
        return operator.sparse_matrix
    else:
        return operator * eye(cols, cols)


class _Solver(object):  # pylint: disable=too-few-public-methods
    """Actual solve of a sparse linear system."""

    # pylint: disable=too-many-locals
    def __init__(self, operator):

        from scipy.sparse import csc_matrix

        if isinstance(operator, SparseDiscreteBoundaryOperator):
            mat = operator.A
        elif isinstance(operator, csc_matrix):
            mat = operator
        else:
            raise ValueError(
                "op must be either of type "
                + "SparseDiscreteBoundaryOperator or of type "
                + "csc_matrix. Actual type: "
                + str(type(operator))
            )

        from scipy.sparse.linalg import splu

        self._solve_fun = None
        self._shape = (mat.shape[1], mat.shape[0])
        self._dtype = mat.dtype

        import time
        import bempp.api

        use_mkl_pardiso = False

        # pylint: disable=bare-except
        try:
            # pylint: disable=E0401
            from mkl_pardiso_solve import PardisoInterface

            # pylint: disable=invalid-name
            solver_interface = PardisoInterface
            actual_mat = mat.tocsr()
            use_mkl_pardiso = True
        except:
            solver_interface = splu
            actual_mat = mat

        bempp.api.log(
            "Start computing LU "
            + "(pseudo)-inverse of ({0}, {1}) matrix.".format(
                mat.shape[0], mat.shape[1]
            )
        )

        start_time = time.time()
        if mat.shape[0] == mat.shape[1]:
            # Square matrix case
            solver = solver_interface(actual_mat)
            self._solve_fun = solver.solve
        elif mat.shape[0] > mat.shape[1]:
            # Thin matrix case
            mat_hermitian = actual_mat.conjugate().transpose()
            if use_mkl_pardiso:
                solver = solver_interface((mat_hermitian * mat).tocsr())
            else:
                solver = solver_interface((mat_hermitian * mat).tocsc())
            self._solve_fun = lambda x: solver.solve(mat_hermitian * x)
        else:
            # Thick matrix case

            mat_hermitian = actual_mat.conjugate().transpose()
            if use_mkl_pardiso:
                solver = solver_interface((mat * mat_hermitian).tocsr())
            else:
                solver = solver_interface((mat * mat_hermitian).tocsc())
            self._solve_fun = lambda x: mat_hermitian * solver.solve(x)

        end_time = time.time()
        bempp.api.log(
            "Finished computation of inverse in %.2E seconds." % (end_time - start_time)
        )

    def solve(self, vec):
        """Solve with right-hand side vec."""

        if self._dtype == "float64" and _np.iscomplexobj(vec):
            return self.solve(_np.real(vec)) + 1j * self.solve(_np.imag(vec))

        result = self._solve_fun(vec.squeeze())

        if vec.ndim > 1:
            return result.reshape(self.shape[0], 1)
        else:
            return result

    @property
    def shape(self):
        """Return the shape of the inverse operator."""
        return self._shape

    @property
    def dtype(self):
        """Return the dtype."""
        return self._dtype
