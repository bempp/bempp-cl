"""Bempp direct solver interface."""

# pylint: disable=invalid-name

def compute_lu_factors(A):
    """
    Precompute the LU factors of a dense operator A.

    This function returns a tuple of LU factors of A.
    This tuple can be used in the `lu_factor` attribute
    of the lu function so that the LU decomposition is
    not recomputed at each call to lu.

    """
    from scipy.linalg import lu_factor
    return lu_factor(bempp.api.as_matrix(A.weak_form()))


def lu(A, b, lu_factor=None):
    """Simple direct solver interface.

    This function takes an operator and a grid function,
    converts the operator into a dense matrix and solves
    the system via LU decomposition. The result is again
    returned as a grid function.

    Parameters
    ----------
    A : bempp.api.BoundaryOperator
         The left-hand side boundary operator
    b : bempp.api.GridFunction
         The right-hand side grid function
    lu_decomp : tuple
         Optionally pass the tuple (lu, piv)
         obtained by the scipy method scipy.linalg.lu_factor

    """
    from bempp.api import GridFunction, as_matrix
    from scipy.linalg import solve, lu_solve

    if lu_factor is not None:
        vec = b.projections(A.dual_to_range)
        sol = lu_solve(lu_factor, vec)
    else:
        mat = as_matrix(A.weak_form())
        vec = b.projections(A.dual_to_range)
        sol = solve(mat, vec)
    return GridFunction(A.domain, coefficients=sol)
