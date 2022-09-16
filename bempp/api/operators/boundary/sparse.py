"""Interfaces to Laplace operators."""
from bempp.api.operators.boundary import common as _common


def identity(
    domain,
    range_,
    dual_to_range,
    parameters=None,
    device_interface=None,
    precision=None,
):
    """Assemble the L^2 identity operator."""
    return _common.create_operator(
        "l2_identity",
        domain,
        range_,
        dual_to_range,
        parameters,
        "sparse",
        [],
        "l2_identity",
        "default_sparse",
        device_interface,
        precision,
        False,
    )


def grad_identity(
    domain,
    range_,
    dual_to_range,
    parameters=None,
    device_interface=None,
    precision=None,
):
    """Assemble the L^2 identity operator."""
    return _common.create_operator(
        "l2_grad_identity",
        domain,
        range_,
        dual_to_range,
        parameters,
        "sparse",
        [],
        "l2_grad_identity",
        "default_sparse",
        device_interface,
        precision,
        False,
    )


def curl_curl_identity(
    domain,
    range_,
    dual_to_range,
    parameters=None,
    device_interface=None,
    precision=None,
):
    """Assemble the L^2 identity operator."""
    return _common.create_operator(
        "l2_curl_curl_identity",
        domain,
        range_,
        dual_to_range,
        parameters,
        "sparse",
        [],
        "l2_curl_curl_identity",
        "default_sparse",
        device_interface,
        precision,
        False,
    )


def multitrace_identity(
    multitrace_operator, parameters=None, device_interface=None, precision=None
):
    """
    Create a multitrace identity operator.

    Parameters
    ----------
    multitrace_operator : Bempp Operator
        A 2 x 2 multitrace operator object whose spaces are used to define
        the identity operator.

    Output
    ------
    A block-diagonal multitrace identity operator.

    """
    from bempp.api.assembly.blocked_operator import BlockedOperator

    domain0, domain1 = multitrace_operator.domain_spaces
    dual_to_range0, dual_to_range1 = multitrace_operator.dual_to_range_spaces
    range0, range1 = multitrace_operator.range_spaces

    blocked_operator = BlockedOperator(2, 2)

    blocked_operator[0, 0] = identity(domain0, range0, dual_to_range0)
    blocked_operator[1, 1] = identity(domain1, range1, dual_to_range1)

    return blocked_operator


def mte_operators(domains_, ranges_, dual_to_ranges_, kappa):
    """
    Create MTE operators.

    Parameters
    ----------
    domains_
        Domain spaces
    ranges_
        Range spaces
    dual_to_ranges_
        Dual spaces

    Output
    ------
    MTE operators.

    """
    IP = identity(domains_[1], ranges_[1], dual_to_ranges_[1])
    IC = identity(domains_[0], ranges_[0], dual_to_ranges_[0])
    N = (1.0 / kappa) ** 2 * curl_curl_identity(domains_[0], ranges_[0], dual_to_ranges_[0])
    LT = grad_identity(domains_[1], ranges_[0], dual_to_ranges_[0])
    L = LT._transpose(LT._domain)
    return IP, IC, N, LT, L


def mte_lambda_1i(mte_operators, beta, kappa):
    """
    Crate the MTE lambda 1i operator.

    TODO: document this
    """
    from scipy.sparse import bmat
    from bempp.api.assembly.discrete_boundary_operator import InverseSparseDiscreteBoundaryOperator
    IP, IC, N, LT, L = mte_operators
    return InverseSparseDiscreteBoundaryOperator(
        bmat([[(IC - beta * N).weak_form().to_sparse(), (beta * LT).weak_form().to_sparse()], [L.weak_form().to_sparse(), kappa ** 2 * IP.weak_form().to_sparse()]], 'csc'))


def mte_lambda_2(mte_operators):
    """
    Crate the MTE lambda 2 operator.

    TODO: document this
    """
    IP, IC, N, LT, L = mte_operators
    return IC - N


def sigma_identity(
    domain,
    range_,
    dual_to_range,
    parameters=None,
    device_interface=None,
    precision=None,
):
    """
    Evaluate the sigma identity operator.

    For Galerkin methods this operator is equivalent to .5 * identity. For
    collocation methods the value may differ from .5 on piecewise smooth
    domains.
    """
    from bempp.api.utils.helpers import assign_parameters

    parameters = assign_parameters(parameters)

    if parameters.assembly.discretization_type == "galerkin":
        return 0.5 * identity(
            domain,
            range_,
            dual_to_range,
            parameters=parameters,
            device_interface=device_interface,
            precision=precision,
        )
    elif parameters.assembly.discretization_type == "collocation":
        raise ValueError("Not yet implemented.")


def laplace_beltrami(
    domain,
    range_,
    dual_to_range,
    parameters=None,
    device_interface=None,
    precision=None,
):
    """Assemble the negative Laplace-Beltrami operator."""
    if domain.shapeset.identifier != "p1_discontinuous":
        raise ValueError("Domain shapeset must be of type 'p1_discontinuous'.")

    if dual_to_range.shapeset.identifier != "p1_discontinuous":
        raise ValueError("Dual to range shapeset must be of type 'p1_discontinuous'.")

    return _common.create_operator(
        "laplace_beltrami",
        domain,
        range_,
        dual_to_range,
        parameters,
        "sparse",
        [],
        "laplace_beltrami",
        "default_sparse",
        device_interface,
        precision,
        False,
    )
