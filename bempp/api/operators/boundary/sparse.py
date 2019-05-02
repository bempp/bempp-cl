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

    if domain.identifier == "rwg0" and dual_to_range.identifier == 'snc0':
        return _snc0_rwg0_identity(
                domain, range_, dual_to_range, parameters, device_interface, precision)

    if domain.identifier == "rwg0" and dual_to_range.identifier == 'rwg0':
        return _rwg0_rwg0_identity(
                domain, range_, dual_to_range, parameters, device_interface, precision)

    if not (domain.codomain_dimension == 1 and dual_to_range.codomain_dimension == 1):
        raise ValueError("domain and codomain must be scalar spaces.")

    """Assemble the L^2 identiy operator."""
    return _common.create_operator(
        "scalar_identity",
        domain,
        range_,
        dual_to_range,
        parameters,
        "sparse",
        {},
        "default_sparse",
        device_interface,
        precision,
    )


def _snc0_rwg0_identity(
    domain,
    range_,
    dual_to_range,
    parameters=None,
    device_interface=None,
    precision=None,
):
    """Assemble the SNC/RWG identiy operator."""

    if not (domain.identifier == "rwg0" and dual_to_range.identifier == "snc0"):
        raise ValueError(
            "Operator only defined for domain = 'rwg' and 'dual_to_range = 'snc"
        )

    return _common.create_operator(
        "snc0_rwg0_identity",
        domain,
        range_,
        dual_to_range,
        parameters,
        "sparse",
        {},
        "default_sparse",
        device_interface,
        precision,
    )

def _rwg0_rwg0_identity(
        domain,
        range_,
        dual_to_range,
        parameters=None,
        device_interface=None,
        precision=None,
):
    """Assemble the RWG/RWG identiy operator."""

    if not (domain.identifier == "rwg0" and dual_to_range.identifier == "rwg0"):
        raise ValueError(
            "Operator only defined for domain = 'rwg0' and 'dual_to_range = 'rwg0"
        )

    return _common.create_operator(
        "rwg0_rwg0_identity",
        domain,
        range_,
        dual_to_range,
        parameters,
        "sparse",
        {},
        "default_sparse",
        device_interface,
        precision,
    )

def multitrace_identity(multitrace_operator, parameters=None, device_interface=None, precision=None):
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
    range0, range1 = multitrace_operator.domain_spaces

    blocked_operator = BlockedOperator(2, 2)

    blocked_operator[0, 0] = identity(domain0, range0, dual_to_range0)
    blocked_operator[1, 1] = identity(domain1, range1, dual_to_range1)

    return blocked_operator
    

