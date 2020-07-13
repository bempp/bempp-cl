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
    # # BC spaces use same code as RWG spaces.
    # # RBC spaces use same code as SNC spaces.
    # maxwell_modifier = {"bc": "rwg0", "rwg0": "rwg0", "rbc": "snc0", "snc0": "snc0"}

    # identity_dispatch = {
        # ("snc0", "rwg0"): _snc0_rwg0_identity,
        # ("rwg0", "snc0"): _rwg0_snc0_identity,
        # ("rwg0", "rwg0"): _rwg0_rwg0_identity,
        # ("snc0", "snc0"): _snc0_snc0_identity,
    # }

    # domain_identifier = domain.identifier
    # dual_to_range_identifier = dual_to_range.identifier

    # if domain_identifier in maxwell_modifier.keys():
        # domain_identifier = maxwell_modifier[domain_identifier]

    # if dual_to_range_identifier in maxwell_modifier.keys():
        # dual_to_range_identifier = maxwell_modifier[dual_to_range_identifier]

    # if (dual_to_range_identifier, domain_identifier) in identity_dispatch.keys():
        # return identity_dispatch[(dual_to_range_identifier, domain_identifier)](
            # domain, range_, dual_to_range, parameters, device_interface, precision
        # )

    # if not (domain.codomain_dimension == 1 and dual_to_range.codomain_dimension == 1):
        # raise ValueError("domain and codomain must be scalar spaces.")

    """Assemble the L^2 identiy operator."""
    return _common.create_operator(
        "l2_identity",
        domain,
        range_,
        dual_to_range,
        parameters,
        "sparse",
        [],
        'l2_identity',
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
