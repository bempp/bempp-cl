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

    if domain.identifier == "rwg0":
        return _maxwell_identity(
            domain, range_, dual_to_range, parameters, device_interface, precision
        )

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


def _maxwell_identity(
    domain,
    range_,
    dual_to_range,
    parameters=None,
    device_interface=None,
    precision=None,
):
    """Assemble the twisted Maxwell identiy operator."""

    if not (domain.identifier == "rwg0" and dual_to_range.identifier == "snc0"):
        raise ValueError(
            "Operator only defined for domain = 'rwg' and 'dual_to_range = 'snc"
        )

    return _common.create_operator(
        "maxwell_identity",
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
