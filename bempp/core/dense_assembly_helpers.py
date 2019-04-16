"""Helper routines for dense assembly."""


def choose_source_name(compute_kernel):
    """Choose source name from identifier."""

    if compute_kernel == "default_scalar":
        return "evaluate_dense"
    if compute_kernel == "maxwell_electric_field":
        return "evaluate_dense_electric_field"
    if compute_kernel == "maxwell_magnetic_field":
        return "evaluate_dense_magnetic_field"
    if compute_kernel == "laplace_hypersingular":
        return "evaluate_dense_laplace_hypersingular"
    if compute_kernel == "helmholtz_hypersingular":
        return "evaluate_dense_helmholtz_hypersingular"

    raise ValueError("Unknown compute kernel identifier.")

def choose_source_name_dense_evaluator(compute_kernel):
    """Choose source name from identifier."""

    if compute_kernel == "default_scalar":
        return "evaluate_dense_vector"
    if compute_kernel == "maxwell_electric_field":
        return "evaluate_dense_vector_electric_field"
    if compute_kernel == "maxwell_magnetic_field":
        return "evaluate_dense_vector_magnetic_field"
    if compute_kernel == "laplace_hypersingular":
        return "evaluate_dense_vector_laplace_hypersingular"
    if compute_kernel == "helmholtz_hypersingular":
        return "evaluate_dense_vector_helmholtz_hypersingular"

    raise ValueError("Unknown compute kernel identifier.")

def choose_source_name_dense_multitrace_evaluator(compute_kernel):
    """Choose source name from identifier."""

    if compute_kernel == "maxwell_multitrace":
        return "evaluate_dense_vector_maxwell_multitrace"

    raise ValueError("Unknown compute kernel identifier.")
