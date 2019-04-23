# Script to be run with legacy Bempp to generate the comparison data.

import bempp.api
import numpy as np

bempp.api.enable_console_logging()

parameters = bempp.api.global_parameters
parameters.assembly.boundary_operator_assembly_type = "dense"
parameters.assembly.potential_operator_assembly_type = "dense"

regular_order = 4
singular_order = 6

parameters.quadrature.near.double_order = regular_order
parameters.quadrature.near.single_order = regular_order
parameters.quadrature.medium.double_order = regular_order
parameters.quadrature.medium.single_order = regular_order 
parameters.quadrature.far.double_order = regular_order
parameters.quadrature.far.single_order = regular_order

parameters.quadrature.double_singular = singular_order

grid = bempp.api.import_grid("sphere.msh")
grid_structured = bempp.api.import_grid("structured_grid.msh")

p0 = bempp.api.function_space(grid, "DP", 0)
dp1 = bempp.api.function_space(grid, "DP", 1)
p1 = bempp.api.function_space(grid, "P", 1)
rwg = bempp.api.function_space(grid, "RWG", 0)
snc = bempp.api.function_space(grid, "SNC", 0)
rwg_structured = bempp.api.function_space(grid_structured, "RWG", 0)
snc_structured = bempp.api.function_space(grid_structured, "SNC", 0)


def generate_bem_matrix(dual_to_range, domain, fname, operator, wavenumber=None):
    """Generate test matrix."""
    print("Generating " + fname)

    if wavenumber is None:
        mat = operator(domain, domain, dual_to_range, use_projection_spaces=False).weak_form().A
    else:
        mat = operator(domain, domain, dual_to_range, wavenumber, use_projection_spaces=False).weak_form().A
    np.save(fname, mat)


print("Generating Laplace BEM matrices.")

generate_bem_matrix(
    p0,
    p0,
    "laplace_single_layer_boundary_p0_p0",
    bempp.api.operators.boundary.laplace.single_layer,
)

generate_bem_matrix(
    p0,
    dp1,
    "laplace_single_layer_boundary_p0_dp1",
    bempp.api.operators.boundary.laplace.single_layer,
)

generate_bem_matrix(
    dp1,
    p0,
    "laplace_single_layer_boundary_dp1_p0",
    bempp.api.operators.boundary.laplace.single_layer,
)
generate_bem_matrix(
    dp1,
    dp1,
    "laplace_single_layer_boundary_dp1_dp1",
    bempp.api.operators.boundary.laplace.single_layer,
)
generate_bem_matrix(
    p1,
    p1,
    "laplace_single_layer_boundary_p1_p1",
    bempp.api.operators.boundary.laplace.single_layer,
)

generate_bem_matrix(
    p1,
    p1,
    "laplace_double_layer_boundary",
    bempp.api.operators.boundary.laplace.double_layer,
)

generate_bem_matrix(
    p1,
    p1,
    "laplace_adj_double_layer_boundary",
    bempp.api.operators.boundary.laplace.adjoint_double_layer,
)

generate_bem_matrix(
    p1,
    p1,
    "laplace_hypersingular_boundary",
    bempp.api.operators.boundary.laplace.hypersingular,
)

################################

print("Generating Helmholtz BEM matrices.")

wavenumber = 2.5
wavenumber_complex = 2.5 + 1j

generate_bem_matrix(
    p0,
    p0,
    "helmholtz_single_layer_boundary_p0_p0",
    bempp.api.operators.boundary.helmholtz.single_layer,
    wavenumber
)

generate_bem_matrix(
    p0,
    dp1,
    "helmholtz_single_layer_boundary_p0_dp1",
    bempp.api.operators.boundary.helmholtz.single_layer,
    wavenumber
)

generate_bem_matrix(
    dp1,
    p0,
    "helmholtz_single_layer_boundary_dp1_p0",
    bempp.api.operators.boundary.helmholtz.single_layer,
    wavenumber
)
generate_bem_matrix(
    dp1,
    dp1,
    "helmholtz_single_layer_boundary_dp1_dp1",
    bempp.api.operators.boundary.helmholtz.single_layer,
    wavenumber
)
generate_bem_matrix(
    p1,
    p1,
    "helmholtz_single_layer_boundary_p1_p1",
    bempp.api.operators.boundary.helmholtz.single_layer,
    wavenumber
)

generate_bem_matrix(
    p1,
    p1,
    "helmholtz_double_layer_boundary",
    bempp.api.operators.boundary.helmholtz.double_layer,
    wavenumber
)

generate_bem_matrix(
    p1,
    p1,
    "helmholtz_adj_double_layer_boundary",
    bempp.api.operators.boundary.helmholtz.adjoint_double_layer,
    wavenumber
)

generate_bem_matrix(
    p1,
    p1,
    "helmholtz_hypersingular_boundary",
    bempp.api.operators.boundary.helmholtz.hypersingular,
    wavenumber
)

generate_bem_matrix(
    p1,
    p1,
    "helmholtz_complex_single_layer_boundary",
    bempp.api.operators.boundary.helmholtz.single_layer,
    wavenumber_complex
)

generate_bem_matrix(
    p1,
    p1,
    "helmholtz_complex_double_layer_boundary",
    bempp.api.operators.boundary.helmholtz.double_layer,
    wavenumber_complex
)
generate_bem_matrix(
    p1,
    p1,
    "helmholtz_complex_adj_double_layer_boundary",
    bempp.api.operators.boundary.helmholtz.adjoint_double_layer,
    wavenumber_complex
)

generate_bem_matrix(
    p1,
    p1,
    "helmholtz_complex_hypersingular_boundary",
    bempp.api.operators.boundary.helmholtz.hypersingular,
    wavenumber_complex
)

##################################

print("Generate modified Helmholtz BEM matrices.")

omega = 2.5

generate_bem_matrix(
    p1,
    p1,
    "modified_helmholtz_single_layer_boundary",
    bempp.api.operators.boundary.modified_helmholtz.single_layer,
    omega
)

generate_bem_matrix(
    p1,
    p1,
    "modified_helmholtz_double_layer_boundary",
    bempp.api.operators.boundary.modified_helmholtz.double_layer,
    omega
)
generate_bem_matrix(
    p1,
    p1,
    "modified_helmholtz_adj_double_layer_boundary",
    bempp.api.operators.boundary.modified_helmholtz.adjoint_double_layer,
    omega
)

generate_bem_matrix(
    p1,
    p1,
    "modified_helmholtz_hypersingular_boundary",
    bempp.api.operators.boundary.modified_helmholtz.hypersingular,
    omega
)

#####################################

print("Generate Maxwell BEM matrices.")

generate_bem_matrix(
        snc,
        rwg,
        "maxwell_electric_field_boundary",
        bempp.api.operators.boundary.maxwell.electric_field,
        wavenumber
)

generate_bem_matrix(
        snc,
        rwg,
        "maxwell_magnetic_field_boundary",
        bempp.api.operators.boundary.maxwell.magnetic_field,
        wavenumber
)

generate_bem_matrix(
        snc,
        rwg,
        "maxwell_electric_field_complex_boundary",
        bempp.api.operators.boundary.maxwell.electric_field,
        wavenumber_complex
)

generate_bem_matrix(
        snc,
        rwg,
        "maxwell_magnetic_field_complex_boundary",
        bempp.api.operators.boundary.maxwell.magnetic_field,
        wavenumber_complex
)

generate_bem_matrix(
        snc_structured,
        rwg_structured,
        "maxwell_electric_field_structured_boundary",
        bempp.api.operators.boundary.maxwell.electric_field,
        wavenumber
)
