#include "bempp_base_types.h"
#include "bempp_helpers.h"
#include "bempp_spaces.h"
#include "kernels.h"

__kernel __attribute__((vec_type_hint(REALTYPEVEC))) void kernel_function(
	 __global REALTYPE* grid,
	 __global int* neighborIndices,
	 __global int* neighborIndexptr,
	 __global REALTYPE* localPoints,
	 __global REALTYPE* coefficients,
	 __global REALTYPE* result,
	 uint nelements,
	 uint npoints
	 )
{
  size_t gid = get_global_id(0);

}
