#include "bempp_base_types.h"
#include "bempp_helpers.h"
#include "bempp_spaces.h"
#include "kernels.h"

__kernel void evaluate_near_field(
        __global long *targetIds,
        __global long *sourceIndexPtr,
        __global long *targetIndexPtr,
        __global uint *sourceElements,
        __global uint *targetElements,
        __global REALTYPE *sourceVertices,
        __global REALTYPE *targetVertices,
        __global REALTYPE *input,
        __global REALTYPE *result
        )
{
   int localId = get_local_id(0);
   int groupId = get_group_id(0);

   int targetIndexStart = targetIndexPtr[groupId];
   int targetIndexEnd = targetIndexPtr[groupId + 1];
   int numberOfTargets = targetIndexEnd - targetIndexStart;

   int sourceIndexStart = sourceIndexPtr[groupId];
   int sourceIndexEnd = sourceIndexPtr[groupId + 1];
   int numberOfSources = sourceIndexEnd - sourceIndexStart;

}
