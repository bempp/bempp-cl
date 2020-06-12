#include "bempp_base_types.h"
#include "bempp_helpers.h"
#include "bempp_spaces.h"
#include "kernels.h"

__inline__ void(

__kernel __attribute__((vec_type_hint(REALTYPEVEC))) void kernel_function(
	 __global REALTYPE* grid,
	 __global int* neighborIndices,
	 __global int* neighborIndexptr,
	 __global REALTYPE* localPoints,
	 __global REALTYPE* coefficients,
	 __global REALTYPE* result,
	 __global REALTYPE* kernelParameters,
	 uint nelements,
	 uint npoints
	 )
{
  size_t gid = get_global_id(0);

  int indexStart = neighborIndexptr[gid];
  int indexEnd = neighborIndexptr[1 + gid];

  int nIndices = indexEnd - indexStart;
  int nChunks = nIndices / VEC_LENGTH;

  REALTYPE globalSourcePoints[MAX_POINTS][3];
  REALTYPE3 targetCorners[3];
  REALTYEP3 sourceCorners[3];
  REALTYPE3 globalPoint;
  REALTYPEVEC sourceVecPoint[3];
  REALTYPE3 dummy;
  REALTYPE3 targetPoint;
  REALTYPE2 point;
  REALTYPEVEC* kernelValuesVec;

  getCorners(grid, gid, targetCorners);

  // First compute global points

  int count = 0;
  for (int i = 0; i < nIndices; ++i){
      int elem = neighborIndices[i + indexStart];
      getCorners(grid, elem, sourceCorners);
      for (int localIndex = 0; localIndex < npoints; ++localIndex)
      	  {
		point = (REALTYPE2)(localPoints[2 * localIndex],
		                    localPoints[2 * localIndex + 1]);
		globalPoint = getGlobalPoint(sourceCorners, &point);
		globalSourcePoints[count][0] = globalPoint.x;
		globalSourcePoints[count][1] = globalPoint.y;
		globalSourcePoints[count][2] = globalPoint.z;
		count += 3;
	}
   }

  for (int targetIndex = 0; targetIndex < npoints; targetIndex++)
    {
      point = (REALTYPE2)(localPoints[2 * targetIndex],
      		    localPoints[2 * targetIndex + 1]);

      for (int chunkIndex = 0; chunkIndex < nChunks ; chunkIndex++){
	

      }

	
      targetPoint = getGlobalPoint(targetCorners, &point);
      KERNEL(VEC_STRING)(targetPoint, 0, dummy, 0, kernelParameters, kernelValuesVec);
      
  
}      
      


