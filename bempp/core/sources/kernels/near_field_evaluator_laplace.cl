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
	 __global REALTYPE* kernelParameters,
	 uint nelements,
	 )
{
  size_t gid = get_global_id(0);

  int indexStart = neighborIndexptr[gid];
  int indexEnd = neighborIndexptr[1 + gid];

  int nIndices = indexEnd - indexStart;
  int nChunks = nIndices / VEC_LENGTH;

  REALTYPE globalSourcePoints[3 * MAX_POINTS];
  REALTYPE3 targetCorners[3];
  REALTYPE3 sourceCorners[3];
  REALTYPE3 globalPoint;
  REALTYPEVEC sourceVecPoint[3];
  REALTYPE3 dummy;
  REALTYPE3 targetPoint;
  REALTYPE2 point;
  REALTYPEVEC coeffsVec;
  REALTYPEVEC myResultVec[4];
  
  getCorners(grid, gid, targetCorners);

  // First compute global points

  int count = 0;
  for (int i = 0; i < nIndices; ++i){
      int elem = neighborIndices[i + indexStart];
      getCorners(grid, elem, sourceCorners);
      for (int localIndex = 0; localIndex < NPOINTS; ++localIndex)
      	  {
		point = (REALTYPE2)(localPoints[2 * localIndex],
		                    localPoints[2 * localIndex + 1]);
		globalPoint = getGlobalPoint(sourceCorners, &point);
		globalSourcePoints[count] = globalPoint.x;
		globalSourcePoints[count + 1] = globalPoint.y;
		globalSourcePoints[count + 2] = globalPoint.z;
		count += 3;
	}
   }

  for (int targetIndex = 0; targetIndex < NPOINTS; targetIndex++)
    {
      point = (REALTYPE2)(localPoints[2 * targetIndex],
      		    localPoints[2 * targetIndex + 1]);

      targetPoint = getGlobalPoint(targetCorners, &point);

      for (int chunkIndex = 0; chunkIndex < nChunks; chunkIndex++){
	// Fill chunk
	for (int vecIndex = 0; vecIndex < VEC_LENGTH; vecIndex++){
	  ((REALTYPE*)&sourceVecPoint[0])[vecIndex] = globalSourcePoints[3 * (VEC_LENGTH * chunkIndex + vecIndex) + 0];
	  ((REALTYPE*)&sourceVecPoint[1])[vecIndex] = globalSourcePoints[3 * (VEC_LENGTH * chunkIndex + vecIndex) + 1];
	  ((REALTYPE*)&sourceVecPoint[2])[vecIndex] = globalSourcePoints[3 * (VEC_LENGTH * chunkIndex + vecIndex) + 2];
	  ((REALTYPE*)&coeffsVec)[vecIndex] = coefficients[VEC_LENGTH * chunkIndex + vecIndex];
	}

	diff[0] = targetPoint.x - sourceVecPoint[0];
	diff[1] = targetPoint.y - sourceVecPoint[1];
	diff[2] = targetPoint.z - sourceVecPoint[2];

	rdist = rsqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]) * M_INV_4PI;

	myResultVec[0] += rdist * coeffsVec;
	myResultVec[1] += -rdist * diff[0] * coeffsVec;
	myResultVec[2] += -rdist * diff[1] * coeffsVec;
	myResultVec[3] += -rdist * diff[2] * coeffsVec;

	

	
	
      }
    }
  
}      
      


