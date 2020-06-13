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
	 uint nelements
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
  REALTYPEVEC diffVec[3];
  REALTYPE diff[3];
  REALTYPEVEC rdistVec;
  REALTYPE rdist;
  REALTYPE3 targetPoint;
  REALTYPE2 point;
  REALTYPEVEC coeffsVec;
  REALTYPEVEC resultVec;
  REALTYPE localResult;
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
      resultVec = M_ZERO;
      localResult = M_ZERO;
      point = (REALTYPE2)(localPoints[2 * targetIndex],
      		    localPoints[2 * targetIndex + 1]);

      targetPoint = getGlobalPoint(targetCorners, &point);

      for (int chunkIndex = 0; chunkIndex < nChunks; chunkIndex++){
	// Fill chunk
	for (int vecIndex = 0; vecIndex < VEC_LENGTH; vecIndex++){
	  VEC_ELEMENT(sourceVecPoint[0], vecIndex) = globalSourcePoints[3 * (VEC_LENGTH * chunkIndex + vecIndex) + 0];
	  VEC_ELEMENT(sourceVecPoint[1], vecIndex) = globalSourcePoints[3 * (VEC_LENGTH * chunkIndex + vecIndex) + 1];
	  VEC_ELEMENT(sourceVecPoint[2], vecIndex) = globalSourcePoints[3 * (VEC_LENGTH * chunkIndex + vecIndex) + 2];
	  ((REALTYPE*)&coeffsVec)[vecIndex] = coefficients[indexStart + VEC_LENGTH * chunkIndex + vecIndex];
	}

	diffVec[0] = targetPoint.x - sourceVecPoint[0];
	diffVec[1] = targetPoint.y - sourceVecPoint[1];
	diffVec[2] = targetPoint.z - sourceVecPoint[2];

	rdistVec = rsqrt(diffVec[0] * diffVec[0] + diffVec[1] * diffVec[1] + diffVec[2] * diffVec[2]) * M_INV_4PI;
	// Check for zero dist case
	for (int vecIndex = 0; vecIndex < VEC_LENGTH; vecIndex++){
	  if ((VEC_ELEMENT(diffVec[0], vecIndex) == M_ZERO) && (VEC_ELEMENT(diffVec[1], vecIndex) == M_ZERO) && (VEC_ELEMENT(diffVec[2], vecIndex) == M_ZERO))
	    VEC_ELEMENT(rdistVec, vecIndex) = M_ZERO;
	}

	resultVec += rdistVec * coeffsVec;
	
      }

      // Now process the remainder scalar points
      for (int remainderIndex = nChunks * VEC_LENGTH; remainderIndex < nIndices; remainderIndex++)
	{
	  diff[0] = targetPoint.x - globalSourcePoints[3 * remainderIndex + 0];
	  diff[1] = targetPoint.y - globalSourcePoints[3 * remainderIndex + 1];
	  diff[2] = targetPoint.z - globalSourcePoints[3 * remainderIndex + 2];

	  rdist = rsqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]) * M_INV_4PI;
	  if ((diff[0] == M_ZERO) && (diff[1] == M_ZERO) && (diff[2] == M_ZERO))
	    rdist = M_ZERO;

	  localResult += rdist * coefficients[indexStart + remainderIndex];
	}
      
      result[NPOINTS * gid + targetIndex] = localResult;
      for (int vecIndex = 0; vecIndex < VEC_LENGTH; vecIndex++)
	result[NPOINTS * gid + targetIndex] += VEC_ELEMENT(resultVec, vecIndex);
      
    }
  
}      
      


