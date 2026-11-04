#include "bempp_base_types.h"
#include "bempp_helpers.h"
#include "bempp_spaces.h"
#include "kernels.h"

__kernel __attribute__((vec_type_hint(REALTYPEVEC))) void
kernel_function(__global REALTYPE *grid,
                                __global uint* indices,
                                __global int* normalSigns,
                                __global REALTYPE *evalPoints,
                                __global REALTYPE *coefficients,
                                __constant REALTYPE* quadPoints,
                                __constant REALTYPE *quadWeights,
                                __global REALTYPE *globalResult,
				__global REALTYPE* kernel_parameters) {
  size_t gid[2];

  gid[0] = get_global_id(0);
  gid[1] = get_global_id(1);

  size_t lid = get_local_id(1);
  size_t groupId = get_group_id(1);
  size_t numGroups = get_num_groups(1);

#if VEC_LENGTH == 4
  size_t elementIndex[4] = {
      indices[4 * gid[1] + 0],
      indices[4 * gid[1] + 1],
      indices[4 * gid[1] + 2],
      indices[4 * gid[1] + 3]};
#elif VEC_LENGTH == 8
  size_t elementIndex[8] = {
      indices[8 * gid[1] + 0], 
      indices[8 * gid[1] + 1], 
      indices[8 * gid[1] + 2], 
      indices[8 * gid[1] + 3], 
      indices[8 * gid[1] + 4], 
      indices[8 * gid[1] + 5], 
      indices[8 * gid[1] + 6], 
      indices[8 * gid[1] + 7]};
#elif VEC_LENGTH == 16
  size_t elementIndex[16] = {
      indices[16 * gid[1] + 0],
      indices[16 * gid[1] + 1],
      indices[16 * gid[1] + 2],
      indices[16 * gid[1] + 3],
      indices[16 * gid[1] + 4],
      indices[16 * gid[1] + 5],
      indices[16 * gid[1] + 6],
      indices[16 * gid[1] + 7],
      indices[16 * gid[1] + 8],
      indices[16 * gid[1] + 9],
      indices[16 * gid[1] + 10],
      indices[16 * gid[1] + 11],
      indices[16 * gid[1] + 12],
      indices[16 * gid[1] + 13],
      indices[16 * gid[1] + 14],
      indices[16 * gid[1] + 15]};
#endif
          

  REALTYPE3 evalGlobalPoint;
//  REALTYPE3 surfaceGlobalPoint;
  REALTYPEVEC surfaceGlobalPoint[3];

  REALTYPEVEC corners[3][3];
  REALTYPEVEC jacobian[2][3];
  REALTYPEVEC normal[3];
  REALTYPE3 dummy;

  REALTYPE2 point;
  REALTYPEVEC diff[3];

  REALTYPEVEC dist;
  REALTYPEVEC invdist3;
  REALTYPEVEC invdist5;

  REALTYPEVEC intElement;
  REALTYPE value[NUMBER_OF_SHAPE_FUNCTIONS];

  size_t quadIndex;
  size_t index;
  size_t j;

  evalGlobalPoint =
      (REALTYPE3)(evalPoints[3 * gid[0] + 0], evalPoints[3 * gid[0] + 1],
                  evalPoints[3 * gid[0] + 2]);


  __local REALTYPEVEC localResult[WORKGROUP_SIZE][3];
  REALTYPEVEC myResult[3] = {M_ZERO, M_ZERO, M_ZERO};

  REALTYPEVEC kernelMatrix[6];
  REALTYPEVEC kernelValue[3];

  REALTYPEVEC tempResult;
  REALTYPEVEC myCoefficients[NUMBER_OF_SHAPE_FUNCTIONS];
  for (int index = 0; index < NUMBER_OF_SHAPE_FUNCTIONS; ++index)
    myCoefficients[index] = (REALTYPEVEC)(
#if VEC_LENGTH == 4
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[0] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[1] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[2] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[3] + index]);
#elif VEC_LENGTH == 8
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[0] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[1] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[2] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[3] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[4] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[5] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[6] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[7] + index]);
#elif VEC_LENGTH == 16
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[0] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[1] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[2] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[3] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[4] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[5] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[6] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[7] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[8] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[9] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[10] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[11] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[12] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[13] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[14] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[15] + index]);
#endif

  getCornersVec(grid, elementIndex, corners);
  getJacobianVec(corners, jacobian);
  getNormalAndIntegrationElementVec(jacobian, normal, &intElement);
  updateNormalsVec(elementIndex, normalSigns, normal);

  for (quadIndex = 0; quadIndex < NUMBER_OF_QUAD_POINTS; ++quadIndex) {
    point = (REALTYPE2)(quadPoints[2 * quadIndex], quadPoints[2 * quadIndex + 1]);
    BASIS(SHAPESET, evaluate)(&point, &value[0]);
    getGlobalPointVec(corners, &point, surfaceGlobalPoint);
//    surfaceGlobalPoint = getGlobalPoint(corners, &point);
    
    diff_vec(evalGlobalPoint, surfaceGlobalPoint, diff);
    dist = sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);
    
    invdist3 = 1 / (dist * dist * dist);
    invdist5 = 1 / (dist * dist * dist * dist * dist);

 //   KERNEL(VEC_STRING)
 //   (evalGlobalPoint, surfaceGlobalPoint, dummy, normal, kernel_parameters, &kernelValue);
    
// xx, xy, xz, yy, yz, zz    
    
    kernelMatrix[0] = -M_INV_4PI * (invdist3 - 3 * invdist5 * diff[0] * diff[0]);
    kernelMatrix[1] = M_INV_4PI * 3 * invdist5 * diff[1] * diff[0];
    kernelMatrix[2] = M_INV_4PI * 3 * invdist5 * diff[2] * diff[0];
    kernelMatrix[3] = -M_INV_4PI * (invdist3 - 3 * invdist5 * diff[1] * diff[1]);
    kernelMatrix[4] = M_INV_4PI * 3 * invdist5 * diff[2] * diff[1];
    kernelMatrix[5] = -M_INV_4PI * (invdist3 - 3 * invdist5 * diff[2] * diff[2]);
    
    kernelValue[0] = kernelMatrix[0] * normal[0] + kernelMatrix[1] * normal[1] + kernelMatrix[2] * normal[2];
    kernelValue[1] = kernelMatrix[1] * normal[0] + kernelMatrix[3] * normal[1] + kernelMatrix[4] * normal[2];
    kernelValue[2] = kernelMatrix[2] * normal[0] + kernelMatrix[4] * normal[1] + kernelMatrix[5] * normal[2];

    tempResult = M_ZERO;
    for (index = 0; index < NUMBER_OF_SHAPE_FUNCTIONS; ++index)
      tempResult += myCoefficients[index] * value[index];
    tempResult *= quadWeights[quadIndex];
    for (j = 0; j < 3; ++j)
      myResult[j] += tempResult * kernelValue[j];
  }

  for (j = 0; j < 3; ++j)
    localResult[lid][j] = myResult[j] * intElement;
  
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    for (index = 1; index < WORKGROUP_SIZE; ++index){
      for (j = 0; j < 3; ++j)
        localResult[0][j] += localResult[index][j];
    }
    for (index = 0; index < VEC_LENGTH; ++index)
      for (j = 0; j < 3; ++j)
//        globalResult[gid[0] * numGroups + groupId][j] +=
//          ((__local REALTYPE *)(&localResult[0][j]))[index];
        globalResult[(3*gid[0]+j) * numGroups + groupId] +=
          ((__local REALTYPE *)(&localResult[0][j]))[index];
          
  }
}
