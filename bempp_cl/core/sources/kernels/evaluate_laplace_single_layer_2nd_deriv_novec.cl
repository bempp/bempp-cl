#include "bempp_base_types.h"
#include "bempp_helpers.h"
#include "bempp_spaces.h"
#include "kernels.h"

__kernel void kernel_function(__global REALTYPE *grid,
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

  size_t elementIndex = indices[gid[1]];

  size_t lid = get_local_id(1);
  size_t groupId = get_group_id(1);
  size_t numGroups = get_num_groups(1);

  REALTYPE3 evalGlobalPoint;
  REALTYPE3 surfaceGlobalPoint;

  REALTYPE3 corners[3];
  REALTYPE3 jacobian[2];
  REALTYPE3 normal;
  REALTYPE3 dummy;
  REALTYPE3 diff;
  REALTYPE dist;
  REALTYPE invdist3;
  REALTYPE invdist5;

  REALTYPE2 point;

  REALTYPE intElement;
  REALTYPE value[NUMBER_OF_SHAPE_FUNCTIONS];

  size_t quadIndex;
  size_t index;
  size_t j;

  evalGlobalPoint =
      (REALTYPE3)(evalPoints[3 * gid[0] + 0], evalPoints[3 * gid[0] + 1],
                  evalPoints[3 * gid[0] + 2]);

  __local REALTYPE localResult[WORKGROUP_SIZE][3];
  REALTYPE myResult[6] = {M_ZERO, M_ZERO, M_ZERO, M_ZERO, M_ZERO, M_ZERO};


  REALTYPE kernelValue[6];

  REALTYPE tempResult;
  REALTYPE myCoefficients[NUMBER_OF_SHAPE_FUNCTIONS];
  for (int index = 0; index < NUMBER_OF_SHAPE_FUNCTIONS; ++index)
    myCoefficients[index] =
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex + index];


  getCorners(grid, elementIndex, corners);
  getJacobian(corners, jacobian);
  getNormalAndIntegrationElement(jacobian, &normal, &intElement);

  updateNormals(elementIndex, normalSigns, &normal);

  for (quadIndex = 0; quadIndex < NUMBER_OF_QUAD_POINTS; ++quadIndex) {
    point = (REALTYPE2)(quadPoints[2 * quadIndex], quadPoints[2 * quadIndex + 1]);
    BASIS(SHAPESET, evaluate)(&point, &value[0]);
    surfaceGlobalPoint = getGlobalPoint(corners, &point);
    
    dist = distance(evalGlobalPoint, surfaceGlobalPoint);
    diff = evalGlobalPoint - surfaceGlobalPoint;

 //   KERNEL(novec)
 //   (evalGlobalPoint, surfaceGlobalPoint, dummy, normal, kernel_parameters, &kernelValue);
    
    invdist3 = 1 / (dist * dist * dist);
    invdist5 = 1 / (dist * dist * dist * dist * dist);
    
 // xx, xy, xz, yy, yz, zz   
    
    kernelValue[0] = M_INV_4PI * (invdist3 - 3 * invdist5 * diff.x * diff.x);
    kernelValue[1] = -M_INV_4PI *  3 * invdist5 * diff.y * diff.x;
    kernelValue[2] = -M_INV_4PI * 3 * invdist5 * diff.z * diff.x;
    kernelValue[3] = M_INV_4PI * (invdist3 - 3 * invdist5 * diff.y * diff.y);
    kernelValue[4] = -M_INV_4PI * 3 * invdist5 * diff.z * diff.y;
    kernelValue[5] = M_INV_4PI * (invdist3 - 3 * invdist5 * diff.z * diff.z);    
    
    tempResult = M_ZERO;
    for (index = 0; index < NUMBER_OF_SHAPE_FUNCTIONS; ++index)
      tempResult += myCoefficients[index] * value[index];
    tempResult *= quadWeights[quadIndex];
    for (j = 0; j < 6; ++j)
      myResult[j] += tempResult * kernelValue[j];      
  }
  for (j = 0; j < 6; ++j)
    localResult[lid][j] = myResult[j] * intElement;

  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    for (index = 1; index < WORKGROUP_SIZE; ++index){
      for (j = 0; j < 6; ++j)
        localResult[0][j] += localResult[index][j];
    }
    for (j = 0; j < 6; ++j)
      globalResult[(6 * gid[0] + j) * numGroups + groupId] += localResult[0][j];
  }

}
