#include "bempp_base_types.h"
#include "bempp_helpers.h"
#include "bempp_spaces.h"
#include "kernels.h"

__kernel void evaluate_scalar_potential_novec(__global REALTYPE *grid,
                                              __global uint* indices,
                                              __global int* normalSigns,
                                              __global REALTYPE *evalPoints,
                                              __global REALTYPE *coefficients,
                                              __constant REALTYPE* quadPoints,
                                              __constant REALTYPE *quadWeights,
                                              __global REALTYPE *globalResult) {
  size_t gid[2];

  gid[0] = get_global_id(0);
  gid[1] = get_global_id(1);

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

  size_t lid = get_local_id(1);
  size_t groupId = get_group_id(1);
  size_t numGroups = get_num_groups(1);

  REALTYPE3 evalGlobalPoint;
  REALTYPE16 surfaceGlobalPoint[3];

  REALTYPE16 corners[3][3];
  REALTYPE16 jacobian[2][3];
  REALTYPE16 normal[3];
  REALTYPE3 dummy;

  REALTYPE2 point;

  REALTYPE16 intElement;
  REALTYPE value[NUMBER_OF_SHAPE_FUNCTIONS];

  size_t quadIndex;
  size_t index;

  evalGlobalPoint =
      (REALTYPE3)(evalPoints[3 * gid[0] + 0], evalPoints[3 * gid[0] + 1],
                  evalPoints[3 * gid[0] + 2]);

  __local REALTYPE16 localResult[WORKGROUP_SIZE][2];
  REALTYPE16 myResult[2] = {M_ZERO, M_ZERO};

  REALTYPE16 kernelValue[2];
  REALTYPE16 tempResult[2];
  REALTYPE16 myCoefficients[NUMBER_OF_SHAPE_FUNCTIONS][2];

#ifndef COMPLEX_COEFFICIENTS
  for (int index = 0; index < NUMBER_OF_SHAPE_FUNCTIONS; ++index){
    myCoefficients[index][0] = (REALTYPE16)(
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
    myCoefficients[index][1] = M_ZERO;
  }
#else
  for (int index = 0; index < NUMBER_OF_SHAPE_FUNCTIONS; ++index) {
    myCoefficients[index][0] = (REALTYPE16)(
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[0] + index)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[1] + index)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[2] + index)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[3] + index)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[4] + index)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[5] + index)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[6] + index)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[7] + index)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[8] + index)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[9] + index)],
        coefficients[2 *
                     (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[10] + index)],
        coefficients[2 *
                     (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[11] + index)],
        coefficients[2 *
                     (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[12] + index)],
        coefficients[2 *
                     (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[13] + index)],
        coefficients[2 *
                     (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[14] + index)],
        coefficients[2 *
                     (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[15] + index)]);
    myCoefficients[index][1] = (REALTYPE16)(
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[0] + index) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[1] + index) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[2] + index) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[3] + index) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[4] + index) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[5] + index) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[6] + index) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[7] + index) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[8] + index) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[9] + index) +
                     1],
        coefficients
            [2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[10] + index) + 1],
        coefficients
            [2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[11] + index) + 1],
        coefficients
            [2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[12] + index) + 1],
        coefficients
            [2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[13] + index) + 1],
        coefficients
            [2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[14] + index) + 1],
        coefficients
            [2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[15] + index) + 1]);
  }
#endif

  getCornersVec16(grid, elementIndex, corners);
  getJacobianVec16(corners, jacobian);
  getNormalAndIntegrationElementVec16(jacobian, normal, &intElement);

  updateNormalsVec16(elementIndex, normalSigns, normal);

  for (quadIndex = 0; quadIndex < NUMBER_OF_QUAD_POINTS; ++quadIndex) {
    point = (REALTYPE2)(quadPoints[2 * quadIndex], quadPoints[2 * quadIndex + 1]);
    BASIS(SHAPESET, evaluate)(&point, &value[0]);
    getGlobalPointVec16(corners, &point, surfaceGlobalPoint);
    KERNEL(vec16)
    (evalGlobalPoint, surfaceGlobalPoint, dummy, normal, kernelValue);

    tempResult[0] = M_ZERO;
    tempResult[1] = M_ZERO;
    for (index = 0; index < NUMBER_OF_SHAPE_FUNCTIONS; ++index) {
      tempResult[0] += myCoefficients[index][0] * value[index];
      tempResult[1] += myCoefficients[index][1] * value[index];
    }
    tempResult[0] *= quadWeights[quadIndex];
    tempResult[1] *= quadWeights[quadIndex];

    myResult[0] +=
        tempResult[0] * kernelValue[0] - tempResult[1] * kernelValue[1];
    myResult[1] +=
        tempResult[0] * kernelValue[1] + tempResult[1] * kernelValue[0];
  }


  localResult[lid][0] = myResult[0] * intElement;
  localResult[lid][1] = myResult[1] * intElement;
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    for (index = 1; index < WORKGROUP_SIZE; ++index) {
      localResult[0][0] += localResult[index][0];
      localResult[0][1] += localResult[index][1];
    }
    for (index = 0; index < 16; ++index) {
      globalResult[2 * (gid[0] * numGroups + groupId)] +=
          ((__local REALTYPE *)(&localResult[0][0]))[index];
      globalResult[2 * (gid[0] * numGroups + groupId) + 1] +=
          ((__local REALTYPE *)(&localResult[0][1]))[index];
    }
  }
}
