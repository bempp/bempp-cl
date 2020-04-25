#include "bempp_base_types.h"
#include "bempp_helpers.h"
#include "bempp_spaces.h"
#include "kernels.h"

__kernel void evaluate_magnetic_field_potential_novec(
    __global REALTYPE *grid, 
    __global uint* indices,
    __global int* normalSigns,
    __global REALTYPE *evalPoints,
    __global REALTYPE *coefficients, __constant REALTYPE *quadPoints,
    __constant REALTYPE *quadWeights, __global REALTYPE *globalResult) {
  size_t gid[2];

  gid[0] = get_global_id(0);
  gid[1] = get_global_id(1);

  size_t elementIndex[8] = {
      indices[8 * gid[1] + 0], 
      indices[8 * gid[1] + 1], 
      indices[8 * gid[1] + 2], 
      indices[8 * gid[1] + 3], 
      indices[8 * gid[1] + 4], 
      indices[8 * gid[1] + 5], 
      indices[8 * gid[1] + 6], 
      indices[8 * gid[1] + 7]}; 

  size_t lid = get_local_id(1);
  size_t groupId = get_group_id(1);
  size_t numGroups = get_num_groups(1);

  size_t vecIndex;

  REALTYPE8 surfaceGlobalPoint[3];

  REALTYPE basisValue[3][2];
  REALTYPE8 elementValue[3][3];

  REALTYPE8 corners[3][3];
  REALTYPE8 jacobian[2][3];
  REALTYPE8 normal[3];

  REALTYPE8 factor1[2];
  REALTYPE3 testNormal; // Dummy variable. Only needed for kernel call.

  REALTYPE2 point;

  REALTYPE8 intElem;

  size_t quadIndex;
  size_t i, j, k;

  REALTYPE8 shapeIntegral[3][3][2];

  __local REALTYPE8 localResult[WORKGROUP_SIZE][3][2];
  REALTYPE8 gradKernelValue[3][2];

  REALTYPE8 myCoefficients[NUMBER_OF_SHAPE_FUNCTIONS][2];

  REALTYPE8 edgeLengths[3];

  REALTYPE3 evalGlobalPoint =
      (REALTYPE3)(evalPoints[3 * gid[0] + 0], evalPoints[3 * gid[0] + 1],
                  evalPoints[3 * gid[0] + 2]);

#ifndef COMPLEX_COEFFICIENTS
  for (i = 0; i < 3; ++i) {
    myCoefficients[i][0] = (REALTYPE8)(
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[0] + i],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[1] + i],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[2] + i],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[3] + i],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[4] + i],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[5] + i],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[6] + i],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[7] + i]);
    myCoefficients[i][1] = M_ZERO;
  }
#else
  for (i = 0; i < 3; ++i) {
    myCoefficients[i][0] = (REALTYPE8)(
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[0] + i)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[1] + i)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[2] + i)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[3] + i)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[4] + i)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[5] + i)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[6] + i)],
        coefficients[2 *
                     (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[7] + i)]);
    myCoefficients[i][1] = (REALTYPE8)(
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[0] + i) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[1] + i) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[2] + i) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[3] + i) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[4] + i) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[5] + i) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[6] + i) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[7] + i) +
                     1]);
  }
#endif

  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j) {
      shapeIntegral[i][j][0] = M_ZERO;
      shapeIntegral[i][j][1] = M_ZERO;
    }

  getCornersVec8(grid, elementIndex, corners);
  getJacobianVec8(corners, jacobian);
  getNormalAndIntegrationElementVec8(jacobian, normal, &intElem);

  updateNormalsVec8(elementIndex, normalSigns, normal);
  computeEdgeLengthVec8(corners, edgeLengths);

  for (quadIndex = 0; quadIndex < NUMBER_OF_QUAD_POINTS; ++quadIndex) {
    point =
        (REALTYPE2)(quadPoints[2 * quadIndex], quadPoints[2 * quadIndex + 1]);
    getGlobalPointVec8(corners, &point, surfaceGlobalPoint);
    BASIS(SHAPESET, evaluate)(&point, &basisValue[0][0]);
    getPiolaTransformVec8(intElem, jacobian, basisValue, elementValue);

    KERNEL(vec8)(evalGlobalPoint, surfaceGlobalPoint, testNormal, normal, gradKernelValue); 


    for (i = 0; i < 3; ++i)
        for (k = 0; k < 2; ++k){
        shapeIntegral[i][0][k] += (gradKernelValue[1][k] * elementValue[i][2] - gradKernelValue[2][k] * elementValue[i][1])  * quadWeights[quadIndex];
        shapeIntegral[i][1][k] += (gradKernelValue[2][k] * elementValue[i][0] - gradKernelValue[0][k] * elementValue[i][2])  * quadWeights[quadIndex];
        shapeIntegral[i][2][k] += (gradKernelValue[0][k] * elementValue[i][1] - gradKernelValue[1][k] * elementValue[i][0])  * quadWeights[quadIndex];
        }
  }

  for (j = 0; j < 3; ++j) {
    factor1[0] = M_ZERO;
    factor1[1] = M_ZERO;
    for (i = 0; i < 3; ++i) {
      factor1[0] += CMP_MULT_REAL(shapeIntegral[i][j], myCoefficients[i]) *
                    edgeLengths[i];
      factor1[1] += CMP_MULT_IMAG(shapeIntegral[i][j], myCoefficients[i]) *
                    edgeLengths[i];
    }
    localResult[lid][j][0] = factor1[0] * intElem;
    localResult[lid][j][1] = factor1[1] * intElem;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    for (i = 1; i < WORKGROUP_SIZE; ++i)
      for (j = 0; j < 3; ++j) {
        localResult[0][j][0] += localResult[i][j][0];
        localResult[0][j][1] += localResult[i][j][1];
      }
    for (int vecIndex = 0; vecIndex < 8; ++vecIndex)
      for (j = 0; j < 3; ++j) {
        globalResult[2 * ((KERNEL_DIMENSION * gid[0] + j) * numGroups +
                          groupId)] += ((__local REALTYPE*)(&localResult[0][j][0]))[vecIndex];
        globalResult[2 * ((KERNEL_DIMENSION * gid[0] + j) * numGroups +
                          groupId) +
                     1] += ((__local REALTYPE*)(&localResult[0][j][1]))[vecIndex];
      }
  }
}
