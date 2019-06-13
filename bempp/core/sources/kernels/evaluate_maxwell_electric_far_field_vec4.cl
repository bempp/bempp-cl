#include "bempp_base_types.h"
#include "bempp_helpers.h"
#include "bempp_spaces.h"
#include "kernels.h"

__kernel void evaluate_electric_far_field_novec(
    __global REALTYPE *grid, 
    __global uint* indices,
    __global int* normalSigns,
    __global REALTYPE *evalPoints,
    __global REALTYPE *coefficients, __constant REALTYPE *quadPoints,
    __constant REALTYPE *quadWeights, __global REALTYPE *globalResult) {
  size_t gid[2];

  gid[0] = get_global_id(0);
  gid[1] = get_global_id(1);

  size_t elementIndex[4] = {
      indices[4 * gid[1] + 0],
      indices[4 * gid[1] + 1],
      indices[4 * gid[1] + 2],
      indices[4 * gid[1] + 3]};

  size_t lid = get_local_id(1);
  size_t groupId = get_group_id(1);
  size_t numGroups = get_num_groups(1);

  REALTYPE4 surfaceGlobalPoint[3];

  REALTYPE basisValue[3][2];
  REALTYPE4 elementValue[3][3];

  REALTYPE4 corners[3][3];
  REALTYPE4 jacobian[2][3];
  REALTYPE4 normal[3];
  REALTYPE4 inner;

  REALTYPE2 point;

  REALTYPE4 intElem;
  REALTYPE4 twiceInvIntElem;

  size_t quadIndex;
  size_t i, j;

  REALTYPE4 shapeIntegral[3][3][2];
  REALTYPE4 kernelValue[2];

  __local REALTYPE4 localResult[WORKGROUP_SIZE][3][2];
  REALTYPE4 myCoefficients[NUMBER_OF_SHAPE_FUNCTIONS][2];
  REALTYPE4 factor1[2];
  REALTYPE4 edgeLengths[3];

  REALTYPE3 evalGlobalPoint =
      (REALTYPE3)(evalPoints[3 * gid[0] + 0], evalPoints[3 * gid[0] + 1],
                  evalPoints[3 * gid[0] + 2]);

#ifndef COMPLEX_COEFFICIENTS
  for (i = 0; i < 3; ++i) {
    myCoefficients[i][0] = (REALTYPE4)(
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[0] + i],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[1] + i],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[2] + i],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[3] + i]);
    myCoefficients[i][1] = M_ZERO;
  }
#else
  for (i = 0; i < 3; ++i) {
    myCoefficients[i][0] =
        (REALTYPE4)(coefficients[2 * (3 * elementIndex[0] + i)],
                    coefficients[2 * (3 * elementIndex[1] + i)],
                    coefficients[2 * (3 * elementIndex[2] + i)],
                    coefficients[2 * (3 * elementIndex[3] + i)]);
    myCoefficients[i][1] =
        (REALTYPE4)(coefficients[2 * (3 * elementIndex[0] + i) + 1],
                    coefficients[2 * (3 * elementIndex[1] + i) + 1],
                    coefficients[2 * (3 * elementIndex[2] + i) + 1],
                    coefficients[2 * (3 * elementIndex[3] + i) + 1]);
  }
#endif

  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j) {
      shapeIntegral[i][j][0] = M_ZERO;
      shapeIntegral[i][j][1] = M_ZERO;
    }

  getCornersVec4(grid, elementIndex, corners);
  getJacobianVec4(corners, jacobian);
  getNormalAndIntegrationElementVec4(jacobian, normal, &intElem);

  updateNormalsVec4(elementIndex, normalSigns, normal);

  computeEdgeLengthVec4(corners, edgeLengths);

  twiceInvIntElem = M_TWO / intElem;

  for (quadIndex = 0; quadIndex < NUMBER_OF_QUAD_POINTS; ++quadIndex) {
    point =
        (REALTYPE2)(quadPoints[2 * quadIndex], quadPoints[2 * quadIndex + 1]);
    getGlobalPointVec4(corners, &point, surfaceGlobalPoint);
    BASIS(SHAPESET, evaluate)(&point, &basisValue[0][0]);
    getPiolaTransformVec4(intElem, jacobian, basisValue, elementValue);

    inner = evalGlobalPoint.x * surfaceGlobalPoint[0] + evalGlobalPoint.y * surfaceGlobalPoint[1] + 
        evalGlobalPoint.z * surfaceGlobalPoint[2];

    kernelValue[0] = M_INV_4PI * cos(-WAVENUMBER_REAL * inner); 
    kernelValue[1] = M_INV_4PI * sin(-WAVENUMBER_REAL * inner);

    for (i = 0; i < 3; ++i)
        for (j = 0; j < 3; ++j)
      {
        shapeIntegral[i][j][0] += (-kernelValue[1] * WAVENUMBER_REAL * elementValue[i][j] -  
                kernelValue[0] * VEC_ELEMENT(evalGlobalPoint, j) * twiceInvIntElem) * quadWeights[quadIndex];
        shapeIntegral[i][j][1] += (kernelValue[0] * WAVENUMBER_REAL * elementValue[i][j] - 
                kernelValue[1] * VEC_ELEMENT(evalGlobalPoint, j) * twiceInvIntElem) * quadWeights[quadIndex];
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
    for (int vecIndex = 0; vecIndex < 4; ++vecIndex)
      for (j = 0; j < 3; ++j) {
        globalResult[2 * ((KERNEL_DIMENSION * gid[0] + j) * numGroups +
                          groupId)] += ((__local REALTYPE*)(&localResult[0][j][0]))[vecIndex];
        globalResult[2 * ((KERNEL_DIMENSION * gid[0] + j) * numGroups +
                          groupId) +
                     1] += ((__local REALTYPE*)(&localResult[0][j][1]))[vecIndex];
      }
  }
}
