#include "bempp_base_types.h"
#include "bempp_helpers.h"
#include "bempp_spaces.h"
#include "kernels.h"

__kernel void evaluate_magnetic_far_field_novec(
    __global REALTYPE *grid, 
    __global uint* indices,
    __global int* normalSigns,
    __global REALTYPE *evalPoints,
    __global REALTYPE *coefficients, __constant REALTYPE *quadPoints,
    __constant REALTYPE *quadWeights, __global REALTYPE *globalResult) {
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

  REALTYPE16 surfaceGlobalPoint[3];

  REALTYPE basisValue[3][2];
  REALTYPE16 elementValue[3][3];

  REALTYPE16 corners[3][3];
  REALTYPE16 jacobian[2][3];
  REALTYPE16 crossProd[3][3];
  REALTYPE16 normal[3];
  REALTYPE16 inner;

  REALTYPE2 point;

  REALTYPE16 intElem;

  size_t quadIndex;
  size_t i, j;

  REALTYPE16 shapeIntegral[3][3][2];
  REALTYPE16 kernelValue[2];

  __local REALTYPE16 localResult[WORKGROUP_SIZE][3][2];
  REALTYPE16 myCoefficients[NUMBER_OF_SHAPE_FUNCTIONS][2];
  REALTYPE16 factor1[2];
  REALTYPE16 edgeLengths[3];

  REALTYPE3 evalGlobalPoint =
      (REALTYPE3)(evalPoints[3 * gid[0] + 0], evalPoints[3 * gid[0] + 1],
                  evalPoints[3 * gid[0] + 2]);

#ifndef COMPLEX_COEFFICIENTS
  for (i = 0; i < 3; ++i) {
    myCoefficients[i][0] = (REALTYPE16)(
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[0] + i],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[1] + i],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[2] + i],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[3] + i],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[4] + i],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[5] + i],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[6] + i],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[7] + i],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[8] + i],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[9] + i],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[10] + i],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[11] + i],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[12] + i],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[13] + i],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[14] + i],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[15] + i]);
    myCoefficients[i][1] = M_ZERO;
  }
#else
  for (i = 0; i < 3; ++i) {
    myCoefficients[i][0] = (REALTYPE16)(
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[0] + i)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[1] + i)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[2] + i)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[3] + i)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[4] + i)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[5] + i)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[6] + i)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[7] + i)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[8] + i)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[9] + i)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[10] + i)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[11] + i)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[12] + i)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[13] + i)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[14] + i)],
        coefficients[2 *
                     (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[15] + i)]);
    myCoefficients[i][1] = (REALTYPE16)(
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
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[8] + i) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[9] + i) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[10] + i) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[11] + i) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[12] + i) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[13] + i) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[14] + i) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[15] + i) +
                     1]);
  }
#endif

  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j) {
      shapeIntegral[i][j][0] = M_ZERO;
      shapeIntegral[i][j][1] = M_ZERO;
    }

  getCornersVec16(grid, elementIndex, corners);
  getJacobianVec16(corners, jacobian);
  getNormalAndIntegrationElementVec16(jacobian, normal, &intElem);

  updateNormalsVec16(elementIndex, normalSigns, normal);

  computeEdgeLengthVec16(corners, edgeLengths);

  for (quadIndex = 0; quadIndex < NUMBER_OF_QUAD_POINTS; ++quadIndex) {
    point =
        (REALTYPE2)(quadPoints[2 * quadIndex], quadPoints[2 * quadIndex + 1]);
    getGlobalPointVec16(corners, &point, surfaceGlobalPoint);
    BASIS(SHAPESET, evaluate)(&point, &basisValue[0][0]);
    getPiolaTransformVec16(intElem, jacobian, basisValue, elementValue);

    inner = evalGlobalPoint.x * surfaceGlobalPoint[0] + evalGlobalPoint.y * surfaceGlobalPoint[1] + 
        evalGlobalPoint.z * surfaceGlobalPoint[2];

    kernelValue[0] = M_INV_4PI * cos(-WAVENUMBER_REAL * inner); 
    kernelValue[1] = M_INV_4PI * sin(-WAVENUMBER_REAL * inner);

    for (i = 0; i < 3; ++i){
        crossProd[i][0] = evalGlobalPoint.y * elementValue[i][2] - evalGlobalPoint.z * elementValue[i][1];
        crossProd[i][1] = evalGlobalPoint.z * elementValue[i][0] - evalGlobalPoint.x * elementValue[i][2];
        crossProd[i][2] = evalGlobalPoint.x * elementValue[i][1] - evalGlobalPoint.y * elementValue[i][0];
    }

    for (i = 0; i < 3; ++i)
        for (j = 0; j < 3; ++j)
      {
        shapeIntegral[i][j][0] += -kernelValue[1] * WAVENUMBER_REAL * crossProd[i][j] * quadWeights[quadIndex];
        shapeIntegral[i][j][1] += kernelValue[0] * WAVENUMBER_REAL * crossProd[i][j] * quadWeights[quadIndex];
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
    for (int vecIndex = 0; vecIndex < 16; ++vecIndex)
      for (j = 0; j < 3; ++j) {
        globalResult[2 * ((KERNEL_DIMENSION * gid[0] + j) * numGroups +
                          groupId)] += ((__local REALTYPE*)(&localResult[0][j][0]))[vecIndex];
        globalResult[2 * ((KERNEL_DIMENSION * gid[0] + j) * numGroups +
                          groupId) +
                     1] += ((__local REALTYPE*)(&localResult[0][j][1]))[vecIndex];
      }
  }
}
