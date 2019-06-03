#include "bempp_base_types.h"
#include "bempp_helpers.h"
#include "bempp_spaces.h"
#include "kernels.h"

__kernel void evaluate_electric_field_potential_vec8(
    __global REALTYPE *grid, 
    __global uint* indices,
    __global int* normalSigns,
    __global REALTYPE *evalPoints,
    __global REALTYPE *coefficients, __constant REALTYPE *quadPoints,
    __constant REALTYPE *quadWeights, __global REALTYPE *globalResult) {
  size_t gid[2];

  gid[0] = get_global_id(0);
  gid[1] = get_global_id(1);

  size_t lid = get_local_id(1);
  size_t groupId = get_group_id(1);
  size_t numGroups = get_num_groups(1);

  size_t elementIndex[8] = {
      indices[8 * gid[1] + 0], 
      indices[8 * gid[1] + 1], 
      indices[8 * gid[1] + 2], 
      indices[8 * gid[1] + 3], 
      indices[8 * gid[1] + 4], 
      indices[8 * gid[1] + 5], 
      indices[8 * gid[1] + 6], 
      indices[8 * gid[1] + 7]}; 
      
      
  REALTYPE8 surfaceGlobalPoint[3];

  REALTYPE basisValue[3][2];
  REALTYPE8 elementValue[3][3];

  REALTYPE8 corners[3][3];
  REALTYPE8 jacobian[2][3];
  REALTYPE8 normal[3];
  REALTYPE8 diff[3];

  REALTYPE8 dist;

  REALTYPE2 point;

  REALTYPE8 intElem;
  REALTYPE8 twiceInvIntElem;

  size_t quadIndex;
  size_t i, j;

  REALTYPE8 shapeIntegral[3][3][2];
  REALTYPE shiftedWavenumber[2] = {M_ZERO, M_ZERO};
  REALTYPE inverseShiftedWavenumber[2] = {M_ZERO, M_ZERO};

  __local REALTYPE8 localResult[WORKGROUP_SIZE][3][2];
  REALTYPE8 kernelValue[2];
  REALTYPE8 gradKernelValue[3][2];

  REALTYPE8 tempResult[3][3][2];
  REALTYPE8 myCoefficients[NUMBER_OF_SHAPE_FUNCTIONS][2];

  REALTYPE8 product[2];
  REALTYPE8 factor1[2];
  REALTYPE8 factor2[2];

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

// Computation of 1i * wavenumber and 1 / (1i * wavenumber)
#ifdef WAVENUMBER_COMPLEX
  shiftedWavenumber[0] = -WAVENUMBER_COMPLEX;
#endif
  shiftedWavenumber[1] = WAVENUMBER_REAL;

  inverseShiftedWavenumber[0] = M_ONE /
                                (shiftedWavenumber[0] * shiftedWavenumber[0] +
                                 shiftedWavenumber[1] * shiftedWavenumber[1]) *
                                shiftedWavenumber[0];
  inverseShiftedWavenumber[1] = -M_ONE /
                                (shiftedWavenumber[0] * shiftedWavenumber[0] +
                                 shiftedWavenumber[1] * shiftedWavenumber[1]) *
                                shiftedWavenumber[1];

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

  twiceInvIntElem = M_TWO / intElem;

  for (quadIndex = 0; quadIndex < NUMBER_OF_QUAD_POINTS; ++quadIndex) {
    point =
        (REALTYPE2)(quadPoints[2 * quadIndex], quadPoints[2 * quadIndex + 1]);
    getGlobalPointVec8(corners, &point, surfaceGlobalPoint);
    BASIS(SHAPESET, evaluate)(&point, &basisValue[0][0]);
    getPiolaTransformVec8(intElem, jacobian, basisValue, elementValue);

    diff_vec8(evalGlobalPoint, surfaceGlobalPoint, diff);
    dist = sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);

    kernelValue[0] = M_INV_4PI * cos(WAVENUMBER_REAL * dist) / dist;
    kernelValue[1] = M_INV_4PI * sin(WAVENUMBER_REAL * dist) / dist;

#ifdef WAVENUMBER_COMPLEX
    kernelValue[0] *= exp(-WAVENUMBER_COMPLEX * dist);
    kernelValue[1] *= exp(-WAVENUMBER_COMPLEX * dist);
#endif

    factor1[0] = kernelValue[0] / (dist * dist);
    factor1[1] = kernelValue[1] / (dist * dist);

    factor2[0] = -M_ONE;
    factor2[1] = WAVENUMBER_REAL * dist;

#ifdef WAVENUMBER_COMPLEX
    factor2[0] += -WAVENUMBER_COMPLEX * dist;
#endif

    product[0] = (factor1[0] * factor2[0] - factor1[1] * factor2[1]);
    product[1] = (factor1[0] * factor2[1] + factor1[1] * factor2[0]);

    gradKernelValue[0][0] = product[0] * diff[0];
    gradKernelValue[0][1] = product[1] * diff[0];
    gradKernelValue[1][0] = product[0] * diff[1];
    gradKernelValue[1][1] = product[1] * diff[1];
    gradKernelValue[2][0] = product[0] * diff[2];
    gradKernelValue[2][1] = product[1] * diff[2];

    factor1[0] = CMP_MULT_REAL(shiftedWavenumber, kernelValue);
    factor1[1] = CMP_MULT_IMAG(shiftedWavenumber, kernelValue);

    for (i = 0; i < 3; ++i) {
      tempResult[i][0][0] = factor1[0] * elementValue[i][0];
      tempResult[i][0][1] = factor1[1] * elementValue[i][0];
      tempResult[i][1][0] = factor1[0] * elementValue[i][1];
      tempResult[i][1][1] = factor1[1] * elementValue[i][1];
      tempResult[i][2][0] = factor1[0] * elementValue[i][2];
      tempResult[i][2][1] = factor1[1] * elementValue[i][2];
    }

    for (i = 0; i < 3; ++i)
      for (j = 0; j < 3; ++j) {
        tempResult[i][j][0] -=
            CMP_MULT_REAL(inverseShiftedWavenumber, gradKernelValue[j]) *
            twiceInvIntElem;
        tempResult[i][j][1] -=
            CMP_MULT_IMAG(inverseShiftedWavenumber, gradKernelValue[j]) *
            twiceInvIntElem;
      }

    for (i = 0; i < 3; ++i)
      for (j = 0; j < 3; ++j) {
        shapeIntegral[i][j][0] += tempResult[i][j][0] * quadWeights[quadIndex];
        shapeIntegral[i][j][1] += tempResult[i][j][1] * quadWeights[quadIndex];
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
