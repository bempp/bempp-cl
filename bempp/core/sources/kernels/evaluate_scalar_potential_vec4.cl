#include "bempp_base_types.h"
#include "bempp_helpers.h"
#include "bempp_spaces.h"
#include "kernels.h"

__kernel __attribute__((vec_type_hint(REALTYPE4))) void
evaluate_scalar_potential_novec(__global REALTYPE *grid,
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

  size_t lid = get_local_id(1);
  size_t groupId = get_group_id(1);
  size_t numGroups = get_num_groups(1);

  size_t elementIndex[4] = {
      indices[4 * gid[1] + 0],
      indices[4 * gid[1] + 1],
      indices[4 * gid[1] + 2],
      indices[4 * gid[1] + 3]};
          
          

  REALTYPE3 evalGlobalPoint;
  REALTYPE4 surfaceGlobalPoint[3];

  REALTYPE4 corners[3][3];
  REALTYPE4 jacobian[2][3];
  REALTYPE4 normal[3];
  REALTYPE3 dummy;

  REALTYPE2 point;

  REALTYPE4 intElement;
  REALTYPE value[NUMBER_OF_SHAPE_FUNCTIONS];

  size_t quadIndex;
  size_t index;

  evalGlobalPoint =
      (REALTYPE3)(evalPoints[3 * gid[0] + 0], evalPoints[3 * gid[0] + 1],
                  evalPoints[3 * gid[0] + 2]);

#ifndef COMPLEX_RESULT
  __local REALTYPE4 localResult[WORKGROUP_SIZE];
  REALTYPE4 myResult = M_ZERO;
#else
  __local REALTYPE4 localResult[WORKGROUP_SIZE][2];
  REALTYPE4 myResult[2] = {M_ZERO, M_ZERO};
#endif

#ifndef COMPLEX_KERNEL
  REALTYPE4 kernelValue;
#else
  REALTYPE4 kernelValue[2];
#endif

#ifndef COMPLEX_COEFFICIENTS
  REALTYPE4 tempResult;
  REALTYPE4 myCoefficients[NUMBER_OF_SHAPE_FUNCTIONS];
  for (int index = 0; index < NUMBER_OF_SHAPE_FUNCTIONS; ++index)
    myCoefficients[index] = (REALTYPE4)(
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[0] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[1] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[2] + index],
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[3] + index]);
#else
  REALTYPE4 tempResult[2];
  REALTYPE4 myCoefficients[NUMBER_OF_SHAPE_FUNCTIONS][2];
  for (int index = 0; index < NUMBER_OF_SHAPE_FUNCTIONS; ++index) {
    myCoefficients[index][0] = (REALTYPE4)(
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[0] + index)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[1] + index)],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[2] + index)],
        coefficients[2 *
                     (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[3] + index)]);
    myCoefficients[index][1] = (REALTYPE4)(
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[0] + index) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[1] + index) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[2] + index) +
                     1],
        coefficients[2 * (NUMBER_OF_SHAPE_FUNCTIONS * elementIndex[3] + index) +
                     1]);
  }
#endif

  getCornersVec4(grid, elementIndex, corners);
  getJacobianVec4(corners, jacobian);
  getNormalAndIntegrationElementVec4(jacobian, normal, &intElement);
  updateNormalsVec4(elementIndex, normalSigns, normal);

  for (quadIndex = 0; quadIndex < NUMBER_OF_QUAD_POINTS; ++quadIndex) {
    point = (REALTYPE2)(quadPoints[2 * quadIndex], quadPoints[2 * quadIndex + 1]);
    BASIS(SHAPESET, evaluate)(&point, &value[0]);
    getGlobalPointVec4(corners, &point, surfaceGlobalPoint);
#ifndef COMPLEX_KERNEL
    KERNEL(vec4)
    (evalGlobalPoint, surfaceGlobalPoint, dummy, normal, &kernelValue);
#else
    KERNEL(vec4)
    (evalGlobalPoint, surfaceGlobalPoint, dummy, normal, kernelValue);
#endif

#ifndef COMPLEX_COEFFICIENTS
    tempResult = M_ZERO;
    for (index = 0; index < NUMBER_OF_SHAPE_FUNCTIONS; ++index)
      tempResult += myCoefficients[index] * value[index];
    tempResult *= quadWeights[quadIndex];
#ifndef COMPLEX_KERNEL
    myResult += tempResult * kernelValue;
#else
    myResult[0] += tempResult * kernelValue[0];
    myResult[1] += tempResult * kernelValue[1];
#endif
#else
    tempResult[0] = M_ZERO;
    tempResult[1] = M_ZERO;
    for (index = 0; index < NUMBER_OF_SHAPE_FUNCTIONS; ++index) {
      tempResult[0] += myCoefficients[index][0] * value[index];
      tempResult[1] += myCoefficients[index][1] * value[index];
    }
    tempResult[0] *= quadWeights[quadIndex];
    tempResult[1] *= quadWeights[quadIndex];

#ifndef COMPLEX_KERNEL
    myResult[0] += tempResult[0] * kernelValue;
    myResult[1] += tempResult[1] * kernelValue;
#else
    myResult[0] +=
        tempResult[0] * kernelValue[0] - tempResult[1] * kernelValue[1];
    myResult[1] +=
        tempResult[0] * kernelValue[1] + tempResult[1] * kernelValue[0];
#endif
#endif
  }

#ifndef COMPLEX_RESULT
  localResult[lid] = myResult * intElement;
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    for (index = 1; index < WORKGROUP_SIZE; ++index)
      localResult[0] += localResult[index];
    for (index = 0; index < 4; ++index)
      globalResult[gid[0] * numGroups + groupId] +=
          ((__local REALTYPE *)(&localResult[0]))[index];
  }

#else
  localResult[lid][0] = myResult[0] * intElement;
  localResult[lid][1] = myResult[1] * intElement;
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    for (index = 1; index < WORKGROUP_SIZE; ++index) {
      localResult[0][0] += localResult[index][0];
      localResult[0][1] += localResult[index][1];
    }
    for (index = 0; index < 4; ++index) {
      globalResult[2 * (gid[0] * numGroups + groupId)] +=
          ((__local REALTYPE *)(&localResult[0][0]))[index];
      globalResult[2 * (gid[0] * numGroups + groupId) + 1] +=
          ((__local REALTYPE *)(&localResult[0][1]))[index];
    }
  }
#endif
}
