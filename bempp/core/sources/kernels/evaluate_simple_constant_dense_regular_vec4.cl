#include "bempp_base_types.h"
#include "bempp_helpers.h"
#include "bempp_spaces.h"
#include "kernels.h"

__kernel __attribute__((vec_type_hint(REALTYPE4))) __kernel void
evaluate_dense_regular(__global REALTYPE *testGrid,
                       __global REALTYPE *trialGrid,
                       __constant REALTYPE* quadPoints,
                       __constant REALTYPE *quadWeights,
                       __global REALTYPE *globalResult, uint number_of_cols) {
  /* Variable declarations */

  size_t gid[2] = {get_global_id(0), get_global_id(1)};
  size_t offset = get_global_offset(1);
  size_t testIndex = gid[0];
  size_t trialIndex[4] = {
      offset + 4 * (gid[1] - offset) + 0, offset + 4 * (gid[1] - offset) + 1,
      offset + 4 * (gid[1] - offset) + 2, offset + 4 * (gid[1] - offset) + 3};

  size_t testQuadIndex;
  size_t trialQuadIndex;
  size_t i;
  size_t j;

  size_t vecIndex;

  REALTYPE3 testGlobalPoint;
  REALTYPE4 trialGlobalPoint[3];

  REALTYPE3 testCorners[3];
  REALTYPE4 trialCorners[3][3];

  REALTYPE3 testJac[2];
  REALTYPE4 trialJac[2][3];

  REALTYPE3 testNormal;
  REALTYPE4 trialNormal[3];

  REALTYPE2 testPoint;
  REALTYPE2 trialPoint;

  REALTYPE testIntElem;
  REALTYPE4 trialIntElem;

#ifndef COMPLEX_KERNEL
  REALTYPE4 kernelValue;
  REALTYPE4 tempResult;
  REALTYPE4 shapeIntegral;
#else
  REALTYPE4 kernelValue[2];
  REALTYPE4 tempResult[2];
  REALTYPE4 shapeIntegral[2];
#endif

#ifndef COMPLEX_KERNEL
  shapeIntegral = M_ZERO;
#else
  shapeIntegral[0] = M_ZERO;
  shapeIntegral[1] = M_ZERO;
#endif

  getCorners(testGrid, testIndex, testCorners);
  getCornersVec4(trialGrid, trialIndex, trialCorners);

  getJacobian(testCorners, testJac);
  getJacobianVec4(trialCorners, trialJac);

  getNormalAndIntegrationElement(testJac, &testNormal, &testIntElem);
  getNormalAndIntegrationElementVec4(trialJac, trialNormal, &trialIntElem);

  for (testQuadIndex = 0; testQuadIndex < NUMBER_OF_QUAD_POINTS;
       ++testQuadIndex) {
    testPoint = (REALTYPE2)(quadPoints[2 * testQuadIndex], quadPoints[2 * testQuadIndex + 1]);
    testGlobalPoint = getGlobalPoint(testCorners, &testPoint);

#ifndef COMPLEX_KERNEL
    tempResult = M_ZERO;
#else
    tempResult[0] = M_ZERO;
    tempResult[1] = M_ZERO;
#endif

    for (trialQuadIndex = 0; trialQuadIndex < NUMBER_OF_QUAD_POINTS;
         ++trialQuadIndex) {
      trialPoint = (REALTYPE2)(quadPoints[2 * trialQuadIndex], quadPoints[2 * trialQuadIndex + 1]);
      getGlobalPointVec4(trialCorners, &trialPoint, trialGlobalPoint);
#ifndef COMPLEX_KERNEL
      KERNEL(vec4)
      (testGlobalPoint, trialGlobalPoint, testNormal, trialNormal,
       &kernelValue);
      tempResult += quadWeights[trialQuadIndex] * kernelValue;
#else
      KERNEL(vec4)
      (testGlobalPoint, trialGlobalPoint, testNormal, trialNormal, kernelValue);
      tempResult[0] += quadWeights[trialQuadIndex] * kernelValue[0];
      tempResult[1] += quadWeights[trialQuadIndex] * kernelValue[1];
#endif
    }
#ifndef COMPLEX_KERNEL
    shapeIntegral += tempResult * quadWeights[testQuadIndex];
#else
    shapeIntegral[0] += tempResult[0] * quadWeights[testQuadIndex];
    shapeIntegral[1] += tempResult[1] * quadWeights[testQuadIndex];
#endif
  }
#ifndef COMPLEX_KERNEL
  shapeIntegral *= testIntElem * trialIntElem;
#else
  shapeIntegral[0] *= testIntElem * trialIntElem;
  shapeIntegral[1] *= testIntElem * trialIntElem;
#endif

  for (vecIndex = 0; vecIndex < 4; ++vecIndex) {
#ifndef COMPLEX_KERNEL
    globalResult[number_of_cols * testIndex + trialIndex[vecIndex]] =
        ((REALTYPE *)(&shapeIntegral))[vecIndex];

#else
    globalResult[2 * (number_of_cols * testIndex + trialIndex[vecIndex])] =
        ((REALTYPE *)(&shapeIntegral[0]))[vecIndex];
    globalResult[2 * (number_of_cols * testIndex + trialIndex[vecIndex]) + 1] =
        ((REALTYPE *)(&shapeIntegral[1]))[vecIndex];
#endif
  }
}
