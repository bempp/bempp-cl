#include "bempp_base_types.h"
#include "bempp_helpers.h"
#include "bempp_spaces.h"
#include "kernels.h"

__kernel void evaluate_dense_regular(__global REALTYPE *testGrid,
                                     __global REALTYPE *trialGrid,
                                     __constant REALTYPE* quadPoints,
                                     __constant REALTYPE *quadWeights,
                                     __global REALTYPE *globalResult,
                                     uint numberOfCols) {
  /* Variable declarations */

  size_t gid[2] = {get_global_id(0), get_global_id(1)};

  size_t testQuadIndex;
  size_t trialQuadIndex;
  size_t i;
  size_t j;

  REALTYPE3 testGlobalPoint;
  REALTYPE3 trialGlobalPoint;

  REALTYPE3 testCorners[3];
  REALTYPE3 trialCorners[3];

  REALTYPE3 testJac[2];
  REALTYPE3 trialJac[2];

  REALTYPE3 testNormal;
  REALTYPE3 trialNormal;

  REALTYPE2 testPoint;
  REALTYPE2 trialPoint;

  REALTYPE testIntElem;
  REALTYPE trialIntElem;

#ifndef COMPLEX_KERNEL
  REALTYPE kernelValue;
  REALTYPE tempResult;
  REALTYPE shapeIntegral;
#else
  REALTYPE kernelValue[2];
  REALTYPE tempResult[2];
  REALTYPE shapeIntegral[2];
#endif

#ifndef COMPLEX_KERNEL
  shapeIntegral = M_ZERO;
#else
  shapeIntegral[0] = M_ZERO;
  shapeIntegral[1] = M_ZERO;
#endif

  getCorners(testGrid, gid[0], testCorners);
  getCorners(trialGrid, gid[1], trialCorners);

  getJacobian(testCorners, testJac);
  getJacobian(trialCorners, trialJac);

  getNormalAndIntegrationElement(testJac, &testNormal, &testIntElem);
  getNormalAndIntegrationElement(trialJac, &trialNormal, &trialIntElem);

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
      trialGlobalPoint = getGlobalPoint(trialCorners, &trialPoint);
#ifndef COMPLEX_KERNEL
      KERNEL(novec)
      (testGlobalPoint, trialGlobalPoint, testNormal, trialNormal,
       &kernelValue);
      tempResult += quadWeights[trialQuadIndex] * kernelValue;
#else
      KERNEL(novec)
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
  globalResult[number_of_cols * gid[0] + gid[1]] =
      shapeIntegral * testIntElem * trialIntElem;

#else
  globalResult[2 * (number_of_cols * gid[0] + gid[1])] =
      shapeIntegral[0] * testIntElem * trialIntElem;
  globalResult[2 * (number_of_cols * gid[0] + gid[1]) + 1] =
      shapeIntegral[1] * testIntElem * trialIntElem;
#endif
}
