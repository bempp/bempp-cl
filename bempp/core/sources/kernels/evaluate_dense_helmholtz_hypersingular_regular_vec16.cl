#include "bempp_base_types.h"
#include "bempp_helpers.h"
#include "bempp_spaces.h"
#include "kernels.h"

__kernel __attribute__((vec_type_hint(REALTYPE16))) void evaluate_dense_regular(
    __global uint* testIndices, __global uint* trialIndices,
    __global int *testNormalSigns, __global int *trialNormalSigns,
    __global REALTYPE* testGrid, __global REALTYPE* trialGrid,
    __global uint* testConnectivity, __global uint* trialConnectivity,
    __global uint* testLocal2Global, __global uint* trialLocal2Global,
    __global REALTYPE* testLocalMultipliers,
    __global REALTYPE* trialLocalMultipliers, __constant REALTYPE* quadPoints,
    __constant REALTYPE* quadWeights, __global REALTYPE* globalResult,
    __global REALTYPE* kernel_parameters,
    int nTest, int nTrial, char gridsAreDisjoint) {
  /* Variable declarations */

  size_t gid[2] = {get_global_id(0), get_global_id(1)};
  size_t offset = get_global_offset(1);

  size_t testIndex = testIndices[gid[0]];
  size_t trialIndex[16] = {trialIndices[offset + 16 * (gid[1] - offset) + 0],
                           trialIndices[offset + 16 * (gid[1] - offset) + 1],
                           trialIndices[offset + 16 * (gid[1] - offset) + 2],
                           trialIndices[offset + 16 * (gid[1] - offset) + 3],
                           trialIndices[offset + 16 * (gid[1] - offset) + 4],
                           trialIndices[offset + 16 * (gid[1] - offset) + 5],
                           trialIndices[offset + 16 * (gid[1] - offset) + 6],
                           trialIndices[offset + 16 * (gid[1] - offset) + 7],
                           trialIndices[offset + 16 * (gid[1] - offset) + 8],
                           trialIndices[offset + 16 * (gid[1] - offset) + 9],
                           trialIndices[offset + 16 * (gid[1] - offset) + 10],
                           trialIndices[offset + 16 * (gid[1] - offset) + 11],
                           trialIndices[offset + 16 * (gid[1] - offset) + 12],
                           trialIndices[offset + 16 * (gid[1] - offset) + 13],
                           trialIndices[offset + 16 * (gid[1] - offset) + 14],
                           trialIndices[offset + 16 * (gid[1] - offset) + 15]};
  size_t testQuadIndex;
  size_t trialQuadIndex;
  size_t i;
  size_t j;
  size_t k;
  size_t globalRowIndex;
  size_t globalColIndex;

  REALTYPE3 testGlobalPoint;
  REALTYPE16 trialGlobalPoint[3];

  REALTYPE3 testCorners[3];
  REALTYPE16 trialCorners[3][3];

  uint testElement[3];
  uint trialElement[16][3];

  uint myTestLocal2Global[NUMBER_OF_TEST_SHAPE_FUNCTIONS];
  uint myTrialLocal2Global[16][NUMBER_OF_TRIAL_SHAPE_FUNCTIONS];

  REALTYPE myTestLocalMultipliers[NUMBER_OF_TEST_SHAPE_FUNCTIONS];
  REALTYPE myTrialLocalMultipliers[16][NUMBER_OF_TRIAL_SHAPE_FUNCTIONS];

  REALTYPE3 testJac[2];
  REALTYPE16 trialJac[2][3];

  REALTYPE3 testNormal;
  REALTYPE16 trialNormal[3];

  REALTYPE2 testPoint;
  REALTYPE2 trialPoint;

  REALTYPE testIntElem;
  REALTYPE16 trialIntElem;
  REALTYPE testValue[NUMBER_OF_TEST_SHAPE_FUNCTIONS];
  REALTYPE trialValue[NUMBER_OF_TRIAL_SHAPE_FUNCTIONS];

  REALTYPE testInv[2][2];
  REALTYPE16 trialInv[2][2];

  REALTYPE16 trialCurl[3][3];
  REALTYPE3 testCurl[3];

  REALTYPE16 basisProduct[3][3];

  REALTYPE16 trialInvBasis[2][3];
  REALTYPE16 trialElementGradient[3][3];

  REALTYPE16 normalProduct;

#ifndef COMPLEX_KERNEL
  REALTYPE16 kernelValue;
  REALTYPE16 tempResult[NUMBER_OF_TRIAL_SHAPE_FUNCTIONS];
  REALTYPE16 tempFactor;
  REALTYPE16 shapeIntegral[NUMBER_OF_TEST_SHAPE_FUNCTIONS]
                          [NUMBER_OF_TRIAL_SHAPE_FUNCTIONS];

  REALTYPE16 firstTermIntegral;
  REALTYPE16 tempFirstTerm;

#else
  REALTYPE16 tmp[2];
  REALTYPE16 kernelValue[2];
  REALTYPE16 tempResult[NUMBER_OF_TRIAL_SHAPE_FUNCTIONS][2];
  REALTYPE16 tempFactor[2];
  REALTYPE16 shapeIntegral[NUMBER_OF_TEST_SHAPE_FUNCTIONS]
                          [NUMBER_OF_TRIAL_SHAPE_FUNCTIONS][2];
  REALTYPE16 firstTermIntegral[2];
  REALTYPE16 tempFirstTerm[2];
  REALTYPE wavenumberProduct[2];

#endif

#ifndef COMPLEX_KERNEL
  firstTermIntegral = M_ZERO;
#else
  firstTermIntegral[0] = M_ZERO;
  firstTermIntegral[1] = M_ZERO;
#endif

  for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
    for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
#ifndef COMPLEX_KERNEL
      shapeIntegral[i][j] = M_ZERO;
#else
      shapeIntegral[i][j][0] = M_ZERO;
      shapeIntegral[i][j][1] = M_ZERO;
#endif
    }

  getCorners(testGrid, testIndex, testCorners);
  getCornersVec16(trialGrid, trialIndex, trialCorners);

  getElement(testConnectivity, testIndex, testElement);
  getElementVec16(trialConnectivity, trialIndex, trialElement);

  getLocal2Global(testLocal2Global, testIndex, myTestLocal2Global,
                  NUMBER_OF_TEST_SHAPE_FUNCTIONS);
  getLocal2GlobalVec16(trialLocal2Global, trialIndex,
                       &myTrialLocal2Global[0][0],
                       NUMBER_OF_TRIAL_SHAPE_FUNCTIONS);

  getJacobian(testCorners, testJac);
  getJacobianVec16(trialCorners, trialJac);

  getNormalAndIntegrationElement(testJac, &testNormal, &testIntElem);
  getNormalAndIntegrationElementVec16(trialJac, trialNormal, &trialIntElem);

  getLocalMultipliers(testLocalMultipliers, testIndex, myTestLocalMultipliers,
                      NUMBER_OF_TEST_SHAPE_FUNCTIONS);
  getLocalMultipliersVec16(trialLocalMultipliers, trialIndex,
                          &myTrialLocalMultipliers[0][0],
                          NUMBER_OF_TRIAL_SHAPE_FUNCTIONS);

  updateNormals(testIndex, testNormalSigns, &testNormal);
  updateNormalsVec16(trialIndex, trialNormalSigns, trialNormal);

  testInv[0][0] = dot(testJac[1], testJac[1]);
  testInv[1][1] = dot(testJac[0], testJac[0]);
  testInv[0][1] = -dot(testJac[0], testJac[1]);
  testInv[1][0] = testInv[0][1];

  trialInv[0][0] = M_ZERO;
  trialInv[1][1] = M_ZERO;
  trialInv[0][1] = M_ZERO;

  for (i = 0; i < 3; ++i) {
    trialInv[0][0] += trialJac[1][i] * trialJac[1][i];
    trialInv[0][1] -= trialJac[0][i] * trialJac[1][i];
    trialInv[1][1] += trialJac[0][i] * trialJac[0][i];
  }

  trialInv[1][0] = trialInv[0][1];

  testCurl[0] =
      cross(testNormal, testJac[0] * (-testInv[0][0] - testInv[0][1]) +
                            testJac[1] * (-testInv[1][0] - testInv[1][1])) /
      (testIntElem * testIntElem);
  testCurl[1] = cross(testNormal,
                      testJac[0] * testInv[0][0] + testJac[1] * testInv[1][0]) /
                (testIntElem * testIntElem);
  testCurl[2] = cross(testNormal,
                      testJac[0] * testInv[0][1] + testJac[1] * testInv[1][1]) /
                (testIntElem * testIntElem);

  for (i = 0; i < 2; ++i) {
    trialInvBasis[i][0] = -trialInv[i][0] - trialInv[i][1];
    trialInvBasis[i][1] = trialInv[i][0];
    trialInvBasis[i][2] = trialInv[i][1];
  }

  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j) {
      trialElementGradient[j][i] = 0;
      for (k = 0; k < 2; ++k)
        trialElementGradient[j][i] += trialJac[k][i] * trialInvBasis[k][j];
    }

  for (i = 0; i < 3; ++i) {
    trialCurl[i][0] = (trialNormal[1] * trialElementGradient[i][2] -
                       trialNormal[2] * trialElementGradient[i][1]) /
                      (trialIntElem * trialIntElem);
    trialCurl[i][1] = (trialNormal[2] * trialElementGradient[i][0] -
                       trialNormal[0] * trialElementGradient[i][2]) /
                      (trialIntElem * trialIntElem);
    trialCurl[i][2] = (trialNormal[0] * trialElementGradient[i][1] -
                       trialNormal[1] * trialElementGradient[i][0]) /
                      (trialIntElem * trialIntElem);
  }

  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j) {
      basisProduct[i][j] = testCurl[i].x * trialCurl[j][0] +
                           testCurl[i].y * trialCurl[j][1] +
                           testCurl[i].z * trialCurl[j][2];
    }

  normalProduct = testNormal.x * trialNormal[0] +
                  testNormal.y * trialNormal[1] + testNormal.z * trialNormal[2];

  for (testQuadIndex = 0; testQuadIndex < NUMBER_OF_QUAD_POINTS;
       ++testQuadIndex) {
    testPoint = (REALTYPE2)(quadPoints[2 * testQuadIndex], quadPoints[2 * testQuadIndex + 1]);
    testGlobalPoint = getGlobalPoint(testCorners, &testPoint);
    BASIS(TEST, evaluate)(&testPoint, &testValue[0]);

#ifndef COMPLEX_KERNEL
    for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
      tempResult[j] = M_ZERO;
    }
    tempFirstTerm = M_ZERO;
#else
    for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
      tempResult[j][0] = M_ZERO;
      tempResult[j][1] = M_ZERO;
    }

    tempFirstTerm[0] = M_ZERO;
    tempFirstTerm[1] = M_ZERO;
#endif

    for (trialQuadIndex = 0; trialQuadIndex < NUMBER_OF_QUAD_POINTS;
         ++trialQuadIndex) {
      trialPoint = (REALTYPE2)(quadPoints[2 * trialQuadIndex], quadPoints[2 * trialQuadIndex + 1]);
      getGlobalPointVec16(trialCorners, &trialPoint, trialGlobalPoint);
      BASIS(TRIAL, evaluate)(&trialPoint, &trialValue[0]);
#ifndef COMPLEX_KERNEL
      KERNEL(vec16)
      (testGlobalPoint, trialGlobalPoint, testNormal, trialNormal, kernel_parameters,
       &kernelValue);
      tempFactor = quadWeights[trialQuadIndex] * kernelValue;
      tempFirstTerm += tempFactor;
      for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j)
        tempResult[j] += trialValue[j] * tempFactor;
#else
      KERNEL(vec16)
      (testGlobalPoint, trialGlobalPoint, testNormal, trialNormal, kernel_parameters, kernelValue);
      tempFactor[0] = quadWeights[trialQuadIndex] * kernelValue[0];
      tempFactor[1] = quadWeights[trialQuadIndex] * kernelValue[1];
      tempFirstTerm[0] += tempFactor[0];
      tempFirstTerm[1] += tempFactor[1];
      for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
        tempResult[j][0] += trialValue[j] * tempFactor[0];
        tempResult[j][1] += trialValue[j] * tempFactor[1];
      }

#endif
    }
#ifndef COMPLEX_KERNEL
    firstTermIntegral += tempFirstTerm * quadWeights[testQuadIndex];
    for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
      for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j)
        shapeIntegral[i][j] +=
            tempResult[j] * quadWeights[testQuadIndex] * testValue[i];
#else
    firstTermIntegral[0] += tempFirstTerm[0] * quadWeights[testQuadIndex];
    firstTermIntegral[1] += tempFirstTerm[1] * quadWeights[testQuadIndex];
    for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
      for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
        shapeIntegral[i][j][0] +=
            tempResult[j][0] * quadWeights[testQuadIndex] * testValue[i];
        shapeIntegral[i][j][1] +=
            tempResult[j][1] * quadWeights[testQuadIndex] * testValue[i];
      }
#endif
  }

#ifndef COMPLEX_KERNEL
  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j)
      shapeIntegral[i][j] =
          (OMEGA * OMEGA * shapeIntegral[i][j] * normalProduct +
           firstTermIntegral * basisProduct[i][j]) *
          testIntElem * trialIntElem;

#else

  wavenumberProduct[0] = kernel_parameters[0] * kernel_parameters[0] -
                         kernel_parameters[1] * kernel_parameters[1];
  wavenumberProduct[1] = M_TWO * kernel_parameters[0] * kernel_parameters[1];

  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j) {
      tmp[0] = shapeIntegral[i][j][0];
      tmp[1] = shapeIntegral[i][j][1];
      shapeIntegral[i][j][0] =
          (-(wavenumberProduct[0] * tmp[0] -
             wavenumberProduct[1] * tmp[1]) *
               normalProduct +
           firstTermIntegral[0] * basisProduct[i][j]) *
          testIntElem * trialIntElem;
      shapeIntegral[i][j][1] =
          (-(wavenumberProduct[0] * tmp[1] +
             wavenumberProduct[1] * tmp[0]) *
               normalProduct +
           firstTermIntegral[1] * basisProduct[i][j]) *
          testIntElem * trialIntElem;
    }

#endif

  for (int vecIndex = 0; vecIndex < 16; ++vecIndex)
    if (!elementsAreAdjacent(testElement, trialElement[vecIndex],
                             gridsAreDisjoint)) {
      for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
        for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
          globalRowIndex = myTestLocal2Global[i];
          globalColIndex = myTrialLocal2Global[vecIndex][j];
#ifndef COMPLEX_KERNEL
          globalResult[globalRowIndex * nTrial + globalColIndex] +=
              ((REALTYPE*)(&shapeIntegral[i][j]))[vecIndex] *
              myTestLocalMultipliers[i] * myTrialLocalMultipliers[vecIndex][j];
#else
          globalResult[2 * (globalRowIndex * nTrial + globalColIndex)] +=
              ((REALTYPE*)(&shapeIntegral[i][j][0]))[vecIndex] *
              myTestLocalMultipliers[i] * myTrialLocalMultipliers[vecIndex][j];
          globalResult[2 * (globalRowIndex * nTrial + globalColIndex) + 1] +=
              ((REALTYPE*)(&shapeIntegral[i][j][1]))[vecIndex] *
              myTestLocalMultipliers[i] * myTrialLocalMultipliers[vecIndex][j];
#endif
        }
    }
}
