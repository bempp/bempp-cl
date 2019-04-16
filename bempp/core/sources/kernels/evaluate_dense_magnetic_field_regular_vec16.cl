#include "bempp_base_types.h"
#include "bempp_helpers.h"
#include "bempp_spaces.h"
#include "kernels.h"

__kernel __attribute__((vec_type_hint(REALTYPE16))) void
evaluate_dense_magnetic_field_regular(
    __global uint *testIndices, __global uint *trialIndices,
    __global REALTYPE *testGrid, __global REALTYPE *trialGrid,
    __global uint *testConnectivity, __global uint *trialConnectivity,
    __global uint *testLocal2Global, __global uint *trialLocal2Global,
    __global REALTYPE *testLocalMultipliers,
    __global REALTYPE *trialLocalMultipliers, __constant REALTYPE* quadPoints,
    __constant REALTYPE *quadWeights, __global REALTYPE *globalResult,
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

  uint myTestLocal2Global[3];
  uint myTrialLocal2Global[16][3];

  REALTYPE myTestLocalMultipliers[3];
  REALTYPE myTrialLocalMultipliers[16][3];

  REALTYPE3 testJac[2];
  REALTYPE16 trialJac[2][3];

  REALTYPE3 testNormal;
  REALTYPE16 trialNormal[3];

  REALTYPE2 testPoint;
  REALTYPE2 trialPoint;

  REALTYPE testIntElem;
  REALTYPE16 trialIntElem;
  REALTYPE testValue[3][2];
  REALTYPE trialValue[3][2];
  REALTYPE3 testElementValue[3];
  REALTYPE16 trialElementValue[3][3];
  REALTYPE testEdgeLength[3];
  REALTYPE16 trialEdgeLength[3];

  REALTYPE16 kernelValue[3][2];

  REALTYPE16 shapeIntegral[3][3][2];

  REALTYPE16 tempFactor[3][2];
  REALTYPE16 tempResult[3][3][2];

  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j) {
      shapeIntegral[i][j][0] = M_ZERO;
      shapeIntegral[i][j][1] = M_ZERO;
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

  getLocalMultipliers(testLocalMultipliers, testIndex, myTestLocalMultipliers,
                      NUMBER_OF_TEST_SHAPE_FUNCTIONS);
  getLocalMultipliersVec16(trialLocalMultipliers, trialIndex,
                           &myTrialLocalMultipliers[0][0],
                           NUMBER_OF_TRIAL_SHAPE_FUNCTIONS);

  getJacobian(testCorners, testJac);
  getJacobianVec16(trialCorners, trialJac);

  getNormalAndIntegrationElement(testJac, &testNormal, &testIntElem);
  getNormalAndIntegrationElementVec16(trialJac, trialNormal, &trialIntElem);

  computeEdgeLength(testCorners, testEdgeLength);
  computeEdgeLengthVec16(trialCorners, trialEdgeLength);

  for (testQuadIndex = 0; testQuadIndex < NUMBER_OF_QUAD_POINTS;
       ++testQuadIndex) {
    testPoint = (REALTYPE2)(quadPoints[2 * testQuadIndex], quadPoints[2 * testQuadIndex + 1]);
    testGlobalPoint = getGlobalPoint(testCorners, &testPoint);
    BASIS(TEST, evaluate)
    (&testPoint, &testValue[0][0]);
    getPiolaTransform(testIntElem, testJac, testValue, testElementValue);

    for (i = 0; i < 3; ++i)
      for (j = 0; j < 3; ++j)
        for (k = 0; k < 2; ++k) tempResult[i][j][k] = M_ZERO;

    for (trialQuadIndex = 0; trialQuadIndex < NUMBER_OF_QUAD_POINTS;
         ++trialQuadIndex) {
      trialPoint = (REALTYPE2)(quadPoints[2 * trialQuadIndex], quadPoints[2 * trialQuadIndex + 1]);
      getGlobalPointVec16(trialCorners, &trialPoint, trialGlobalPoint);
      BASIS(TRIAL, evaluate)
      (&trialPoint, &trialValue[0][0]);
      getPiolaTransformVec16(trialIntElem, trialJac, trialValue,
                             trialElementValue);

      KERNEL(vec16)
      (testGlobalPoint, trialGlobalPoint, testNormal, trialNormal, kernelValue);

      for (j = 0; j < 3; ++j)
        for (k = 0; k < 2; ++k)
          tempFactor[j][k] = kernelValue[j][k] * quadWeights[trialQuadIndex];

      for (j = 0; j < 3; ++j)
        for (k = 0; k < 2; ++k) {
          tempResult[j][0][k] += tempFactor[1][k] * trialElementValue[j][2] -
                                 tempFactor[2][k] * trialElementValue[j][1];
          tempResult[j][1][k] += tempFactor[2][k] * trialElementValue[j][0] -
                                 tempFactor[0][k] * trialElementValue[j][2];
          tempResult[j][2][k] += tempFactor[0][k] * trialElementValue[j][1] -
                                 tempFactor[1][k] * trialElementValue[j][0];
        }
    }

    for (i = 0; i < 3; ++i)
      for (j = 0; j < 3; ++j)
        for (k = 0; k < 2; ++k)
          shapeIntegral[i][j][k] -=
              quadWeights[testQuadIndex] *
              (testElementValue[i].x * tempResult[j][0][k] +
               testElementValue[i].y * tempResult[j][1][k] +
               testElementValue[i].z * tempResult[j][2][k]);
  }

  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j)
      for (k = 0; k < 2; ++k)
        shapeIntegral[i][j][k] *= testEdgeLength[i] * trialEdgeLength[j] *
                                  testIntElem * trialIntElem *
                                  myTestLocalMultipliers[i];

  for (int vecIndex = 0; vecIndex < 16; ++vecIndex)
    if (!elementsAreAdjacent(testElement, trialElement[vecIndex],
                             gridsAreDisjoint)) {
      for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
        for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
          globalRowIndex = myTestLocal2Global[i];
          globalColIndex = myTrialLocal2Global[vecIndex][j];
          globalResult[2 * (globalRowIndex * nTrial + globalColIndex)] +=
              ((REALTYPE *)(&shapeIntegral[i][j][0]))[vecIndex] *
              myTrialLocalMultipliers[vecIndex][j];
          globalResult[2 * (globalRowIndex * nTrial + globalColIndex) + 1] +=
              ((REALTYPE *)(&shapeIntegral[i][j][1]))[vecIndex] *
              myTrialLocalMultipliers[vecIndex][j];
        }
    }
}
