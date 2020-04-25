#include "bempp_base_types.h"
#include "bempp_helpers.h"
#include "bempp_spaces.h"
#include "kernels.h"

__kernel void evaluate_dense_regular(
    __global uint* testIndices, __global uint* trialIndices,
    __global int *testNormalSigns, __global int *trialNormalSigns,
    __global REALTYPE* testGrid, __global REALTYPE* trialGrid,
    __global uint* testConnectivity, __global uint* trialConnectivity,
    __global uint* testLocal2Global, __global uint* trialLocal2Global,
    __global REALTYPE* testLocalMultipliers,
    __global REALTYPE* trialLocalMultipliers, __constant REALTYPE* quadPoints,
    __constant REALTYPE* quadWeights, 
    __constant REALTYPE* collocationPoints, 
    __global REALTYPE* globalResult,
    int nTest, int nTrial, char gridsAreDisjoint) {
  /* Variable declarations */

  size_t gid[2] = {get_global_id(0), get_global_id(1)};
  size_t offset = get_global_offset(1);

  size_t testIndex = testIndices[gid[0]];
  size_t trialIndex[4] = {trialIndices[offset + 4 * (gid[1] - offset) + 0],
                          trialIndices[offset + 4 * (gid[1] - offset) + 1],
                          trialIndices[offset + 4 * (gid[1] - offset) + 2],
                          trialIndices[offset + 4 * (gid[1] - offset) + 3]};
  size_t collocationIndex;
  size_t trialQuadIndex;
  size_t i;
  size_t j;
  size_t globalRowIndex;
  size_t globalColIndex;

  REALTYPE3 testGlobalPoint;
  REALTYPE4 trialGlobalPoint[3];

  REALTYPE3 testCorners[3];
  REALTYPE4 trialCorners[3][3];

  uint testElement[3];
  uint trialElement[4][3];

  uint myTestLocal2Global[NUMBER_OF_TEST_SHAPE_FUNCTIONS];
  uint myTrialLocal2Global[4][NUMBER_OF_TRIAL_SHAPE_FUNCTIONS];

  REALTYPE myTestLocalMultipliers[NUMBER_OF_TEST_SHAPE_FUNCTIONS];
  REALTYPE myTrialLocalMultipliers[4][NUMBER_OF_TRIAL_SHAPE_FUNCTIONS];

  REALTYPE3 testJac[2];
  REALTYPE4 trialJac[2][3];

  REALTYPE3 testNormal;
  REALTYPE4 trialNormal[3];

  REALTYPE2 testPoint;
  REALTYPE2 trialPoint;

  REALTYPE testIntElem;
  REALTYPE4 trialIntElem;
  REALTYPE trialValue[NUMBER_OF_TRIAL_SHAPE_FUNCTIONS];

#ifndef COMPLEX_KERNEL
  REALTYPE4 kernelValue;
  REALTYPE4 tempResult[NUMBER_OF_TRIAL_SHAPE_FUNCTIONS];
  REALTYPE4 tempFactor;
  REALTYPE4 shapeIntegral[NUMBER_OF_TEST_SHAPE_FUNCTIONS]
                         [NUMBER_OF_TRIAL_SHAPE_FUNCTIONS];
#else
  REALTYPE4 kernelValue[2];
  REALTYPE4 tempResult[NUMBER_OF_TRIAL_SHAPE_FUNCTIONS][2];
  REALTYPE4 tempFactor[2];
  REALTYPE4 shapeIntegral[NUMBER_OF_TEST_SHAPE_FUNCTIONS]
                         [NUMBER_OF_TRIAL_SHAPE_FUNCTIONS][2];
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
  getCornersVec4(trialGrid, trialIndex, trialCorners);

  getElement(testConnectivity, testIndex, testElement);
  getElementVec4(trialConnectivity, trialIndex, trialElement);

  getLocal2Global(testLocal2Global, testIndex, myTestLocal2Global,
                  NUMBER_OF_TEST_SHAPE_FUNCTIONS);
  getLocal2GlobalVec4(trialLocal2Global, trialIndex, &myTrialLocal2Global[0][0],
                      NUMBER_OF_TRIAL_SHAPE_FUNCTIONS);

  getLocalMultipliers(testLocalMultipliers, testIndex, myTestLocalMultipliers,
                      NUMBER_OF_TEST_SHAPE_FUNCTIONS);
  getLocalMultipliersVec4(trialLocalMultipliers, trialIndex,
                          &myTrialLocalMultipliers[0][0],
                          NUMBER_OF_TRIAL_SHAPE_FUNCTIONS);

  getJacobian(testCorners, testJac);
  getJacobianVec4(trialCorners, trialJac);

  getNormalAndIntegrationElement(testJac, &testNormal, &testIntElem);
  getNormalAndIntegrationElementVec4(trialJac, trialNormal, &trialIntElem);

  updateNormals(testIndex, testNormalSigns, &testNormal);
  updateNormalsVec4(trialIndex, trialNormalSigns, trialNormal);

  for (collocationIndex = 0; collocationIndex < NUMBER_OF_TEST_SHAPE_FUNCTIONS;
          ++collocationIndex) {

    testPoint = (REALTYPE2)(collocationPoints[2 * collocationIndex], collocationPoints[2 * collocationIndex + 1]);
    testGlobalPoint = getGlobalPoint(testCorners, &testPoint);

    for (trialQuadIndex = 0; trialQuadIndex < NUMBER_OF_QUAD_POINTS;
         ++trialQuadIndex) {
      trialPoint = (REALTYPE2)(quadPoints[2 * trialQuadIndex], quadPoints[2 * trialQuadIndex + 1]);
      getGlobalPointVec4(trialCorners, &trialPoint, trialGlobalPoint);
      BASIS(TRIAL, evaluate)(&trialPoint, &trialValue[0]);
#ifndef COMPLEX_KERNEL
      KERNEL(vec4)
      (testGlobalPoint, trialGlobalPoint, testNormal, trialNormal,
       &kernelValue);
      tempFactor = quadWeights[trialQuadIndex] * kernelValue;
      for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j)
        shapeIntegral[collocationIndex][j] += trialValue[j] * tempFactor;
#else
      KERNEL(vec4)
      (testGlobalPoint, trialGlobalPoint, testNormal, trialNormal, kernelValue);
      tempFactor[0] = quadWeights[trialQuadIndex] * kernelValue[0];
      tempFactor[1] = quadWeights[trialQuadIndex] * kernelValue[1];
      for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
        shapeIntegral[collocationIndex][j][0] += trialValue[j] * tempFactor[0];
        shapeIntegral[collocationIndex][j][1] += trialValue[j] * tempFactor[1];
      }

#endif
    }

  }

  for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
    for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
#ifndef COMPLEX_KERNEL
      shapeIntegral[i][j] *=
          myTestLocalMultipliers[i] * trialIntElem;
#else
      shapeIntegral[i][j][0] *=
          myTestLocalMultipliers[i] * trialIntElem;
      shapeIntegral[i][j][1] *=
          myTestLocalMultipliers[i] * trialIntElem;
#endif
    }

  for (int vecIndex = 0; vecIndex < 4; ++vecIndex)
    if (!elementsAreAdjacentCollocation(testElement, trialElement[vecIndex],
                             gridsAreDisjoint)) {
      for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
        for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
          globalRowIndex = myTestLocal2Global[i];
          globalColIndex = myTrialLocal2Global[vecIndex][j];
#ifndef COMPLEX_KERNEL
          globalResult[globalRowIndex * nTrial + globalColIndex] +=
              ((REALTYPE*)(&shapeIntegral[i][j]))[vecIndex] *
              myTrialLocalMultipliers[vecIndex][j];
#else
          globalResult[2 * (globalRowIndex * nTrial + globalColIndex)] +=
              ((REALTYPE*)(&shapeIntegral[i][j][0]))[vecIndex] *
              myTrialLocalMultipliers[vecIndex][j];
          globalResult[2 * (globalRowIndex * nTrial + globalColIndex) + 1] +=
              ((REALTYPE*)(&shapeIntegral[i][j][1]))[vecIndex] *
              myTrialLocalMultipliers[vecIndex][j];
#endif
        }
    }
}
