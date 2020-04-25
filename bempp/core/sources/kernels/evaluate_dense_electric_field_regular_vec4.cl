#include "bempp_base_types.h"
#include "bempp_helpers.h"
#include "bempp_spaces.h"
#include "kernels.h"

__kernel __attribute__((vec_type_hint(REALTYPE4)))
void evaluate_dense_electric_field_regular(
    __global uint *testIndices, __global uint *trialIndices,
    __global int *testNormalSigns, __global int *trialNormalSigns,
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
  size_t trialIndex[4] = {trialIndices[offset + 4 * (gid[1] - offset) + 0],
                          trialIndices[offset + 4 * (gid[1] - offset) + 1],
                          trialIndices[offset + 4 * (gid[1] - offset) + 2],
                          trialIndices[offset + 4 * (gid[1] - offset) + 3]};

  size_t testQuadIndex;
  size_t trialQuadIndex;
  size_t i;
  size_t j;
  size_t k;
  size_t globalRowIndex;
  size_t globalColIndex;

  REALTYPE3 testGlobalPoint;
  REALTYPE4 trialGlobalPoint[3];

  REALTYPE3 testCorners[3];
  REALTYPE4 trialCorners[3][3];

  uint testElement[3];
  uint trialElement[4][3];

  uint myTestLocal2Global[3];
  uint myTrialLocal2Global[4][3];

  REALTYPE myTestLocalMultipliers[3];
  REALTYPE myTrialLocalMultipliers[4][3];

  REALTYPE3 testJac[2];
  REALTYPE4 trialJac[2][3];

  REALTYPE3 testNormal;
  REALTYPE4 trialNormal[3];

  REALTYPE2 testPoint;
  REALTYPE2 trialPoint;

  REALTYPE testIntElem;
  REALTYPE4 trialIntElem;
  REALTYPE testValue[3][2];
  REALTYPE trialValue[3][2];
  REALTYPE3 testElementValue[3];
  REALTYPE4 trialElementValue[3][3];
  REALTYPE testEdgeLength[3];
  REALTYPE4 trialEdgeLength[3];

  REALTYPE4 kernelValue[2];
  REALTYPE4 tempFactor[2];
  REALTYPE4 tempResultFirstComponent[3][3][2];
  REALTYPE4 tempResultSecondComponent[2];
  REALTYPE4 shapeIntegralFirstComponent[3][3][2];
  REALTYPE4 shapeIntegralSecondComponent[2];
  REALTYPE shiftedWavenumber[2] = {M_ZERO, M_ZERO};
  REALTYPE inverseShiftedWavenumber[2] = {M_ZERO, M_ZERO};
  REALTYPE4 divergenceProduct;

  REALTYPE4 shapeIntegral[3][3][2];


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
      shapeIntegralFirstComponent[i][j][0] = M_ZERO;
      shapeIntegralFirstComponent[i][j][1] = M_ZERO;
    }
  shapeIntegralSecondComponent[0] = M_ZERO;
  shapeIntegralSecondComponent[1] = M_ZERO;

  getCorners(testGrid, testIndex, testCorners);
  getCornersVec4(trialGrid, trialIndex, trialCorners);

  getElement(testConnectivity, testIndex, testElement);
  getElementVec4(trialConnectivity, trialIndex, trialElement);

  getLocal2Global(testLocal2Global, testIndex, myTestLocal2Global, 3);
  getLocal2GlobalVec4(trialLocal2Global, trialIndex, &myTrialLocal2Global[0][0],
                      3);

  getLocalMultipliers(testLocalMultipliers, testIndex, myTestLocalMultipliers,
                      3);
  getLocalMultipliersVec4(trialLocalMultipliers, trialIndex,
                          &myTrialLocalMultipliers[0][0], 3);

  getJacobian(testCorners, testJac);
  getJacobianVec4(trialCorners, trialJac);

  getNormalAndIntegrationElement(testJac, &testNormal, &testIntElem);
  getNormalAndIntegrationElementVec4(trialJac, trialNormal, &trialIntElem);

  computeEdgeLength(testCorners, testEdgeLength);
  computeEdgeLengthVec4(trialCorners, trialEdgeLength);

  updateNormals(testIndex, testNormalSigns, &testNormal);
  updateNormalsVec4(trialIndex, trialNormalSigns, trialNormal);

  for (testQuadIndex = 0; testQuadIndex < NUMBER_OF_QUAD_POINTS;
       ++testQuadIndex) {
    testPoint = (REALTYPE2)(quadPoints[2 * testQuadIndex], quadPoints[2 * testQuadIndex + 1]);
    testGlobalPoint = getGlobalPoint(testCorners, &testPoint);
    BASIS(TEST, evaluate)
    (&testPoint, &testValue[0][0]);
    getPiolaTransform(testIntElem, testJac, testValue, testElementValue);

    for (i = 0; i < 3; ++i)
      for (j = 0; j < 3; ++j) {
        tempResultFirstComponent[i][j][0] = M_ZERO;
        tempResultFirstComponent[i][j][1] = M_ZERO;
      }
    tempResultSecondComponent[0] = M_ZERO;
    tempResultSecondComponent[1] = M_ZERO;

    for (trialQuadIndex = 0; trialQuadIndex < NUMBER_OF_QUAD_POINTS;
         ++trialQuadIndex) {
      trialPoint = (REALTYPE2)(quadPoints[2 * trialQuadIndex], quadPoints[2 * trialQuadIndex + 1]);
      getGlobalPointVec4(trialCorners, &trialPoint, trialGlobalPoint);
      BASIS(TRIAL, evaluate)
      (&trialPoint, &trialValue[0][0]);
      getPiolaTransformVec4(trialIntElem, trialJac, trialValue,
                            trialElementValue);
      KERNEL(vec4)
      (testGlobalPoint, trialGlobalPoint, testNormal, trialNormal, kernelValue);

      tempFactor[0] = kernelValue[0] * quadWeights[trialQuadIndex];
      tempFactor[1] = kernelValue[1] * quadWeights[trialQuadIndex];
      for (i = 0; i < 3; ++i)
        for (j = 0; j < 3; ++j) {
          tempResultFirstComponent[i][j][0] +=
              tempFactor[0] * trialElementValue[i][j];
          tempResultFirstComponent[i][j][1] +=
              tempFactor[1] * trialElementValue[i][j];
        }
      tempResultSecondComponent[0] += tempFactor[0];
      tempResultSecondComponent[1] += tempFactor[1];
    }

    for (i = 0; i < 3; ++i)
      for (j = 0; j < 3; ++j)
        for (k = 0; k < 3; ++k) {
          shapeIntegralFirstComponent[i][j][0] +=
              quadWeights[testQuadIndex] * testElementValue[i][k] *
              tempResultFirstComponent[j][k][0];
          shapeIntegralFirstComponent[i][j][1] +=
              quadWeights[testQuadIndex] * testElementValue[i][k] *
              tempResultFirstComponent[j][k][1];
        }
    shapeIntegralSecondComponent[0] +=
        quadWeights[testQuadIndex] * tempResultSecondComponent[0];
    shapeIntegralSecondComponent[1] +=
        quadWeights[testQuadIndex] * tempResultSecondComponent[1];
  }

  divergenceProduct = M_TWO * M_TWO / testIntElem / trialIntElem;

  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j) {
      shapeIntegral[i][j][0] =
          -(shiftedWavenumber[0] * shapeIntegralFirstComponent[i][j][0] -
            shiftedWavenumber[1] * shapeIntegralFirstComponent[i][j][1]);
      shapeIntegral[i][j][1] =
          -(shiftedWavenumber[0] * shapeIntegralFirstComponent[i][j][1] +
            shiftedWavenumber[1] * shapeIntegralFirstComponent[i][j][0]);
      shapeIntegral[i][j][0] -=
          divergenceProduct *
          (inverseShiftedWavenumber[0] * shapeIntegralSecondComponent[0] -
           inverseShiftedWavenumber[1] * shapeIntegralSecondComponent[1]);
      shapeIntegral[i][j][1] -=
          divergenceProduct *
          (inverseShiftedWavenumber[0] * shapeIntegralSecondComponent[1] +
           inverseShiftedWavenumber[1] * shapeIntegralSecondComponent[0]);
      shapeIntegral[i][j][0] *= testEdgeLength[i] * trialEdgeLength[j];
      shapeIntegral[i][j][1] *= testEdgeLength[i] * trialEdgeLength[j];
    }

  for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
    for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
      shapeIntegral[i][j][0] *=
          (testIntElem * myTestLocalMultipliers[i]) * trialIntElem;
      shapeIntegral[i][j][1] *=
          (testIntElem * myTestLocalMultipliers[i]) * trialIntElem;
    }

  for (int vecIndex = 0; vecIndex < 4; ++vecIndex)
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
