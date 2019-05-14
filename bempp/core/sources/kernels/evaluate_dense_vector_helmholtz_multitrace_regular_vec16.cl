#include "bempp_base_types.h"
#include "bempp_helpers.h"
#include "bempp_spaces.h"
#include "kernels.h"

__kernel void evaluate_dense_helmholtz_multitrace_vector_regular(
    __global uint *testIndices, __global uint *trialIndices,
    __global int *testNormalSigns, __global int *trialNormalSigns,
    __global REALTYPE *testGrid, __global REALTYPE *trialGrid,
    __global uint *testConnectivity, __global uint *trialConnectivity,
    __constant REALTYPE *quadPoints, __constant REALTYPE *quadWeights,
    __global REALTYPE *input, char gridsAreDisjoint,
    __global REALTYPE *globalResult) {
  size_t gid[2] = {get_global_id(0), get_global_id(1)};
  size_t offset = get_global_offset(1);
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

  size_t lid = get_local_id(1);
  size_t groupId = get_group_id(1);
  size_t numGroups = get_num_groups(1);
  size_t vecIndex;

  size_t testIndex = testIndices[gid[0]];

  size_t testQuadIndex;
  size_t trialQuadIndex;
  size_t i;
  size_t j;
  size_t k;
  size_t globalRowIndex;
  size_t globalColIndex;
  size_t localIndex;

  REALTYPE3 testGlobalPoint;
  REALTYPE16 trialGlobalPoint[3];

  REALTYPE3 testCorners[3];
  REALTYPE16 trialCorners[3][3];

  uint testElement[3];
  uint trialElement[16][3];

  REALTYPE3 testJac[2];
  REALTYPE16 trialJac[2][3];

  REALTYPE16 diff[3];
  REALTYPE16 dist;
  REALTYPE16 rdist;

  REALTYPE3 testNormal;
  REALTYPE16 trialNormal[3];

  REALTYPE2 testPoint;
  REALTYPE2 trialPoint;

  REALTYPE testIntElem;
  REALTYPE16 trialIntElem;

  REALTYPE16 kernelValue[2];
  REALTYPE16 kernelGradient[3][2];
  REALTYPE16 shapeIntegralSingleLayer[3][3][2];
  REALTYPE16 shapeIntegralDoubleLayer[3][3][2];
  REALTYPE16 shapeIntegralAdjDoubleLayer[3][3][2];
  REALTYPE16 shapeIntegralHypersingular[3][3][2];
  REALTYPE16 shapeIntegralHypersingularFirstComp[2];
  REALTYPE16 tempShapeIntegralSingleLayer[3][2];
  REALTYPE16 tempShapeIntegralDoubleLayer[3][2];
  REALTYPE16 tempShapeIntegralAdjDoubleLayer[3][2];
  REALTYPE16 tempShapeIntegralHypersingular[2];
  REALTYPE testValue[3];
  REALTYPE trialValue[3];
  REALTYPE testInv[2][2];
  REALTYPE16 trialInv[2][2];

  REALTYPE16 tempFactorDouble[2];
  REALTYPE16 tempFactorAdj[2];
  REALTYPE16 factor1[2];
  REALTYPE16 factor2[2];
  REALTYPE16 product[2];

  REALTYPE16 trialCurl[3][3];
  REALTYPE3 testCurl[3];

  REALTYPE16 basisProduct[3][3];
  REALTYPE16 normalProduct;

  REALTYPE localCoeffsDirichlet[3][2];
  REALTYPE localCoeffsNeumann[3][2];

  REALTYPE16 trialInvBasis[2][3];
  REALTYPE16 trialElementGradient[3][3];

  REALTYPE wavenumberProduct[2];

  __local REALTYPE localResultDirichlet[WORKGROUP_SIZE][3][3][2];
  __local REALTYPE localResultNeumann[WORKGROUP_SIZE][3][3][2];

#ifdef TRANSMISSION
  REALTYPE16 kernelValueInt[2];
  REALTYPE16 kernelGradientInt[3][2];
  REALTYPE16 shapeIntegralSingleLayerInt[3][3][2];
  REALTYPE16 shapeIntegralDoubleLayerInt[3][3][2];
  REALTYPE16 shapeIntegralAdjDoubleLayerInt[3][3][2];
  REALTYPE16 shapeIntegralHypersingularInt[3][3][2];
  REALTYPE16 shapeIntegralHypersingularFirstCompInt[2];

  REALTYPE16 tempShapeIntegralSingleLayerInt[3][2];
  REALTYPE16 tempShapeIntegralDoubleLayerInt[3][2];
  REALTYPE16 tempShapeIntegralAdjDoubleLayerInt[3][2];
  REALTYPE16 tempShapeIntegralHypersingularInt[2];
#endif

  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j)
      for (k = 0; k < 2; ++k) {
        shapeIntegralSingleLayer[i][j][k] = M_ZERO;
        shapeIntegralDoubleLayer[i][j][k] = M_ZERO;
        shapeIntegralAdjDoubleLayer[i][j][k] = M_ZERO;
        shapeIntegralHypersingular[i][j][k] = M_ZERO;
#ifdef TRANSMISSION
        shapeIntegralSingleLayerInt[i][j][k] = M_ZERO;
        shapeIntegralDoubleLayerInt[i][j][k] = M_ZERO;
        shapeIntegralAdjDoubleLayerInt[i][j][k] = M_ZERO;
        shapeIntegralHypersingularInt[i][j][k] = M_ZERO;
#endif
      }

  shapeIntegralHypersingularFirstComp[0] = M_ZERO;
  shapeIntegralHypersingularFirstComp[1] = M_ZERO;
#ifdef TRANSMISSION
  shapeIntegralHypersingularFirstCompInt[0] = M_ZERO;
  shapeIntegralHypersingularFirstCompInt[1] = M_ZERO;
#endif

  getCorners(testGrid, testIndex, testCorners);
  getCornersVec16(trialGrid, trialIndex, trialCorners);

  getElement(testConnectivity, testIndex, testElement);
  getElementVec16(trialConnectivity, trialIndex, trialElement);

  getJacobian(testCorners, testJac);
  getJacobianVec16(trialCorners, trialJac);

  getNormalAndIntegrationElement(testJac, &testNormal, &testIntElem);
  getNormalAndIntegrationElementVec16(trialJac, trialNormal, &trialIntElem);
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
    testPoint = (REALTYPE2)(quadPoints[2 * testQuadIndex],
                            quadPoints[2 * testQuadIndex + 1]);
    testGlobalPoint = getGlobalPoint(testCorners, &testPoint);
    BASIS(p1_discontinuous, evaluate)(&testPoint, &testValue[0]);

    for (j = 0; j < 3; ++j)
      for (k = 0; k < 2; ++k) {
        tempShapeIntegralSingleLayer[j][k] = M_ZERO;
        tempShapeIntegralDoubleLayer[j][k] = M_ZERO;
        tempShapeIntegralAdjDoubleLayer[j][k] = M_ZERO;
#ifdef TRANSMISSION
        tempShapeIntegralSingleLayerInt[j][k] = M_ZERO;
        tempShapeIntegralDoubleLayerInt[j][k] = M_ZERO;
        tempShapeIntegralAdjDoubleLayerInt[j][k] = M_ZERO;
#endif
      }

    tempShapeIntegralHypersingular[0] = M_ZERO;
    tempShapeIntegralHypersingular[1] = M_ZERO;

#ifdef TRANSMISSION
    tempShapeIntegralHypersingularInt[0] = M_ZERO;
    tempShapeIntegralHypersingularInt[1] = M_ZERO;
#endif

    for (trialQuadIndex = 0; trialQuadIndex < NUMBER_OF_QUAD_POINTS;
         ++trialQuadIndex) {
      trialPoint = (REALTYPE2)(quadPoints[2 * trialQuadIndex],
                               quadPoints[2 * trialQuadIndex + 1]);
      getGlobalPointVec16(trialCorners, &trialPoint, trialGlobalPoint);
      BASIS(p1_discontinuous, evaluate)(&trialPoint, &trialValue[0]);

      diff[0] = trialGlobalPoint[0] - testGlobalPoint.x;
      diff[1] = trialGlobalPoint[1] - testGlobalPoint.y;
      diff[2] = trialGlobalPoint[2] - testGlobalPoint.z;
      dist = sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);
      rdist = M_ONE / dist;

      kernelValue[0] = M_INV_4PI * cos(WAVENUMBER_REAL * dist) * rdist *
                       quadWeights[trialQuadIndex];
      kernelValue[1] = M_INV_4PI * sin(WAVENUMBER_REAL * dist) * rdist *
                       quadWeights[trialQuadIndex];

#ifdef WAVENUMBER_COMPLEX
      kernelValue[0] *= exp(-WAVENUMBER_COMPLEX * dist);
      kernelValue[1] *= exp(-WAVENUMBER_COMPLEX * dist);
#endif

      factor1[0] = kernelValue[0] * rdist * rdist;
      factor1[1] = kernelValue[1] * rdist * rdist;

      factor2[0] = -M_ONE;
      factor2[1] = WAVENUMBER_REAL * dist;

#ifdef WAVENUMBER_COMPLEX
      factor2[0] += -WAVENUMBER_COMPLEX * dist;
#endif

      product[0] = -(factor1[0] * factor2[0] - factor1[1] * factor2[1]);
      product[1] = -(factor1[0] * factor2[1] + factor1[1] * factor2[0]);

      kernelGradient[0][0] = product[0] * diff[0];
      kernelGradient[0][1] = product[1] * diff[0];
      kernelGradient[1][0] = product[0] * diff[1];
      kernelGradient[1][1] = product[1] * diff[1];
      kernelGradient[2][0] = product[0] * diff[2];
      kernelGradient[2][1] = product[1] * diff[2];

      tempFactorDouble[0] = -(kernelGradient[0][0] * trialNormal[0] +
                              kernelGradient[1][0] * trialNormal[1] +
                              kernelGradient[2][0] * trialNormal[2]);
      tempFactorDouble[1] = -(kernelGradient[0][1] * trialNormal[0] +
                              kernelGradient[1][1] * trialNormal[1] +
                              kernelGradient[2][1] * trialNormal[2]);
      tempFactorAdj[0] = (kernelGradient[0][0] * testNormal.x +
                          kernelGradient[1][0] * testNormal.y +
                          kernelGradient[2][0] * testNormal.z);
      tempFactorAdj[1] = (kernelGradient[0][1] * testNormal.x +
                          kernelGradient[1][1] * testNormal.y +
                          kernelGradient[2][1] * testNormal.z);

      for (j = 0; j < 3; ++j) {
        tempShapeIntegralSingleLayer[j][0] += kernelValue[0] * trialValue[j];
        tempShapeIntegralDoubleLayer[j][0] +=
            tempFactorDouble[0] * trialValue[j];
        tempShapeIntegralAdjDoubleLayer[j][0] +=
            tempFactorAdj[0] * trialValue[j];
        tempShapeIntegralSingleLayer[j][1] += kernelValue[1] * trialValue[j];
        tempShapeIntegralDoubleLayer[j][1] +=
            tempFactorDouble[1] * trialValue[j];
        tempShapeIntegralAdjDoubleLayer[j][1] +=
            tempFactorAdj[1] * trialValue[j];
      }

      tempShapeIntegralHypersingular[0] += kernelValue[0];
      tempShapeIntegralHypersingular[1] += kernelValue[1];

#ifdef TRANSMISSION

      kernelValue[0] = M_INV_4PI * cos(WAVENUMBER_INT_REAL * dist) * rdist *
                       quadWeights[trialQuadIndex];
      kernelValue[1] = M_INV_4PI * sin(WAVENUMBER_INT_REAL * dist) * rdist *
                       quadWeights[trialQuadIndex];

#ifdef WAVENUMBER_INT_COMPLEX
      kernelValue[0] *= exp(-WAVENUMBER_INT_COMPLEX * dist);
      kernelValue[1] *= exp(-WAVENUMBER_INT_COMPLEX * dist);
#endif

      factor1[0] = kernelValue[0] * rdist * rdist;
      factor1[1] = kernelValue[1] * rdist * rdist;

      factor2[0] = -M_ONE;
      factor2[1] = WAVENUMBER_INT_REAL * dist;

#ifdef WAVENUMBER_INT_COMPLEX
      factor2[0] += -WAVENUMBER_INT_COMPLEX * dist;
#endif

      product[0] = -(factor1[0] * factor2[0] - factor1[1] * factor2[1]);
      product[1] = -(factor1[0] * factor2[1] + factor1[1] * factor2[0]);

      kernelGradient[0][0] = product[0] * diff[0];
      kernelGradient[0][1] = product[1] * diff[0];
      kernelGradient[1][0] = product[0] * diff[1];
      kernelGradient[1][1] = product[1] * diff[1];
      kernelGradient[2][0] = product[0] * diff[2];
      kernelGradient[2][1] = product[1] * diff[2];

      tempFactorDouble[0] = -(kernelGradient[0][0] * trialNormal[0] +
                              kernelGradient[1][0] * trialNormal[1] +
                              kernelGradient[2][0] * trialNormal[2]);
      tempFactorDouble[1] = -(kernelGradient[0][1] * trialNormal[0] +
                              kernelGradient[1][1] * trialNormal[1] +
                              kernelGradient[2][1] * trialNormal[2]);
      tempFactorAdj[0] = (kernelGradient[0][0] * testNormal.x +
                          kernelGradient[1][0] * testNormal.y +
                          kernelGradient[2][0] * testNormal.z);
      tempFactorAdj[1] = (kernelGradient[0][1] * testNormal.x +
                          kernelGradient[1][1] * testNormal.y +
                          kernelGradient[2][1] * testNormal.z);

      for (j = 0; j < 3; ++j) {
        tempShapeIntegralSingleLayerInt[j][0] += kernelValue[0] * trialValue[j];
        tempShapeIntegralDoubleLayerInt[j][0] +=
            tempFactorDouble[0] * trialValue[j];
        tempShapeIntegralAdjDoubleLayerInt[j][0] +=
            tempFactorAdj[0] * trialValue[j];
        tempShapeIntegralSingleLayerInt[j][1] += kernelValue[1] * trialValue[j];
        tempShapeIntegralDoubleLayerInt[j][1] +=
            tempFactorDouble[1] * trialValue[j];
        tempShapeIntegralAdjDoubleLayerInt[j][1] +=
            tempFactorAdj[1] * trialValue[j];
      }

      tempShapeIntegralHypersingularInt[0] += kernelValue[0];
      tempShapeIntegralHypersingularInt[1] += kernelValue[1];
#endif
    }

    for (i = 0; i < 3; ++i)
      for (j = 0; j < 3; ++j)
        for (k = 0; k < 2; ++k) {
          shapeIntegralSingleLayer[i][j][k] +=
              quadWeights[testQuadIndex] * testValue[i] *
              tempShapeIntegralSingleLayer[j][k];
          shapeIntegralDoubleLayer[i][j][k] +=
              quadWeights[testQuadIndex] * testValue[i] *
              tempShapeIntegralDoubleLayer[j][k];
          shapeIntegralAdjDoubleLayer[i][j][k] +=
              quadWeights[testQuadIndex] * testValue[i] *
              tempShapeIntegralAdjDoubleLayer[j][k];
        }

    shapeIntegralHypersingularFirstComp[0] +=
        quadWeights[testQuadIndex] * tempShapeIntegralHypersingular[0];
    shapeIntegralHypersingularFirstComp[1] +=
        quadWeights[testQuadIndex] * tempShapeIntegralHypersingular[1];

#ifdef TRANSMISSION
    for (i = 0; i < 3; ++i)
      for (j = 0; j < 3; ++j)
        for (k = 0; k < 2; ++k) {
          shapeIntegralSingleLayerInt[i][j][k] +=
              quadWeights[testQuadIndex] * testValue[i] *
              tempShapeIntegralSingleLayerInt[j][k];
          shapeIntegralDoubleLayerInt[i][j][k] +=
              quadWeights[testQuadIndex] * testValue[i] *
              tempShapeIntegralDoubleLayerInt[j][k];
          shapeIntegralAdjDoubleLayerInt[i][j][k] +=
              quadWeights[testQuadIndex] * testValue[i] *
              tempShapeIntegralAdjDoubleLayerInt[j][k];
        }

    shapeIntegralHypersingularFirstCompInt[0] +=
        quadWeights[testQuadIndex] * tempShapeIntegralHypersingularInt[0];
    shapeIntegralHypersingularFirstCompInt[1] +=
        quadWeights[testQuadIndex] * tempShapeIntegralHypersingularInt[1];
#endif
  }

#ifdef WAVENUMBER_COMPLEX
  wavenumberProduct[0] = WAVENUMBER_REAL * WAVENUMBER_REAL -
                         WAVENUMBER_COMPLEX * WAVENUMBER_COMPLEX;
  wavenumberProduct[1] = M_TWO * WAVENUMBER_REAL * WAVENUMBER_COMPLEX;
#else
  wavenumberProduct[0] = WAVENUMBER_REAL * WAVENUMBER_REAL;
  wavenumberProduct[1] = M_ZERO;
#endif

  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j) {
      shapeIntegralHypersingular[i][j][0] =
          shapeIntegralHypersingularFirstComp[0] * basisProduct[i][j] -
          (wavenumberProduct[0] * shapeIntegralSingleLayer[i][j][0] -
           wavenumberProduct[1] * shapeIntegralSingleLayer[i][j][1]) *
              normalProduct;
      shapeIntegralHypersingular[i][j][1] =
          shapeIntegralHypersingularFirstComp[1] * basisProduct[i][j] -
          (wavenumberProduct[0] * shapeIntegralSingleLayer[i][j][1] +
           wavenumberProduct[1] * shapeIntegralSingleLayer[i][j][0]) *
              normalProduct;
    }

  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j)
      for (k = 0; k < 2; ++k) {
        shapeIntegralSingleLayer[i][j][k] *= testIntElem * trialIntElem;
        shapeIntegralDoubleLayer[i][j][k] *= testIntElem * trialIntElem;
        shapeIntegralAdjDoubleLayer[i][j][k] *= testIntElem * trialIntElem;
        shapeIntegralHypersingular[i][j][k] *= testIntElem * trialIntElem;
      }

#ifdef TRANSMISSION

#ifdef WAVENUMBER_INT_COMPLEX
  wavenumberProduct[0] = WAVENUMBER_INT_REAL * WAVENUMBER_INT_REAL -
                         WAVENUMBER_INT_COMPLEX * WAVENUMBER_INT_COMPLEX;
  wavenumberProduct[1] = M_TWO * WAVENUMBER_INT_REAL * WAVENUMBER_INT_COMPLEX;
#else
  wavenumberProduct[0] = WAVENUMBER_INT_REAL * WAVENUMBER_INT_REAL;
  wavenumberProduct[1] = M_ZERO;
#endif

  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j) {
      shapeIntegralHypersingularInt[i][j][0] =
          shapeIntegralHypersingularFirstCompInt[0] * basisProduct[i][j] -
          (wavenumberProduct[0] * shapeIntegralSingleLayerInt[i][j][0] -
           wavenumberProduct[1] * shapeIntegralSingleLayerInt[i][j][1]) *
              normalProduct;
      shapeIntegralHypersingularInt[i][j][1] =
          shapeIntegralHypersingularFirstCompInt[1] * basisProduct[i][j] -
          (wavenumberProduct[0] * shapeIntegralSingleLayerInt[i][j][1] +
           wavenumberProduct[1] * shapeIntegralSingleLayerInt[i][j][0]) *
              normalProduct;
    }

  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j)
      for (k = 0; k < 2; ++k) {
        shapeIntegralSingleLayerInt[i][j][k] *= testIntElem * trialIntElem;
        shapeIntegralDoubleLayerInt[i][j][k] *= testIntElem * trialIntElem;
        shapeIntegralAdjDoubleLayerInt[i][j][k] *= testIntElem * trialIntElem;
        shapeIntegralHypersingularInt[i][j][k] *= testIntElem * trialIntElem;
      }

  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j)
      for (k = 0; k < 2; ++k) {
        shapeIntegralSingleLayer[i][j][k] +=
            RHO_REL_REAL * shapeIntegralSingleLayerInt[i][j][k];
        shapeIntegralDoubleLayer[i][j][k] +=
            shapeIntegralDoubleLayerInt[i][j][k];
        shapeIntegralAdjDoubleLayer[i][j][k] +=
            shapeIntegralAdjDoubleLayerInt[i][j][k];
        shapeIntegralHypersingular[i][j][k] +=
            M_ONE / RHO_REL_REAL * shapeIntegralHypersingularInt[i][j][k];
      }

#endif

  for (vecIndex = 0; vecIndex < 16; ++vecIndex)
    if (!elementsAreAdjacent(testElement, trialElement[vecIndex],
                             gridsAreDisjoint)) {
      for (j = 0; j < 3; ++j) {
        localCoeffsDirichlet[j][0] = input[2 * (3 * trialIndex[vecIndex] + j)];
        localCoeffsDirichlet[j][1] =
            input[2 * (3 * trialIndex[vecIndex] + j) + 1];

        localCoeffsNeumann[j][0] = input[6 * TRIAL0_NUMBER_OF_ELEMENTS +
                                         2 * (3 * trialIndex[vecIndex] + j)];
        localCoeffsNeumann[j][1] =
            input[6 * TRIAL0_NUMBER_OF_ELEMENTS +
                  2 * (3 * trialIndex[vecIndex] + j) + 1];
      }

      for (i = 0; i < 3; ++i)
        for (j = 0; j < 3; ++j) {
          localResultDirichlet[16 * lid + vecIndex][i][j][0] =
              -(VEC_ELEMENT(shapeIntegralDoubleLayer[i][j][0], vecIndex) *
                    localCoeffsDirichlet[j][0] -
                VEC_ELEMENT(shapeIntegralDoubleLayer[i][j][1], vecIndex) *
                    localCoeffsDirichlet[j][1]) +
              (VEC_ELEMENT(shapeIntegralSingleLayer[i][j][0], vecIndex) *
                   localCoeffsNeumann[j][0] -
               VEC_ELEMENT(shapeIntegralSingleLayer[i][j][1], vecIndex) *
                   localCoeffsNeumann[j][1]);
          localResultDirichlet[16 * lid + vecIndex][i][j][1] =
              -(VEC_ELEMENT(shapeIntegralDoubleLayer[i][j][0], vecIndex) *
                    localCoeffsDirichlet[j][1] +
                VEC_ELEMENT(shapeIntegralDoubleLayer[i][j][1], vecIndex) *
                    localCoeffsDirichlet[j][0]) +
              (VEC_ELEMENT(shapeIntegralSingleLayer[i][j][0], vecIndex) *
                   localCoeffsNeumann[j][1] +
               VEC_ELEMENT(shapeIntegralSingleLayer[i][j][1], vecIndex) *
                   localCoeffsNeumann[j][0]);

          localResultNeumann[16 * lid + vecIndex][i][j][0] =
              (VEC_ELEMENT(shapeIntegralHypersingular[i][j][0], vecIndex) *
                   localCoeffsDirichlet[j][0] -
               VEC_ELEMENT(shapeIntegralHypersingular[i][j][1], vecIndex) *
                   localCoeffsDirichlet[j][1]) +
              (VEC_ELEMENT(shapeIntegralAdjDoubleLayer[i][j][0], vecIndex) *
                   localCoeffsNeumann[j][0] -
               VEC_ELEMENT(shapeIntegralAdjDoubleLayer[i][j][1], vecIndex) *
                   localCoeffsNeumann[j][1]);
          localResultNeumann[16 * lid + vecIndex][i][j][1] =
              (VEC_ELEMENT(shapeIntegralHypersingular[i][j][0], vecIndex) *
                   localCoeffsDirichlet[j][1] +
               VEC_ELEMENT(shapeIntegralHypersingular[i][j][1], vecIndex) *
                   localCoeffsDirichlet[j][0]) +
              (VEC_ELEMENT(shapeIntegralAdjDoubleLayer[i][j][0], vecIndex) *
                   localCoeffsNeumann[j][1] +
               VEC_ELEMENT(shapeIntegralAdjDoubleLayer[i][j][1], vecIndex) *
                   localCoeffsNeumann[j][0]);
        }

    } else {
      for (i = 0; i < 3; ++i)
        for (j = 0; j < 3; ++j) {
          localResultDirichlet[16 * lid + vecIndex][i][j][0] = M_ZERO;
          localResultDirichlet[16 * lid + vecIndex][i][j][1] = M_ZERO;

          localResultNeumann[16 * lid + vecIndex][i][j][0] = M_ZERO;
          localResultNeumann[16 * lid + vecIndex][i][j][1] = M_ZERO;
        }
    }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    for (localIndex = 1; localIndex < WORKGROUP_SIZE; ++localIndex)
      for (i = 0; i < 3; ++i)
        for (j = 0; j < 3; ++j) {
          localResultDirichlet[0][i][j][0] +=
              localResultDirichlet[localIndex][i][j][0];
          localResultDirichlet[0][i][j][1] +=
              localResultDirichlet[localIndex][i][j][1];

          localResultNeumann[0][i][j][0] +=
              localResultNeumann[localIndex][i][j][0];
          localResultNeumann[0][i][j][1] +=
              localResultNeumann[localIndex][i][j][1];
        }

    for (i = 0; i < 3; ++i) {
      for (j = 1; j < 3; ++j) {
        localResultDirichlet[0][i][0][0] += localResultDirichlet[0][i][j][0];
        localResultDirichlet[0][i][0][1] += localResultDirichlet[0][i][j][1];

        localResultNeumann[0][i][0][0] += localResultNeumann[0][i][j][0];
        localResultNeumann[0][i][0][1] += localResultNeumann[0][i][j][1];
      }

      globalResult[2 * (numGroups * (3 * gid[0] + i) + groupId)] +=
          localResultDirichlet[0][i][0][0];
      globalResult[2 * (numGroups * (3 * gid[0] + i) + groupId) + 1] +=
          localResultDirichlet[0][i][0][1];

      globalResult[6 * TRIAL0_NUMBER_OF_ELEMENTS * numGroups +
                   2 * (numGroups * (3 * gid[0] + i) + groupId)] +=
          localResultNeumann[0][i][0][0];
      globalResult[6 * TRIAL0_NUMBER_OF_ELEMENTS * numGroups +
                   2 * (numGroups * (3 * gid[0] + i) + groupId) + 1] +=
          localResultNeumann[0][i][0][1];
    }
  }
}
