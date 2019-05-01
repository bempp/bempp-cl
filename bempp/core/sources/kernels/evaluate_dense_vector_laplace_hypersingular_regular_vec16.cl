#include "bempp_base_types.h"
#include "bempp_helpers.h"
#include "bempp_spaces.h"
#include "kernels.h"

__kernel __attribute__((vec_type_hint(REALTYPE16))) __kernel void
evaluate_dense_laplace_hypersingular_regular(
    __global uint* testIndices, __global uint* trialIndices,
    __global REALTYPE* testGrid, __global REALTYPE* trialGrid,
    __global uint* testConnectivity, __global uint* trialConnectivity,
    __constant REALTYPE* quadPoints,
    __constant REALTYPE* quadWeights, __global REALTYPE* input,
    char gridsAreDisjoint, __global REALTYPE* globalResult) {
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
    size_t localIndex;
    size_t vecIndex;

    size_t lid = get_local_id(1);
    size_t groupId = get_group_id(1);
    size_t numGroups = get_num_groups(1);


    REALTYPE3 testGlobalPoint;
    REALTYPE16 trialGlobalPoint[3];

    REALTYPE3 testCorners[3];
    REALTYPE16 trialCorners[3][3];

    uint testElement[3];
    uint trialElement[16][3];

    REALTYPE3 testJac[2];
    REALTYPE16 trialJac[2][3];

    REALTYPE3 testNormal;
    REALTYPE16 trialNormal[3];

    REALTYPE2 testPoint;
    REALTYPE2 trialPoint;

    REALTYPE testIntElem;
    REALTYPE16 trialIntElem;

    REALTYPE testInv[2][2];
    REALTYPE16 trialInv[2][2];

    REALTYPE16 trialCurl[3][3];
    REALTYPE3 testCurl[3];

    REALTYPE16 basisProduct[3][3];

    REALTYPE16 trialInvBasis[2][3];
    REALTYPE16 trialElementGradient[3][3];

    REALTYPE16 kernelValue;
    REALTYPE16 tempResult;
    REALTYPE16 shapeIntegral;

    REALTYPE localCoeffs[NUMBER_OF_TRIAL_SHAPE_FUNCTIONS];
    __local REALTYPE localResult[WORKGROUP_SIZE][NUMBER_OF_TEST_SHAPE_FUNCTIONS][NUMBER_OF_TRIAL_SHAPE_FUNCTIONS];


    getCorners(testGrid, testIndex, testCorners);
    getCornersVec16(trialGrid, trialIndex, trialCorners);

    getElement(testConnectivity, testIndex, testElement);
    getElementVec16(trialConnectivity, trialIndex, trialElement);

    getJacobian(testCorners, testJac);
    getJacobianVec16(trialCorners, trialJac);

    getNormalAndIntegrationElement(testJac, &testNormal, &testIntElem);
    getNormalAndIntegrationElementVec16(trialJac, trialNormal, &trialIntElem);

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
              testJac[1] * (-testInv[1][0] - testInv[1][1]));
    testCurl[1] = cross(testNormal,
                        testJac[0] * testInv[0][0] + testJac[1] * testInv[1][0]);
    testCurl[2] = cross(testNormal,
                        testJac[0] * testInv[0][1] + testJac[1] * testInv[1][1]);

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
        trialCurl[i][0] = trialNormal[1] * trialElementGradient[i][2] -
                          trialNormal[2] * trialElementGradient[i][1];
        trialCurl[i][1] = trialNormal[2] * trialElementGradient[i][0] -
                          trialNormal[0] * trialElementGradient[i][2];
        trialCurl[i][2] = trialNormal[0] * trialElementGradient[i][1] -
                          trialNormal[1] * trialElementGradient[i][0];
    }

    for (i = 0; i < 3; ++i)
        for (j = 0; j < 3; ++j) {
            basisProduct[i][j] = testCurl[i].x * trialCurl[j][0] +
                                 testCurl[i].y * trialCurl[j][1] +
                                 testCurl[i].z * trialCurl[j][2];
        }

    shapeIntegral = M_ZERO;

    for (testQuadIndex = 0; testQuadIndex < NUMBER_OF_QUAD_POINTS;
            ++testQuadIndex) {
        testPoint = (REALTYPE2)(quadPoints[2 * testQuadIndex], quadPoints[2 * testQuadIndex + 1]);
        testGlobalPoint = getGlobalPoint(testCorners, &testPoint);
        tempResult = M_ZERO;

        for (trialQuadIndex = 0; trialQuadIndex < NUMBER_OF_QUAD_POINTS;
                ++trialQuadIndex) {
            trialPoint = (REALTYPE2)(quadPoints[2 * trialQuadIndex], quadPoints[2 * trialQuadIndex + 1]);
            getGlobalPointVec16(trialCorners, &trialPoint, trialGlobalPoint);
            KERNEL(vec16)
            (testGlobalPoint, trialGlobalPoint, testNormal, trialNormal,
             &kernelValue);
            tempResult += quadWeights[trialQuadIndex] * kernelValue;
        }

        shapeIntegral += tempResult * quadWeights[testQuadIndex];
    }

    // the Jacobian Inverse must by divded by the squared of the integration
    // elements. The integral must be multiplied by the integration elements. So
    // in total we have to divide once.

    shapeIntegral /= (testIntElem * trialIntElem);

    for (vecIndex = 0; vecIndex < 16; ++vecIndex)
        if (!elementsAreAdjacent(testElement, trialElement[vecIndex], gridsAreDisjoint)) {
            for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j)
                localCoeffs[j] = input[NUMBER_OF_TRIAL_SHAPE_FUNCTIONS * trialIndex[vecIndex] + j];
            for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
                for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j)
                    localResult[16 * lid + vecIndex][i][j] = ((REALTYPE*)(&shapeIntegral))[vecIndex] *
                                                            ((REALTYPE*)(&basisProduct[i][j]))[vecIndex] *
                                                            localCoeffs[j];
        }
        else {
            for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
                for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j)
                    localResult[16 * lid + vecIndex][i][j] = M_ZERO;
        }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) {
        for (localIndex = 1; localIndex < WORKGROUP_SIZE; ++localIndex)
            for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
                for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j)
                    localResult[0][i][j] += localResult[localIndex][i][j];

        for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
        {
            for (j = 1; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j)
                localResult[0][i][0] += localResult[0][i][j];
            globalResult[numGroups * (NUMBER_OF_TEST_SHAPE_FUNCTIONS * gid[0] + i) + groupId] += localResult[0][i][0];
        }
    }

}
