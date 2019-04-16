#include "bempp_base_types.h"
#include "bempp_helpers.h"
#include "bempp_spaces.h"
#include "kernels.h"

__kernel void evaluate_dense_vector_laplace_hypersingular_regular(
    __global uint* testIndices, __global uint* trialIndices,
    __global REALTYPE* testGrid, __global REALTYPE* trialGrid,
    __global uint* testConnectivity, __global uint* trialConnectivity,
    __constant REALTYPE* quadPoints,
    __constant REALTYPE* quadWeights,
    __global REALTYPE* input, char gridsAreDisjoint,
    __global REALTYPE* globalResult) {
    /* Variable declarations */

    size_t gid[2] = {get_global_id(0), get_global_id(1)};

    size_t testIndex = testIndices[gid[0]];
    size_t trialIndex = trialIndices[gid[1]];
    size_t localIndex;

    size_t testQuadIndex;
    size_t trialQuadIndex;
    size_t i;
    size_t j;
    size_t globalRowIndex;
    size_t globalColIndex;

    size_t lid = get_local_id(1);
    size_t groupId = get_group_id(1);
    size_t numGroups = get_num_groups(1);


    REALTYPE3 testGlobalPoint;
    REALTYPE3 trialGlobalPoint;

    REALTYPE3 testCorners[3];
    REALTYPE3 trialCorners[3];

    uint testElement[3];
    uint trialElement[3];

    REALTYPE3 testJac[2];
    REALTYPE3 trialJac[2];

    REALTYPE3 testNormal;
    REALTYPE3 trialNormal;

    REALTYPE2 testPoint;
    REALTYPE2 trialPoint;

    REALTYPE testIntElem;
    REALTYPE trialIntElem;

    REALTYPE testInv[2][2];
    REALTYPE trialInv[2][2];

    REALTYPE3 trialCurl[3];
    REALTYPE3 testCurl[3];

    REALTYPE basisProduct[3][3];

    REALTYPE kernelValue;
    REALTYPE tempResult;
    REALTYPE shapeIntegral;

    REALTYPE localCoeffs[3];
    __local REALTYPE localResult[WORKGROUP_SIZE][3][3];


    getCorners(testGrid, testIndex, testCorners);
    getCorners(trialGrid, trialIndex, trialCorners);

    getElement(testConnectivity, testIndex, testElement);
    getElement(trialConnectivity, trialIndex, trialElement);

    getJacobian(testCorners, testJac);
    getJacobian(trialCorners, trialJac);

    getNormalAndIntegrationElement(testJac, &testNormal, &testIntElem);
    getNormalAndIntegrationElement(trialJac, &trialNormal, &trialIntElem);

    testInv[0][0] = dot(testJac[1], testJac[1]);
    testInv[1][1] = dot(testJac[0], testJac[0]);
    testInv[0][1] = -dot(testJac[0], testJac[1]);
    testInv[1][0] = testInv[0][1];

    trialInv[0][0] = dot(trialJac[1], trialJac[1]);
    trialInv[1][1] = dot(trialJac[0], trialJac[0]);
    trialInv[0][1] = -dot(trialJac[0], trialJac[1]);
    trialInv[1][0] = trialInv[0][1];

    testCurl[0] =
        cross(testNormal, testJac[0] * (-testInv[0][0] - testInv[0][1]) +
              testJac[1] * (-testInv[1][0] - testInv[1][1]));
    testCurl[1] = cross(testNormal,
                        testJac[0] * testInv[0][0] + testJac[1] * testInv[1][0]);
    testCurl[2] = cross(testNormal,
                        testJac[0] * testInv[0][1] + testJac[1] * testInv[1][1]);

    trialCurl[0] =
        cross(trialNormal, trialJac[0] * (-trialInv[0][0] - trialInv[0][1]) +
              trialJac[1] * (-trialInv[1][0] - trialInv[1][1]));
    trialCurl[1] = cross(
                       trialNormal, trialJac[0] * trialInv[0][0] + trialJac[1] * trialInv[1][0]);
    trialCurl[2] = cross(
                       trialNormal, trialJac[0] * trialInv[0][1] + trialJac[1] * trialInv[1][1]);

    for (i = 0; i < 3; ++i)
        for (j = 0; j < 3; ++j) basisProduct[i][j] = dot(testCurl[i], trialCurl[j]);

    shapeIntegral = M_ZERO;

    for (testQuadIndex = 0; testQuadIndex < NUMBER_OF_QUAD_POINTS;
            ++testQuadIndex) {
        testPoint = (REALTYPE2)(quadPoints[2 * testQuadIndex], quadPoints[2 * testQuadIndex + 1]);
        testGlobalPoint = getGlobalPoint(testCorners, &testPoint);
        tempResult = M_ZERO;

        for (trialQuadIndex = 0; trialQuadIndex < NUMBER_OF_QUAD_POINTS;
                ++trialQuadIndex) {
            trialPoint = (REALTYPE2)(quadPoints[2 * trialQuadIndex], quadPoints[2 * trialQuadIndex + 1]);
            trialGlobalPoint = getGlobalPoint(trialCorners, &trialPoint);
            KERNEL(novec)
            (testGlobalPoint, trialGlobalPoint, testNormal, trialNormal,
             &kernelValue);
            tempResult += quadWeights[trialQuadIndex] * kernelValue;
        }

        shapeIntegral += tempResult * quadWeights[testQuadIndex];
    }

    // the Jacobian Inverse must be divded by the squared of the integration
    // elements. The integral must be multiplied by the integration elements. So
    // in total we have to divide once.

    shapeIntegral /= (testIntElem * trialIntElem);

    if (!elementsAreAdjacent(testElement, trialElement, gridsAreDisjoint)) {
        for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j)
            localCoeffs[j] = input[NUMBER_OF_TRIAL_SHAPE_FUNCTIONS * trialIndex + j];
        for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
            for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j)
                localResult[lid][i][j] = shapeIntegral * basisProduct[i][j] * localCoeffs[j];
    }
    else {
        for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
            for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j)
                localResult[lid][i][j] = M_ZERO;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        for (localIndex = 1; localIndex < WORKGROUP_SIZE; ++ localIndex)
            for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
                for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j)
                    localResult[0][i][j] += localResult[localIndex][i][j];
        for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
        {
            for (j = 1; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j)
                localResult[0][i][0] += localResult[0][i][j];
            globalResult[numGroups * (NUMBER_OF_TEST_SHAPE_FUNCTIONS * testIndex + i) + groupId] += localResult[0][i][0];
        }
    }

}
