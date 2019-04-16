#include "bempp_base_types.h"
#include "bempp_helpers.h"
#include "bempp_spaces.h"
#include "kernels.h"

__kernel void evaluate_dense_magnetic_field_regular(
    __global uint *testIndices, __global uint *trialIndices,
    __global REALTYPE *testGrid, __global REALTYPE *trialGrid,
    __global uint *testConnectivity, __global uint *trialConnectivity,
    __constant REALTYPE* quadPoints,
    __constant REALTYPE *quadWeights,
    __global REALTYPE* input,
    char gridsAreDisjoint,
    __global REALTYPE *globalResult) {
    /* Variable declarations */

    size_t gid[2] = {get_global_id(0), get_global_id(1)};

    size_t lid = get_local_id(1);
    size_t groupId = get_group_id(1);
    size_t numGroups = get_num_groups(1);


    size_t testIndex = testIndices[gid[0]];
    size_t trialIndex = trialIndices[gid[1]];

    size_t testQuadIndex;
    size_t trialQuadIndex;
    size_t i;
    size_t j;
    size_t k;
    size_t globalRowIndex;
    size_t globalColIndex;
    size_t localIndex;

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
    REALTYPE testValue[3][2];
    REALTYPE trialValue[3][2];
    REALTYPE3 testElementValue[3];
    REALTYPE3 trialElementValue[3];
    REALTYPE testEdgeLength[3];
    REALTYPE trialEdgeLength[3];

    REALTYPE kernelValue[3][2];

    REALTYPE shapeIntegral[3][3][2];

    REALTYPE tempFactor[3][2];
    REALTYPE tempResult[3][3][2];

    REALTYPE localCoeffs[NUMBER_OF_TRIAL_SHAPE_FUNCTIONS][2];

    __local REALTYPE localResult[WORKGROUP_SIZE][NUMBER_OF_TEST_SHAPE_FUNCTIONS][NUMBER_OF_TRIAL_SHAPE_FUNCTIONS][2];


    for (i = 0; i < 3; ++i)
        for (j = 0; j < 3; ++j) {
            shapeIntegral[i][j][0] = M_ZERO;
            shapeIntegral[i][j][1] = M_ZERO;
        }

    getCorners(testGrid, testIndex, testCorners);
    getCorners(trialGrid, trialIndex, trialCorners);

    getElement(testConnectivity, testIndex, testElement);
    getElement(trialConnectivity, trialIndex, trialElement);

    getJacobian(testCorners, testJac);
    getJacobian(trialCorners, trialJac);

    getNormalAndIntegrationElement(testJac, &testNormal, &testIntElem);
    getNormalAndIntegrationElement(trialJac, &trialNormal, &trialIntElem);

    computeEdgeLength(testCorners, testEdgeLength);
    computeEdgeLength(trialCorners, trialEdgeLength);

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
            trialGlobalPoint = getGlobalPoint(trialCorners, &trialPoint);
            BASIS(TRIAL, evaluate)
            (&trialPoint, &trialValue[0][0]);
            getPiolaTransform(trialIntElem, trialJac, trialValue, trialElementValue);

            KERNEL(novec)
            (testGlobalPoint, trialGlobalPoint, testNormal, trialNormal, kernelValue);

            for (j = 0; j < 3; ++j)
                for (k = 0; k < 2; ++k)
                    tempFactor[j][k] = kernelValue[j][k] * quadWeights[trialQuadIndex];

            for (j = 0; j < 3; ++j)
                for (k = 0; k < 2; ++k) {
                    tempResult[j][0][k] += tempFactor[1][k] * trialElementValue[j].z -
                                           tempFactor[2][k] * trialElementValue[j].y;
                    tempResult[j][1][k] += tempFactor[2][k] * trialElementValue[j].x -
                                           tempFactor[0][k] * trialElementValue[j].z;
                    tempResult[j][2][k] += tempFactor[0][k] * trialElementValue[j].y -
                                           tempFactor[1][k] * trialElementValue[j].x;
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
                shapeIntegral[i][j][k] *= testEdgeLength[i] * trialEdgeLength[j];


    if (!elementsAreAdjacent(testElement, trialElement, gridsAreDisjoint)) {
        for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
            localCoeffs[j][0] = input[2 * (NUMBER_OF_TRIAL_SHAPE_FUNCTIONS * trialIndex + j)];
            localCoeffs[j][1] = input[2 * (NUMBER_OF_TRIAL_SHAPE_FUNCTIONS * trialIndex + j) + 1];
        }

        for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
            for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
                localResult[lid][i][j][0] = testIntElem * trialIntElem * (shapeIntegral[i][j][0] * localCoeffs[j][0] - shapeIntegral[i][j][1] * localCoeffs[j][1]);
                localResult[lid][i][j][1] = testIntElem * trialIntElem * (shapeIntegral[i][j][0] * localCoeffs[j][1] + shapeIntegral[i][j][1] * localCoeffs[j][0]);
            }

    }
    else {
        for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
            for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
                localResult[lid][i][j][0] = M_ZERO;
                localResult[lid][i][j][1] = M_ZERO;
            }


    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        for (localIndex = 1; localIndex < WORKGROUP_SIZE; ++ localIndex)
            for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
                for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
                    localResult[0][i][j][0] += localResult[localIndex][i][j][0];
                    localResult[0][i][j][1] += localResult[localIndex][i][j][1];

                }

        for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
        {
            for (j = 1; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
                localResult[0][i][0][0] += localResult[0][i][j][0];
                localResult[0][i][0][1] += localResult[0][i][j][1];
            }
            globalResult[2 * (numGroups * (NUMBER_OF_TEST_SHAPE_FUNCTIONS * testIndex + i) + groupId)] += localResult[0][i][0][0];
            globalResult[2 * (numGroups * (NUMBER_OF_TEST_SHAPE_FUNCTIONS * testIndex + i) + groupId) + 1] += localResult[0][i][0][1];
        }
    }


}
