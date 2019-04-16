#include "bempp_base_types.h"
#include "bempp_helpers.h"
#include "bempp_spaces.h"
#include "kernels.h"

__kernel __attribute__((vec_type_hint(REALTYPE8))) void evaluate_dense_regular(
    __global uint* testIndices, __global uint* trialIndices,
    __global REALTYPE* testGrid, __global REALTYPE* trialGrid,
    __global uint* testConnectivity, __global uint* trialConnectivity,
    __constant REALTYPE* quadPoints,
    __constant REALTYPE* quadWeights,
    __global REALTYPE* input,
    char gridsAreDisjoint,
    __global REALTYPE* globalResult) {
    /* Variable declarations */

    size_t gid[2] = {get_global_id(0), get_global_id(1)};
    size_t offset = get_global_offset(1);

    size_t testIndex = testIndices[gid[0]];
    size_t trialIndex[8] = {trialIndices[offset + 8 * (gid[1] - offset) + 0],
                            trialIndices[offset + 8 * (gid[1] - offset) + 1],
                            trialIndices[offset + 8 * (gid[1] - offset) + 2],
                            trialIndices[offset + 8 * (gid[1] - offset) + 3],
                            trialIndices[offset + 8 * (gid[1] - offset) + 4],
                            trialIndices[offset + 8 * (gid[1] - offset) + 5],
                            trialIndices[offset + 8 * (gid[1] - offset) + 6],
                            trialIndices[offset + 8 * (gid[1] - offset) + 7]
                           };
    size_t testQuadIndex;
    size_t trialQuadIndex;
    size_t i;
    size_t j;
    size_t globalRowIndex;
    size_t globalColIndex;
    size_t vecIndex;
    size_t localIndex;

    size_t lid = get_local_id(1);
    size_t groupId = get_group_id(1);
    size_t numGroups = get_num_groups(1);

    REALTYPE3 testGlobalPoint;
    REALTYPE8 trialGlobalPoint[3];

    REALTYPE3 testCorners[3];
    REALTYPE8 trialCorners[3][3];

    uint testElement[3];
    uint trialElement[8][3];

    REALTYPE3 testJac[2];
    REALTYPE8 trialJac[2][3];

    REALTYPE3 testNormal;
    REALTYPE8 trialNormal[3];

    REALTYPE2 testPoint;
    REALTYPE2 trialPoint;

    REALTYPE testIntElem;
    REALTYPE8 trialIntElem;
    REALTYPE testValue[NUMBER_OF_TEST_SHAPE_FUNCTIONS];
    REALTYPE trialValue[NUMBER_OF_TRIAL_SHAPE_FUNCTIONS];

#ifndef COMPLEX_KERNEL
    REALTYPE8 kernelValue;
    REALTYPE8 tempResult[NUMBER_OF_TRIAL_SHAPE_FUNCTIONS];
    REALTYPE8 tempFactor;
    REALTYPE8 shapeIntegral[NUMBER_OF_TEST_SHAPE_FUNCTIONS][NUMBER_OF_TRIAL_SHAPE_FUNCTIONS];
    REALTYPE localCoeffs[NUMBER_OF_TRIAL_SHAPE_FUNCTIONS];
    __local REALTYPE localResult[WORKGROUP_SIZE][NUMBER_OF_TEST_SHAPE_FUNCTIONS][NUMBER_OF_TRIAL_SHAPE_FUNCTIONS];

#else
    REALTYPE8 kernelValue[2];
    REALTYPE8 tempResult[NUMBER_OF_TRIAL_SHAPE_FUNCTIONS][2];
    REALTYPE8 tempFactor[2];
    REALTYPE8 shapeIntegral[NUMBER_OF_TEST_SHAPE_FUNCTIONS]
    [NUMBER_OF_TRIAL_SHAPE_FUNCTIONS][2];
    REALTYPE localCoeffs[NUMBER_OF_TRIAL_SHAPE_FUNCTIONS][2];
    __local REALTYPE localResult[WORKGROUP_SIZE][NUMBER_OF_TEST_SHAPE_FUNCTIONS][NUMBER_OF_TRIAL_SHAPE_FUNCTIONS][2];

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
    getCornersVec8(trialGrid, trialIndex, trialCorners);

    getElement(testConnectivity, testIndex, testElement);
    getElementVec8(trialConnectivity, trialIndex, trialElement);

    getJacobian(testCorners, testJac);
    getJacobianVec8(trialCorners, trialJac);

    getNormalAndIntegrationElement(testJac, &testNormal, &testIntElem);
    getNormalAndIntegrationElementVec8(trialJac, trialNormal, &trialIntElem);

    for (testQuadIndex = 0; testQuadIndex < NUMBER_OF_QUAD_POINTS;
            ++testQuadIndex) {
        testPoint = (REALTYPE2)(quadPoints[2 * testQuadIndex], quadPoints[2 * testQuadIndex + 1]);
        testGlobalPoint = getGlobalPoint(testCorners, &testPoint);
        BASIS(TEST, evaluate)(&testPoint, &testValue[0]);

        for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
#ifndef COMPLEX_KERNEL
            tempResult[j] = M_ZERO;
#else
            tempResult[j][0] = M_ZERO;
            tempResult[j][1] = M_ZERO;
#endif
        }

        for (trialQuadIndex = 0; trialQuadIndex < NUMBER_OF_QUAD_POINTS;
                ++trialQuadIndex) {
            trialPoint = (REALTYPE2)(quadPoints[2 * trialQuadIndex], quadPoints[2 * trialQuadIndex + 1]);
            getGlobalPointVec8(trialCorners, &trialPoint, trialGlobalPoint);
            BASIS(TRIAL, evaluate)(&trialPoint, &trialValue[0]);
#ifndef COMPLEX_KERNEL
            KERNEL(vec8)
            (testGlobalPoint, trialGlobalPoint, testNormal, trialNormal,
             &kernelValue);
            tempFactor = quadWeights[trialQuadIndex] * kernelValue;
            for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j)
                tempResult[j] += trialValue[j] * tempFactor;
#else
            KERNEL(vec8)
            (testGlobalPoint, trialGlobalPoint, testNormal, trialNormal, kernelValue);
            tempFactor[0] = quadWeights[trialQuadIndex] * kernelValue[0];
            tempFactor[1] = quadWeights[trialQuadIndex] * kernelValue[1];
            for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
                tempResult[j][0] += trialValue[j] * tempFactor[0];
                tempResult[j][1] += trialValue[j] * tempFactor[1];
            }

#endif
        }

        for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
            for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
#ifndef COMPLEX_KERNEL
                shapeIntegral[i][j] +=
                    tempResult[j] * quadWeights[testQuadIndex] * testValue[i];
#else
                shapeIntegral[i][j][0] +=
                    tempResult[j][0] * quadWeights[testQuadIndex] * testValue[i];
                shapeIntegral[i][j][1] +=
                    tempResult[j][1] * quadWeights[testQuadIndex] * testValue[i];
#endif
            }
    }

#ifndef COMPLEX_KERNEL

    for (vecIndex = 0; vecIndex < 8; ++vecIndex)
        if (!elementsAreAdjacent(testElement, trialElement[vecIndex], gridsAreDisjoint)) {
            for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j)
                localCoeffs[j] = input[NUMBER_OF_TRIAL_SHAPE_FUNCTIONS * trialIndex[vecIndex] + j];
            for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
                for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j)
                    localResult[8 * lid + vecIndex][i][j] = ((REALTYPE*)(&shapeIntegral[i][j]))[vecIndex] *
                                                            testIntElem * ((REALTYPE*)(&trialIntElem))[vecIndex] * localCoeffs[j];
        }
        else {
            for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
                for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j)
                    localResult[8 * lid + vecIndex][i][j] = M_ZERO;
        }
#else
    for (vecIndex = 0; vecIndex < 8; ++vecIndex)
        if (!elementsAreAdjacent(testElement, trialElement[vecIndex], gridsAreDisjoint)) {
            for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
                localCoeffs[j][0] = input[2 * (NUMBER_OF_TRIAL_SHAPE_FUNCTIONS * trialIndex[vecIndex] + j)];
                localCoeffs[j][1] = input[2 * (NUMBER_OF_TRIAL_SHAPE_FUNCTIONS * trialIndex[vecIndex] + j) + 1];
            }


            for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
                for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
                    localResult[8 * lid + vecIndex][i][j][0] = testIntElem * ((REALTYPE*)(&trialIntElem))[vecIndex] *
                            (((REALTYPE*)(&shapeIntegral[i][j][0]))[vecIndex] * localCoeffs[j][0] - ((REALTYPE*)(&shapeIntegral[i][j][1]))[vecIndex] * localCoeffs[j][1]);
                    localResult[8 * lid + vecIndex][i][j][1] = testIntElem * ((REALTYPE*)(&trialIntElem))[vecIndex] *
                            (((REALTYPE*)(&shapeIntegral[i][j][0]))[vecIndex] * localCoeffs[j][1] + ((REALTYPE*)(&shapeIntegral[i][j][1]))[vecIndex] * localCoeffs[j][0]);
                }

        }
        else {
            for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
                for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
                    localResult[8 * lid + vecIndex][i][j][0] = M_ZERO;
                    localResult[8 * lid + vecIndex][i][j][1] = M_ZERO;
                }


        }

#endif

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        for (localIndex = 1; localIndex < WORKGROUP_SIZE; ++ localIndex)
            for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
                for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
#ifndef COMPLEX_KERNEL
                    localResult[0][i][j] += localResult[localIndex][i][j];
#else
                    localResult[0][i][j][0] += localResult[localIndex][i][j][0];
                    localResult[0][i][j][1] += localResult[localIndex][i][j][1];
#endif

                }

        for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
#ifndef COMPLEX_KERNEL
        {
            for (j = 1; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j)
                localResult[0][i][0] += localResult[0][i][j];
            globalResult[numGroups * (NUMBER_OF_TEST_SHAPE_FUNCTIONS * testIndex + i) + groupId] += localResult[0][i][0];
        }
#else
        {
            for (j = 1; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
                localResult[0][i][0][0] += localResult[0][i][j][0];
                localResult[0][i][0][1] += localResult[0][i][j][1];
            }
            globalResult[2 * (numGroups * (NUMBER_OF_TEST_SHAPE_FUNCTIONS * testIndex + i) + groupId)] += localResult[0][i][0][0];
            globalResult[2 * (numGroups * (NUMBER_OF_TEST_SHAPE_FUNCTIONS * testIndex + i) + groupId) + 1] += localResult[0][i][0][1];
        }

#endif
    }

}
