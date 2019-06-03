#include "bempp_base_types.h"
#include "bempp_helpers.h"
#include "bempp_spaces.h"
#include "kernels.h"

__kernel __attribute__((vec_type_hint(REALTYPE16))) void
evaluate_dense_magnetic_field_regular(
    __global uint *testIndices, __global uint *trialIndices,
    __global int *testNormalSigns, __global int *trialNormalSigns,
    __global REALTYPE *testGrid, __global REALTYPE *trialGrid,
    __global uint *testConnectivity, __global uint *trialConnectivity,
    __constant REALTYPE* quadPoints,
    __constant REALTYPE *quadWeights,
    __global REALTYPE* input,
    char gridsAreDisjoint,
    __global REALTYPE *globalResult) {
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
                             trialIndices[offset + 16 * (gid[1] - offset) + 15]
                            };

    size_t testQuadIndex;
    size_t trialQuadIndex;
    size_t i;
    size_t j;
    size_t k;
    size_t globalRowIndex;
    size_t globalColIndex;
    size_t vecIndex;
    size_t localIndex;

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

    REALTYPE localCoeffs[NUMBER_OF_TRIAL_SHAPE_FUNCTIONS][2];
    __local REALTYPE localResult[WORKGROUP_SIZE][NUMBER_OF_TEST_SHAPE_FUNCTIONS][NUMBER_OF_TRIAL_SHAPE_FUNCTIONS][2];


    for (i = 0; i < 3; ++i)
        for (j = 0; j < 3; ++j) {
            shapeIntegral[i][j][0] = M_ZERO;
            shapeIntegral[i][j][1] = M_ZERO;
        }

    getCorners(testGrid, testIndex, testCorners);
    getCornersVec16(trialGrid, trialIndex, trialCorners);

    getElement(testConnectivity, testIndex, testElement);
    getElementVec16(trialConnectivity, trialIndex, trialElement);

    getJacobian(testCorners, testJac);
    getJacobianVec16(trialCorners, trialJac);

    getNormalAndIntegrationElement(testJac, &testNormal, &testIntElem);
    getNormalAndIntegrationElementVec16(trialJac, trialNormal, &trialIntElem);

    computeEdgeLength(testCorners, testEdgeLength);
    computeEdgeLengthVec16(trialCorners, trialEdgeLength);

    updateNormals(testIndex, testNormalSigns, &testNormal);
    updateNormalsVec16(trialIndex, trialNormalSigns, trialNormal);

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
                shapeIntegral[i][j][k] *= testEdgeLength[i] * trialEdgeLength[j];

    for (vecIndex = 0; vecIndex < 16; ++vecIndex)
        if (!elementsAreAdjacent(testElement, trialElement[vecIndex], gridsAreDisjoint)) {
            for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
                localCoeffs[j][0] = input[2 * (NUMBER_OF_TRIAL_SHAPE_FUNCTIONS * trialIndex[vecIndex] + j)];
                localCoeffs[j][1] = input[2 * (NUMBER_OF_TRIAL_SHAPE_FUNCTIONS * trialIndex[vecIndex] + j) + 1];
            }


            for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
                for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
                    localResult[16 * lid + vecIndex][i][j][0] = testIntElem * ((REALTYPE*)(&trialIntElem))[vecIndex] *
                            (((REALTYPE*)(&shapeIntegral[i][j][0]))[vecIndex] * localCoeffs[j][0] - ((REALTYPE*)(&shapeIntegral[i][j][1]))[vecIndex] * localCoeffs[j][1]);
                    localResult[16 * lid + vecIndex][i][j][1] = testIntElem * ((REALTYPE*)(&trialIntElem))[vecIndex] *
                            (((REALTYPE*)(&shapeIntegral[i][j][0]))[vecIndex] * localCoeffs[j][1] + ((REALTYPE*)(&shapeIntegral[i][j][1]))[vecIndex] * localCoeffs[j][0]);
                }

        }
        else {
            for (i = 0; i < NUMBER_OF_TEST_SHAPE_FUNCTIONS; ++i)
                for (j = 0; j < NUMBER_OF_TRIAL_SHAPE_FUNCTIONS; ++j) {
                    localResult[16 * lid + vecIndex][i][j][0] = M_ZERO;
                    localResult[16 * lid + vecIndex][i][j][1] = M_ZERO;
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
            globalResult[2 * (numGroups * (NUMBER_OF_TEST_SHAPE_FUNCTIONS * gid[0] + i) + groupId)] += localResult[0][i][0][0];
            globalResult[2 * (numGroups * (NUMBER_OF_TEST_SHAPE_FUNCTIONS * gid[0] + i) + groupId) + 1] += localResult[0][i][0][1];
        }
    }



}
