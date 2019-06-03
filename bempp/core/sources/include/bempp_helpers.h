#ifndef bempp_helpers_h
#define bempp_helpers_h

#include "bempp_base_types.h"

inline void getCorners(__global REALTYPE *grid, size_t elementIndex, REALTYPE3 *corners)
{

    for (int i = 0; i < 3; ++i)
        corners[i] = (REALTYPE3)(grid[9 * elementIndex + 3 * i], 
                                 grid[9 * elementIndex + 3 * i + 1],
                                 grid[9 * elementIndex + 3 * i + 2]);

}

inline void getCornersVec4(__global REALTYPE *grid, size_t *elementIndex, REALTYPE4 corners[3][3])
{
    /* corners[i][j] is the jth element of the ith corner in each of the 4 elements
       in the elementIndex array */

    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            corners[i][j] = (REALTYPE4)(grid[9 * elementIndex[0] + 3 * i + j],
                                        grid[9 * elementIndex[1] + 3 * i + j],
                                        grid[9 * elementIndex[2] + 3 * i + j],
                                        grid[9 * elementIndex[3] + 3 * i + j]);
}

inline void getCornersVec8(__global REALTYPE *grid, size_t *elementIndex, REALTYPE8 corners[3][3])
{
    /* corners[i][j] is the jth element of the ith corner in each of the 4 elements
       in the elementIndex array */

    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            corners[i][j] = (REALTYPE8)(grid[9 * elementIndex[0] + 3 * i + j],
                                        grid[9 * elementIndex[1] + 3 * i + j],
                                        grid[9 * elementIndex[2] + 3 * i + j],
                                        grid[9 * elementIndex[3] + 3 * i + j],
                                        grid[9 * elementIndex[4] + 3 * i + j],
                                        grid[9 * elementIndex[5] + 3 * i + j],
                                        grid[9 * elementIndex[6] + 3 * i + j],
                                        grid[9 * elementIndex[7] + 3 * i + j]
                                        );
}

inline void getCornersVec16(__global REALTYPE *grid, size_t *elementIndex, REALTYPE16 corners[3][3])
{
    /* corners[i][j] is the jth element of the ith corner in each of the 4 elements
       in the elementIndex array */

    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            corners[i][j] = (REALTYPE16)(grid[9 * elementIndex[0] + 3 * i + j],
                                        grid[9 * elementIndex[1] + 3 * i + j],
                                        grid[9 * elementIndex[2] + 3 * i + j],
                                        grid[9 * elementIndex[3] + 3 * i + j],
                                        grid[9 * elementIndex[4] + 3 * i + j],
                                        grid[9 * elementIndex[5] + 3 * i + j],
                                        grid[9 * elementIndex[6] + 3 * i + j],
                                        grid[9 * elementIndex[7] + 3 * i + j],
                                        grid[9 * elementIndex[8] + 3 * i + j],
                                        grid[9 * elementIndex[9] + 3 * i + j],
                                        grid[9 * elementIndex[10] + 3 * i + j],
                                        grid[9 * elementIndex[11] + 3 * i + j],
                                        grid[9 * elementIndex[12] + 3 * i + j],
                                        grid[9 * elementIndex[13] + 3 * i + j],
                                        grid[9 * elementIndex[14] + 3 * i + j],
                                        grid[9 * elementIndex[15] + 3 * i + j]
                                        );
}


inline void getElement(__global uint* connectivity, size_t elementIndex, uint* element)
{
    for (int i = 0; i < 3; ++i)
        element[i] = connectivity[3 * elementIndex + i];

}

inline void getElementVec4(__global uint* connectivity, size_t *elementIndex, uint element[4][3])
{    
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 3; ++j)
            element[i][j] = connectivity[3 * elementIndex[i] + j];

}

inline void getElementVec8(__global uint* connectivity, size_t *elementIndex, uint element[8][3])
{    
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 3; ++j)
            element[i][j] = connectivity[3 * elementIndex[i] + j];

}

inline void getElementVec16(__global uint* connectivity, size_t *elementIndex, uint element[16][3])
{    
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 3; ++j)
            element[i][j] = connectivity[3 * elementIndex[i] + j];

}


inline void getLocal2Global(__global uint* local2Global, size_t elementIndex, uint* myLocal2Global, int count)
{
    for (int i = 0; i < count; ++i)
        myLocal2Global[i] = local2Global[count * elementIndex + i];

}

inline void getLocal2GlobalVec4(__global uint* local2Global, size_t *elementIndex, uint* myLocal2Global, int count)
{
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < count; ++j)
        myLocal2Global[count * i + j] = local2Global[count * elementIndex[i] + j];

}

inline void getLocal2GlobalVec8(__global uint* local2Global, size_t *elementIndex, uint* myLocal2Global, int count)
{
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < count; ++j)
        myLocal2Global[count * i + j] = local2Global[count * elementIndex[i] + j];

}

inline void getLocal2GlobalVec16(__global uint* local2Global, size_t *elementIndex, uint* myLocal2Global, int count)
{
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < count; ++j)
        myLocal2Global[count * i + j] = local2Global[count * elementIndex[i] + j];

}


inline void getLocalMultipliers(__global REALTYPE* localMultipliers, size_t elementIndex, REALTYPE* myMultipliers, int count)
{
    for (int i = 0; i < count; ++i)
        myMultipliers[i] = localMultipliers[count * elementIndex + i];

}

inline void getLocalMultipliersVec4(__global REALTYPE* localMultipliers, size_t *elementIndex, REALTYPE* myMultipliers, int count)
{
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < count; ++j)
            myMultipliers[count * i + j] = localMultipliers[count * elementIndex[i] + j];

}

inline void getLocalMultipliersVec8(__global REALTYPE* localMultipliers, size_t *elementIndex, REALTYPE* myMultipliers, int count)
{
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < count; ++j)
            myMultipliers[count * i + j] = localMultipliers[count * elementIndex[i] + j];

}

inline void getLocalMultipliersVec16(__global REALTYPE* localMultipliers, size_t *elementIndex, REALTYPE* myMultipliers, int count)
{
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < count; ++j)
            myMultipliers[count * i + j] = localMultipliers[count * elementIndex[i] + j];

}

inline void getJacobian(REALTYPE3 *corners, REALTYPE3 *jacobian)
{
    jacobian[0] = corners[1] - corners[0];
    jacobian[1] = corners[2] - corners[0];
}

inline void getJacobianVec4(REALTYPE4 corners[3][3], REALTYPE4 jacobian[2][3])
{
    jacobian[0][0] = corners[1][0] - corners[0][0];
    jacobian[0][1] = corners[1][1] - corners[0][1];
    jacobian[0][2] = corners[1][2] - corners[0][2];

    jacobian[1][0] = corners[2][0] - corners[0][0];
    jacobian[1][1] = corners[2][1] - corners[0][1];
    jacobian[1][2] = corners[2][2] - corners[0][2];

}

inline void getJacobianVec8(REALTYPE8 corners[3][3], REALTYPE8 jacobian[2][3])
{
    jacobian[0][0] = corners[1][0] - corners[0][0];
    jacobian[0][1] = corners[1][1] - corners[0][1];
    jacobian[0][2] = corners[1][2] - corners[0][2];

    jacobian[1][0] = corners[2][0] - corners[0][0];
    jacobian[1][1] = corners[2][1] - corners[0][1];
    jacobian[1][2] = corners[2][2] - corners[0][2];

}

inline void getJacobianVec16(REALTYPE16 corners[3][3], REALTYPE16 jacobian[2][3])
{
    jacobian[0][0] = corners[1][0] - corners[0][0];
    jacobian[0][1] = corners[1][1] - corners[0][1];
    jacobian[0][2] = corners[1][2] - corners[0][2];

    jacobian[1][0] = corners[2][0] - corners[0][0];
    jacobian[1][1] = corners[2][1] - corners[0][1];
    jacobian[1][2] = corners[2][2] - corners[0][2];

}

inline void getNormalAndIntegrationElement(REALTYPE3 *jacobian, REALTYPE3 *normal, REALTYPE* integrationElement)
{
    *normal = cross(jacobian[0], jacobian[1]);
    *integrationElement = length(*normal);
    *normal /= *integrationElement;
}

inline void getNormalAndIntegrationElementVec4(REALTYPE4 jacobian[2][3], REALTYPE4 normal[3], REALTYPE4* integrationElement)
{

    normal[0] = jacobian[0][1] * jacobian[1][2] - jacobian[0][2] * jacobian[1][1];
    normal[1] = jacobian[0][2] * jacobian[1][0] - jacobian[0][0] * jacobian[1][2];
    normal[2] = jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0];

    *integrationElement = sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    normal[0] /= *integrationElement;
    normal[1] /= *integrationElement;
    normal[2] /= *integrationElement;
    
}

inline void getNormalAndIntegrationElementVec8(REALTYPE8 jacobian[2][3], REALTYPE8 normal[3], REALTYPE8* integrationElement)
{

    normal[0] = jacobian[0][1] * jacobian[1][2] - jacobian[0][2] * jacobian[1][1];
    normal[1] = jacobian[0][2] * jacobian[1][0] - jacobian[0][0] * jacobian[1][2];
    normal[2] = jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0];

    *integrationElement = sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    normal[0] /= *integrationElement;
    normal[1] /= *integrationElement;
    normal[2] /= *integrationElement;
    
}

inline void getNormalAndIntegrationElementVec16(REALTYPE16 jacobian[2][3], REALTYPE16 normal[3], REALTYPE16* integrationElement)
{

    normal[0] = jacobian[0][1] * jacobian[1][2] - jacobian[0][2] * jacobian[1][1];
    normal[1] = jacobian[0][2] * jacobian[1][0] - jacobian[0][0] * jacobian[1][2];
    normal[2] = jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0];

    *integrationElement = sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    normal[0] /= *integrationElement;
    normal[1] /= *integrationElement;
    normal[2] /= *integrationElement;
    
}

inline REALTYPE3 getGlobalPoint(REALTYPE3 *corners, REALTYPE2 *localPoint)
{
    return corners[0] * (M_ONE - localPoint->x - localPoint->y) + corners[1] * localPoint->x + corners[2] * localPoint->y;

}

inline void getGlobalPointVec4(REALTYPE4 corners[3][3], REALTYPE2 *localPoint, REALTYPE4 globalPoint[3])
{
    globalPoint[0] =  corners[0][0] * (M_ONE - localPoint->x - localPoint->y) + corners[1][0] * localPoint->x + corners[2][0] * localPoint->y;
    globalPoint[1] =  corners[0][1] * (M_ONE - localPoint->x - localPoint->y) + corners[1][1] * localPoint->x + corners[2][1] * localPoint->y;
    globalPoint[2] =  corners[0][2] * (M_ONE - localPoint->x - localPoint->y) + corners[1][2] * localPoint->x + corners[2][2] * localPoint->y;

}

inline void getGlobalPointVec8(REALTYPE8 corners[3][3], REALTYPE2 *localPoint, REALTYPE8 globalPoint[3])
{
    globalPoint[0] =  corners[0][0] * (M_ONE - localPoint->x - localPoint->y) + corners[1][0] * localPoint->x + corners[2][0] * localPoint->y;
    globalPoint[1] =  corners[0][1] * (M_ONE - localPoint->x - localPoint->y) + corners[1][1] * localPoint->x + corners[2][1] * localPoint->y;
    globalPoint[2] =  corners[0][2] * (M_ONE - localPoint->x - localPoint->y) + corners[1][2] * localPoint->x + corners[2][2] * localPoint->y;

}

inline void getGlobalPointVec16(REALTYPE16 corners[3][3], REALTYPE2 *localPoint, REALTYPE16 globalPoint[3])
{
    globalPoint[0] =  corners[0][0] * (M_ONE - localPoint->x - localPoint->y) + corners[1][0] * localPoint->x + corners[2][0] * localPoint->y;
    globalPoint[1] =  corners[0][1] * (M_ONE - localPoint->x - localPoint->y) + corners[1][1] * localPoint->x + corners[2][1] * localPoint->y;
    globalPoint[2] =  corners[0][2] * (M_ONE - localPoint->x - localPoint->y) + corners[1][2] * localPoint->x + corners[2][2] * localPoint->y;

}

inline void getPiolaTransform(REALTYPE integrationElement, REALTYPE3 jacobian[2], 
                           REALTYPE referenceValues[3][2], REALTYPE3 result[3])
{
    for (int i = 0; i < 3; ++i)
        result[i] = M_ONE / integrationElement * (
            jacobian[0] * referenceValues[i][0] + jacobian[1] * referenceValues[i][1]); 


}

inline void getPiolaTransformVec4(REALTYPE4 integrationElement, REALTYPE4 jacobian[2][3], 
                           REALTYPE referenceValues[3][2], REALTYPE4 result[3][3])
{
    for (int base = 0; base < 3; ++base)
        for (int row = 0; row < 3; ++row)
            result[base][row] = M_ONE / integrationElement * (
                jacobian[0][row] * referenceValues[base][0] + jacobian[1][row] * referenceValues[base][1]
            );

}

inline void getPiolaTransformVec8(REALTYPE8 integrationElement, REALTYPE8 jacobian[2][3], 
                           REALTYPE referenceValues[3][2], REALTYPE8 result[3][3])
{
    for (int base = 0; base < 3; ++base)
        for (int row = 0; row < 3; ++row)
            result[base][row] = M_ONE / integrationElement * (
                jacobian[0][row] * referenceValues[base][0] + jacobian[1][row] * referenceValues[base][1]
            );

}

inline void getPiolaTransformVec16(REALTYPE16 integrationElement, REALTYPE16 jacobian[2][3], 
                           REALTYPE referenceValues[3][2], REALTYPE16 result[3][3])
{
    for (int base = 0; base < 3; ++base)
        for (int row = 0; row < 3; ++row)
            result[base][row] = M_ONE / integrationElement * (
                jacobian[0][row] * referenceValues[base][0] + jacobian[1][row] * referenceValues[base][1]
            );

}


inline void computeEdgeLength(REALTYPE3 corners[3], REALTYPE result[3])
{

    result[0] = distance(corners[1], corners[0]);
    result[1] = distance(corners[2], corners[0]);
    result[2] = distance(corners[2], corners[1]);
}

inline void computeEdgeLengthVec4(REALTYPE4 corners[3][3], REALTYPE4 result[3])
{

    REALTYPE4 diff;
    int j;

    result[0] = M_ZERO;
    result[1] = M_ZERO;
    result[2] = M_ZERO;

    for (j = 0; j < 3; ++j){
        diff = corners[1][j] - corners[0][j];
        result[0] += diff * diff;
    }

    for (j = 0; j < 3; ++j){
        diff = corners[2][j] - corners[0][j];
        result[1] += diff * diff;
    }

    for (j = 0; j < 3; ++j){
        diff = corners[2][j] - corners[1][j];
        result[2] += diff * diff;
    }

    result[0] = sqrt(result[0]);
    result[1] = sqrt(result[1]);
    result[2] = sqrt(result[2]);
}

inline void computeEdgeLengthVec8(REALTYPE8 corners[3][3], REALTYPE8 result[3])
{

    REALTYPE8 diff;
    int j;

    result[0] = M_ZERO;
    result[1] = M_ZERO;
    result[2] = M_ZERO;

    for (j = 0; j < 3; ++j){
        diff = corners[1][j] - corners[0][j];
        result[0] += diff * diff;
    }

    for (j = 0; j < 3; ++j){
        diff = corners[2][j] - corners[0][j];
        result[1] += diff * diff;
    }

    for (j = 0; j < 3; ++j){
        diff = corners[2][j] - corners[1][j];
        result[2] += diff * diff;
    }

    result[0] = sqrt(result[0]);
    result[1] = sqrt(result[1]);
    result[2] = sqrt(result[2]);
}
inline void computeEdgeLengthVec16(REALTYPE16 corners[3][3], REALTYPE16 result[3])
{

    REALTYPE16 diff;
    int j;

    result[0] = M_ZERO;
    result[1] = M_ZERO;
    result[2] = M_ZERO;

    for (j = 0; j < 3; ++j){
        diff = corners[1][j] - corners[0][j];
        result[0] += diff * diff;
    }

    for (j = 0; j < 3; ++j){
        diff = corners[2][j] - corners[0][j];
        result[1] += diff * diff;
    }

    for (j = 0; j < 3; ++j){
        diff = corners[2][j] - corners[1][j];
        result[2] += diff * diff;
    }

    result[0] = sqrt(result[0]);
    result[1] = sqrt(result[1]);
    result[2] = sqrt(result[2]);
}


inline uint elementsAreAdjacent(uint* element1, uint* element2, bool gridsAreDisjoint )
{
    return  !gridsAreDisjoint && 
            (element1[0] == element2[0] || element1[0] == element2[1] || element1[0] == element2[2] ||
            element1[1] == element2[0] || element1[1] == element2[1] || element1[1] == element2[2] ||
            element1[2] == element2[0] || element1[2] == element2[1] || element1[2] == element2[2]);
            
}

inline void updateNormals(size_t index, __global int *signs, REALTYPE3 *normal){

    *normal *= signs[index];

}

inline void updateNormalsVec4(size_t index[4], __global int *signs, REALTYPE4 normal[3]){

    REALTYPE4 signFlip = (REALTYPE4)(signs[index[0]], signs[index[1]],
                                     signs[index[2]], signs[index[3]]);

    normal[0] *= signFlip;
    normal[1] *= signFlip;
    normal[2] *= signFlip;
}

inline void updateNormalsVec8(size_t index[8], __global int *signs, REALTYPE8 normal[3]){

    REALTYPE8 signFlip = (REALTYPE8)(signs[index[0]], signs[index[1]],
                                     signs[index[2]], signs[index[3]],
                                     signs[index[4]], signs[index[5]],
                                     signs[index[6]], signs[index[7]]);


    normal[0] *= signFlip;
    normal[1] *= signFlip;
    normal[2] *= signFlip;
}

inline void updateNormalsVec16(size_t index[16], __global int *signs, REALTYPE16 normal[3]){

    REALTYPE16 signFlip = (REALTYPE16)(signs[index[0]], signs[index[1]],
                                      signs[index[2]], signs[index[3]],
                                      signs[index[4]], signs[index[5]],
                                      signs[index[6]], signs[index[7]],
                                      signs[index[8]], signs[index[9]],
                                      signs[index[10]], signs[index[11]],
                                      signs[index[12]], signs[index[13]],
                                      signs[index[14]], signs[index[15]]);



    normal[0] *= signFlip;
    normal[1] *= signFlip;
    normal[2] *= signFlip;
}
#endif
