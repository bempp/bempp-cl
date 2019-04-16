#include "bempp_base_types.h"
#include "bempp_helpers.h"

/*
Evaluate the Lagrange function l(x) for each quadrature point x.
If a coordinate of a quadrature point is identical to an interpolation
point the corresponding term is not multiplied (due to being zero)
*/

__kernel void evaluate_lagrange_pol_on_leafs(__global REALTYPE* grid,
                                             __global REALTYPE* boxLowerBounds,
                                             __global REALTYPE* lagrangeValues;
                                             __global uint* boxIndices,
                                             __constant REALTYPE* interpolationPoints,
                                             __constant REALTYPE* quadPoints,
                                             double boxDiameter){

size_t gid, boxIndex, index, interPointIndex;
REALTYPE3 corners[3];
REALTYPE myLagrangeValues[NUM_QUAD_POINTS];
REALTYPE myInterPolationPoints[NUM_INTERP_POINTS];

REALTYPE2 localPoint;
REALTYPE3 globalPoints[NUM_QUAD_POINTS];
REALTYPE3 boxCorner;
REALTYPE3 diff;

REALTYPE3 interpolationCoordinate;


gid = get_global_id(0);

boxIndex = boxIndices[gid];

boxCorner = (REALTYPE3)(boxLowerBounds[3 * boxIndex + 0], 
                        boxLowerBounds[3 * boxIndex + 1],
                        boxLowerBounds[3 * boxIndex + 2]);


getCorners(*grid, gid, &corners);

for (size_t index = 0; index < NUM_INTERP_POINTS; ++index)
    myInterpolationPoints[index] = interpolationPoints[index]; 

for (size_t index = 0; index < NUM_QUAD_POINTS; ++index ){
    myLagrangeValues[index] = 1;
}

for (size_t index = 0; index < NUM_QUAD_POINTS; ++index ){
    localPoint = quadPoints[index];
    globalPoints[i] = getGlobalPoint(corners, &localPoint);
}


/*
The following for loop does not need to be a triple loop along tensor points.
We only need to multiply the Lagrange function along each dimension as long as it is not zero
with respect to the corresponding coordinate.
*/

for (size_t interPointIndex = 0; interPointIndex < NUM_INTERP_POINTS; ++interPointIndex){
    interpolationCoordinate = boxCorner + (M_ONE + myInterpolationPoints[interPointIndex]) / M_TWO * boxDiameter;

    for (size_t index = 0; index < QUAD_POINTS; ++index){

        diff = globalPoints[index] - interpolationCoordinate;
        if (diff.x != M_ZERO) myLagrangeValues[index] *= diff.x;
        if (diff.y != M_ZERO) myLagrangeValues[index] *= diff.y;
        if (diff.z != M_ZERO) myLagrangeValues[index] *= diff.z;
     }
}

for (size_t index = 0; index < NUM_QUAD_POINTS; ++index ){
    lagrangeValues[NUM_QUAD_POINTS * gid + index] = myLagrangeValues[index];
}


}


