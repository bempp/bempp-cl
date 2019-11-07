#include "bempp_base_types.h"
#include "bempp_helpers.h"
#include "bempp_spaces.h"
#include "kernels.h"

#if VEC_LENGTH == 1
    typedef REALTYPE REALVECTYPE;
#elif VEC_LENTH == 4
    typedef REALTYPE4 REALVECTYPE;
#elif VEC_LENGTH == 8
    typedef REALTYPE8 REALVECTYPE;
#elif VEC_LENGTH == 16
    typedef REALTYPE16 REALVECTYPE;
#endif

__kernel void evaluate_near_field(
        __global long *targetIds,
        __global long *sourceIndexPtr,
        __global long *targetIndexPtr,
        __global uint *sourceElements,
        __global uint *targetElements,
        __global REALTYPE *sourceVertices,
        __global REALTYPE *targetVertices,
        __global REALTYPE *input,
        __global REALTYPE *result
        )
{
   int myId = get_global_id(0);

   int targetIndexStart = targetIndexPtr[myId];
   int targetIndexEnd = targetIndexPtr[myId + 1];
   int numberOfTargets = targetIndexEnd - targetIndexStart;

   int sourceIndexStart = sourceIndexPtr[myId];
   int sourceIndexEnd = sourceIndexPtr[myId + 1];
   int numberOfSources = sourceIndexEnd - sourceIndexStart;

   int numSourceTiles = numberOfSources / VEC_LENGTH;

   REALTYPE myResults[MAX_NUM_TARGETS];
   REALTYPE myTargets[3 * MAX_NUM_TARGETS];
   uint myTargetElements[3 * MAX_NUM_TARGETS];

   uint mySourceElements[3 * VEC_LENGTH];
   REALVECTYPE sources[3];
   REALVECTYPE vecDiff[3];
   REALTYPE diff[3];
   REALTYPE targetVertex[3];
   REALVECTYPE myResult;
   REALVECTYPE myInput;
   REALVECTYPE tmp;

   for (int targetIndex = 0; targetIndex < numberOfTargets; targetIndex++){
       myResults[targetIndex] = M_ZERO;
       myTargets[3 * targetIndex + 0] = targetVertices[3 * (targetIndexStart + targetIndex) + 0];
       myTargets[3 * targetIndex + 1] = targetVertices[3 * (targetIndexStart + targetIndex) + 1];
       myTargets[3 * targetIndex + 2] = targetVertices[3 * (targetIndexStart + targetIndex) + 2];
       myTargetElements[3 * targetIndex + 0] = targetElements[3 * (targetIndexStart + targetIndex) + 0];
       myTargetElements[3 * targetIndex + 1] = targetElements[3 * (targetIndexStart + targetIndex) + 1];
       myTargetElements[3 * targetIndex + 2] = targetElements[3 * (targetIndexStart + targetIndex) + 2];
   }



   for (int sourceTile = 0; sourceTile < numSourceTiles; sourceTile++){
       int myTileStart = sourceIndexStart + VEC_LENGTH * sourceTile;
#if VEC_LENGTH == 1
       sources[0] = sourceVertices[3 * myTileStart + 0];
       sources[1] = sourceVertices[3 * myTileStart + 1];
       sources[2] = sourceVertices[3 * myTileStart + 2];
       myInput = input[myTileStart];
#endif


       for (int i = 0; i < VEC_LENGTH; ++i)
           for (int j = 0; j < 3; ++j)
               mySourceElements[3 * i + j] = sourceElements[3 * (myTileStart + i) + j];


       for (int targetVertexIndex = 0; targetVertexIndex < numberOfTargets; targetVertexIndex += 1){
            
            diff[0] = sources[0] - myTargets[3 * targetVertexIndex + 0];
            diff[1] = sources[1] - myTargets[3 * targetVertexIndex + 1];
            diff[2] = sources[2] - myTargets[3 * targetVertexIndex + 2];


            tmp = M_INV_4PI * rsqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);
            // Check if the source and target do not belong to adjacent elements. If yes

            for (int index = 0; index < VEC_LENGTH; ++index)
                if (elementsAreAdjacent(&mySourceElements[3 * index],
                                                       &myTargetElements[3 * targetVertexIndex],
                                                       false))
                    VEC_ELEMENT(tmp, index) = M_ZERO;

            myResult = myInput * tmp;

            for (int index = 0; index < VEC_LENGTH; ++index)
                myResults[targetVertexIndex] += VEC_ELEMENT(myResult, index);


       }

   }

   for (int targetVertexIndex = 0; targetVertexIndex < numberOfTargets; targetVertexIndex += 1){
       long myTargetId = targetIds[targetIndexStart + targetVertexIndex];
       result[myTargetId] = myResults[targetVertexIndex];

   }


}
