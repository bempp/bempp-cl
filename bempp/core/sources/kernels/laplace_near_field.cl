#include "bempp_base_types.h"
#include "bempp_helpers.h"
#include "bempp_spaces.h"
#include "kernels.h"

#if VEC_LENGTH == 1
    typedef REALTYPE REALVECTYPE;
#elif VEC_LENGTH == 4
    typedef REALTYPE4 REALVECTYPE;
#elif VEC_LENGTH == 8
    typedef REALTYPE8 REALVECTYPE;
#elif VEC_LENGTH == 16
    typedef REALTYPE16 REALVECTYPE;
#endif

__kernel void evaluate_near_field(
        __global long *sourceIds,
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

   long targetIndexStart = targetIndexPtr[myId];
   long targetIndexEnd = targetIndexPtr[myId + 1];
   long numberOfTargets = targetIndexEnd - targetIndexStart;

   long sourceIndexStart = sourceIndexPtr[myId];
   long sourceIndexEnd = sourceIndexPtr[myId + 1];
   long numberOfSources = sourceIndexEnd - sourceIndexStart;

   long numSourceTiles = numberOfSources / VEC_LENGTH;

   REALTYPE myResults[MAX_NUM_TARGETS];
   uint myTargetElements[3];

   uint mySourceElements[3 * VEC_LENGTH];
   REALVECTYPE vecSources[3];
   REALTYPE sources[3];
   REALVECTYPE vecDiff[3];
   REALTYPE diff[3];
   REALTYPE targetVertex[3];
   REALVECTYPE myVecResult;
   REALVECTYPE myVecInput;
   REALVECTYPE vecTmp;
   REALTYPE myResult;
   REALTYPE tmp;

   for (long targetIndex = 0; targetIndex < numberOfTargets; targetIndex++)
       myResults[targetIndex] = M_ZERO;
   
   for (long sourceTile = 0; sourceTile < numSourceTiles; sourceTile++){
       long myTileStart = sourceIndexStart + VEC_LENGTH * sourceTile;
       
#if VEC_LENGTH == 1
       vecSources[0] = sourceVertices[3 * sourceIds[myTileStart] + 0];
       vecSources[1] = sourceVertices[3 * sourceIds[myTileStart] + 1];
       vecSources[2] = sourceVertices[3 * sourceIds[myTileStart] + 2];
       myVecInput = input[sourceIds[myTileStart]];

#elif VEC_LENGTH == 4
       vecSources[0] = (REALVECTYPE)(sourceVertices[3 * (sourceIds[myTileStart + 0]) + 0],
                                     sourceVertices[3 * (sourceIds[myTileStart + 1]) + 0],
                                     sourceVertices[3 * (sourceIds[myTileStart + 2]) + 0],
                                     sourceVertices[3 * (sourceIds[myTileStart + 3]) + 0]);
       vecSources[1] = (REALVECTYPE)(sourceVertices[3 * (sourceIds[myTileStart + 0]) + 1],
                                     sourceVertices[3 * (sourceIds[myTileStart + 1]) + 1],
                                     sourceVertices[3 * (sourceIds[myTileStart + 2]) + 1],
                                     sourceVertices[3 * (sourceIds[myTileStart + 3]) + 1]);
       vecSources[2] = (REALVECTYPE)(sourceVertices[3 * (sourceIds[myTileStart + 0]) + 2],
                                     sourceVertices[3 * (sourceIds[myTileStart + 1]) + 2],
                                     sourceVertices[3 * (sourceIds[myTileStart + 2]) + 2],
                                     sourceVertices[3 * (sourceIds[myTileStart + 3]) + 2]);
       myVecInput = (REALVECTYPE)(input[sourceIds[myTileStart + 0]],
                                  input[sourceIds[myTileStart + 1]],
                                  input[sourceIds[myTileStart + 2]],
                                  input[sourceIds[myTileStart + 3]]);
#elif VEC_LENGTH == 8
       vecSources[0] = (REALVECTYPE)(sourceVertices[3 * (sourceIds[myTileStart + 0]) + 0],
                                     sourceVertices[3 * (sourceIds[myTileStart + 1]) + 0],
                                     sourceVertices[3 * (sourceIds[myTileStart + 2]) + 0],
                                     sourceVertices[3 * (sourceIds[myTileStart + 3]) + 0],
                                     sourceVertices[3 * (sourceIds[myTileStart + 4]) + 0],
                                     sourceVertices[3 * (sourceIds[myTileStart + 5]) + 0],
                                     sourceVertices[3 * (sourceIds[myTileStart + 6]) + 0],
                                     sourceVertices[3 * (sourceIds[myTileStart + 7]) + 0]);
       vecSources[1] = (REALVECTYPE)(sourceVertices[3 * (sourceIds[myTileStart + 0]) + 1],
                                     sourceVertices[3 * (sourceIds[myTileStart + 1]) + 1],
                                     sourceVertices[3 * (sourceIds[myTileStart + 2]) + 1],
                                     sourceVertices[3 * (sourceIds[myTileStart + 3]) + 1],
                                     sourceVertices[3 * (sourceIds[myTileStart + 4]) + 1],
                                     sourceVertices[3 * (sourceIds[myTileStart + 5]) + 1],
                                     sourceVertices[3 * (sourceIds[myTileStart + 6]) + 1],
                                     sourceVertices[3 * (sourceIds[myTileStart + 7]) + 1]);
       vecSources[2] = (REALVECTYPE)(sourceVertices[3 * (sourceIds[myTileStart + 0]) + 2],
                                     sourceVertices[3 * (sourceIds[myTileStart + 1]) + 2],
                                     sourceVertices[3 * (sourceIds[myTileStart + 2]) + 2],
                                     sourceVertices[3 * (sourceIds[myTileStart + 3]) + 2],
                                     sourceVertices[3 * (sourceIds[myTileStart + 4]) + 2],
                                     sourceVertices[3 * (sourceIds[myTileStart + 5]) + 2],
                                     sourceVertices[3 * (sourceIds[myTileStart + 6]) + 2],
                                     sourceVertices[3 * (sourceIds[myTileStart + 7]) + 2]);
       myVecInput = (REALVECTYPE)(input[sourceIds[myTileStart + 0]],
                                  input[sourceIds[myTileStart + 1]],
                                  input[sourceIds[myTileStart + 2]],
                                  input[sourceIds[myTileStart + 3]],
                                  input[sourceIds[myTileStart + 4]],
                                  input[sourceIds[myTileStart + 5]],
                                  input[sourceIds[myTileStart + 6]],
                                  input[sourceIds[myTileStart + 7]]);
#elif VEC_LENGTH == 16
       vecSources[0] = (REALVECTYPE)(sourceVertices[3 * (sourceIds[myTileStart + 0]) + 0],
                                     sourceVertices[3 * (sourceIds[myTileStart + 1]) + 0],
                                     sourceVertices[3 * (sourceIds[myTileStart + 2]) + 0],
                                     sourceVertices[3 * (sourceIds[myTileStart + 3]) + 0],
                                     sourceVertices[3 * (sourceIds[myTileStart + 4]) + 0],
                                     sourceVertices[3 * (sourceIds[myTileStart + 5]) + 0],
                                     sourceVertices[3 * (sourceIds[myTileStart + 6]) + 0],
                                     sourceVertices[3 * (sourceIds[myTileStart + 7]) + 0],
                                     sourceVertices[3 * (sourceIds[myTileStart + 8]) + 0],
                                     sourceVertices[3 * (sourceIds[myTileStart + 9]) + 0],
                                     sourceVertices[3 * (sourceIds[myTileStart + 10]) + 0],
                                     sourceVertices[3 * (sourceIds[myTileStart + 11]) + 0],
                                     sourceVertices[3 * (sourceIds[myTileStart + 12]) + 0],
                                     sourceVertices[3 * (sourceIds[myTileStart + 13]) + 0],
                                     sourceVertices[3 * (sourceIds[myTileStart + 14]) + 0],
                                     sourceVertices[3 * (sourceIds[myTileStart + 15]) + 0]);
       vecSources[1] = (REALVECTYPE)(sourceVertices[3 * (sourceIds[myTileStart + 0]) + 1],
                                     sourceVertices[3 * (sourceIds[myTileStart + 1]) + 1],
                                     sourceVertices[3 * (sourceIds[myTileStart + 2]) + 1],
                                     sourceVertices[3 * (sourceIds[myTileStart + 3]) + 1],
                                     sourceVertices[3 * (sourceIds[myTileStart + 4]) + 1],
                                     sourceVertices[3 * (sourceIds[myTileStart + 5]) + 1],
                                     sourceVertices[3 * (sourceIds[myTileStart + 6]) + 1],
                                     sourceVertices[3 * (sourceIds[myTileStart + 7]) + 1],
                                     sourceVertices[3 * (sourceIds[myTileStart + 8]) + 1],
                                     sourceVertices[3 * (sourceIds[myTileStart + 9]) + 1],
                                     sourceVertices[3 * (sourceIds[myTileStart + 10]) + 1],
                                     sourceVertices[3 * (sourceIds[myTileStart + 11]) + 1],
                                     sourceVertices[3 * (sourceIds[myTileStart + 12]) + 1],
                                     sourceVertices[3 * (sourceIds[myTileStart + 13]) + 1],
                                     sourceVertices[3 * (sourceIds[myTileStart + 14]) + 1],
                                     sourceVertices[3 * (sourceIds[myTileStart + 15]) + 1]);
       vecSources[2] = (REALVECTYPE)(sourceVertices[3 * (sourceIds[myTileStart + 0]) + 2],
                                     sourceVertices[3 * (sourceIds[myTileStart + 1]) + 2],
                                     sourceVertices[3 * (sourceIds[myTileStart + 2]) + 2],
                                     sourceVertices[3 * (sourceIds[myTileStart + 3]) + 2],
                                     sourceVertices[3 * (sourceIds[myTileStart + 4]) + 2],
                                     sourceVertices[3 * (sourceIds[myTileStart + 5]) + 2],
                                     sourceVertices[3 * (sourceIds[myTileStart + 6]) + 2],
                                     sourceVertices[3 * (sourceIds[myTileStart + 7]) + 2],
                                     sourceVertices[3 * (sourceIds[myTileStart + 8]) + 2],
                                     sourceVertices[3 * (sourceIds[myTileStart + 9]) + 2],
                                     sourceVertices[3 * (sourceIds[myTileStart + 10]) + 2],
                                     sourceVertices[3 * (sourceIds[myTileStart + 11]) + 2],
                                     sourceVertices[3 * (sourceIds[myTileStart + 12]) + 2],
                                     sourceVertices[3 * (sourceIds[myTileStart + 13]) + 2],
                                     sourceVertices[3 * (sourceIds[myTileStart + 14]) + 2],
                                     sourceVertices[3 * (sourceIds[myTileStart + 15]) + 2]);
       myVecInput = (REALVECTYPE)(input[sourceIds[myTileStart + 0]],
                                  input[sourceIds[myTileStart + 1]],
                                  input[sourceIds[myTileStart + 2]],
                                  input[sourceIds[myTileStart + 3]],
                                  input[sourceIds[myTileStart + 4]],
                                  input[sourceIds[myTileStart + 5]],
                                  input[sourceIds[myTileStart + 6]],
                                  input[sourceIds[myTileStart + 7]],
                                  input[sourceIds[myTileStart + 8]],
                                  input[sourceIds[myTileStart + 9]],
                                  input[sourceIds[myTileStart + 10]],
                                  input[sourceIds[myTileStart + 11]],
                                  input[sourceIds[myTileStart + 12]],
                                  input[sourceIds[myTileStart + 13]],
                                  input[sourceIds[myTileStart + 14]],
                                  input[sourceIds[myTileStart + 15]]);
#endif

       for (uint i = 0; i < VEC_LENGTH; ++i)
           for (uint j = 0; j < 3; ++j)
               mySourceElements[3 * i + j] = sourceElements[3 * (sourceIds[myTileStart + i] / N_LOCAL_POINTS) + j];


       for (long targetVertexIndex = 0; targetVertexIndex < numberOfTargets; targetVertexIndex += 1){
            vecDiff[0] = vecSources[0] - targetVertices[3 * targetIds[targetIndexStart + targetVertexIndex] + 0];
            vecDiff[1] = vecSources[1] - targetVertices[3 * targetIds[targetIndexStart + targetVertexIndex] + 1];
            vecDiff[2] = vecSources[2] - targetVertices[3 * targetIds[targetIndexStart + targetVertexIndex] + 2];


            vecTmp = M_INV_4PI * rsqrt(vecDiff[0] * vecDiff[0] + 
                    vecDiff[1] * vecDiff[1] + vecDiff[2] * vecDiff[2]);
            // Check if the source and target do not belong to adjacent elements. If yes

            myTargetElements[0] = targetElements[
                3 * (targetIds[targetIndexStart + targetVertexIndex] / N_LOCAL_POINTS) + 0];
            myTargetElements[1] = targetElements[
                3 * (targetIds[targetIndexStart + targetVertexIndex] / N_LOCAL_POINTS) + 1];
            myTargetElements[2] = targetElements[
                3 * (targetIds[targetIndexStart + targetVertexIndex] / N_LOCAL_POINTS) + 2];

            myVecResult = myVecInput * vecTmp;


            for (int index = 0; index < VEC_LENGTH; ++index)
                if (!elementsAreAdjacent(&mySourceElements[3 * index], &myTargetElements[0], GRIDS_DISJOINT))
                    myResults[targetVertexIndex] += VEC_ELEMENT(myVecResult, index);

                
       }

   }

   // Now do the remainder case with non-vector variables.

   for (long mySourceIndex = sourceIndexStart + VEC_LENGTH * numSourceTiles; 
           mySourceIndex < sourceIndexEnd; mySourceIndex++){


       sources[0] = sourceVertices[3 * sourceIds[mySourceIndex] + 0];
       sources[1] = sourceVertices[3 * sourceIds[mySourceIndex] + 1];
       sources[2] = sourceVertices[3 * sourceIds[mySourceIndex] + 2];


       for (int j = 0; j < 3; ++j)
           mySourceElements[j] = sourceElements[3 * (sourceIds[mySourceIndex] / N_LOCAL_POINTS) + j];


       for (long targetVertexIndex = 0; targetVertexIndex < numberOfTargets; targetVertexIndex += 1){
            diff[0] = sources[0] - targetVertices[
                3 * targetIds[targetIndexStart + targetVertexIndex] + 0];
            diff[1] = sources[1] - targetVertices[
                3 * targetIds[targetIndexStart + targetVertexIndex] + 1];
            diff[2] = sources[2] - targetVertices[
                3 * targetIds[targetIndexStart + targetVertexIndex] + 2];

            tmp = M_INV_4PI * rsqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);

            myTargetElements[0] = targetElements[
                3 * (targetIds[targetIndexStart + targetVertexIndex] / N_LOCAL_POINTS) + 0];
            myTargetElements[1] = targetElements[
                3 * (targetIds[targetIndexStart + targetVertexIndex] / N_LOCAL_POINTS) + 1];
            myTargetElements[2] = targetElements[
                3 * (targetIds[targetIndexStart + targetVertexIndex] / N_LOCAL_POINTS) + 2];

            if (!elementsAreAdjacent(&mySourceElements[0], &myTargetElements[0], false)) 
                myResults[targetVertexIndex] += tmp * input[sourceIds[mySourceIndex]];

       }

   }


   for (long targetVertexIndex = 0; targetVertexIndex < numberOfTargets; targetVertexIndex += 1){
       long myTargetId = targetIds[targetIndexStart + targetVertexIndex];
       result[myTargetId] = myResults[targetVertexIndex];

   }



}
