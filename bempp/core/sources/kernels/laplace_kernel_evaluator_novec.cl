#include "bempp_base_types.h"

__kernel void evaluate_kernel(__global REALTYPE* nodes,
                              __global REALTYPE* result) {
  size_t gid[2] = {get_global_id(0), get_global_id(1)};
  size_t gcols = get_global_size(1);
  size_t tmpx = gid[0];
  size_t tmpy = gid[1];

  size_t indx[3];
  size_t indy[3];

  size_t i;

  for (size_t i = 0; i < 3; ++i) {
    indx[2 - i] = tmpx % NNODES;
    indy[2 - i] = tmpy % NNODES;
    tmpx /= NNODES;
    tmpy /= NNODES;
  }

  REALTYPE3 xlow = (REALTYPE3)(X_XMIN, X_YMIN, X_ZMIN);
  REALTYPE3 xhigh = (REALTYPE3)(X_XMAX, X_YMAX, X_ZMAX);

  REALTYPE3 diam_x = (xhigh - xlow) / 2;
  REALTYPE3 average_x = (xhigh + xlow) / 2;

  REALTYPE3 ylow = (REALTYPE3)(Y_XMIN, Y_YMIN, Y_ZMIN);
  REALTYPE3 yhigh = (REALTYPE3)(Y_XMAX, Y_YMAX, Y_ZMAX);

  REALTYPE3 diam_y = (yhigh - ylow) / 2;
  REALTYPE3 average_y = (yhigh + ylow) / 2;

  REALTYPE3 xpoint =
      (REALTYPE3)(nodes[indx[0]], nodes[indx[1]], nodes[indx[2]]);

  REALTYPE3 ypoint =
      (REALTYPE3)(nodes[indy[0]], nodes[indy[1]], nodes[indy[2]]);

  REALTYPE3 xglobal = average_x + diam_x * xpoint;
  REALTYPE3 yglobal = average_y + diam_y * ypoint;

  result[gid[0] * gcols + gid[1]] = M_INV_4PI / distance(xglobal, yglobal);
}