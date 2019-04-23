#ifndef bempp_base_types_h
#define bempp_base_types_h

/* Heler to concatenate two strings */
#define CAT(X, B) X ## B

/* Multiply two complex numbers */
#define CMP_MULT_REAL(A, B) ((A)[0] * (B)[0] - (A)[1] * (B)[1])
#define CMP_MULT_IMAG(A, B) ((A)[0] * (B)[1] + (A)[1] * (B)[0]) 


#if PRECISION == 0
    #define M_ZERO 0.0f
    #define M_ONE 1.0f
    #define M_TWO 2.0f
    #define M_ZERO4 (float4)(0.0f)
    #define M_ZERO8 (float8)(0.0f)
    #define M_ZERO16 (float16)(0.0f)
    #define M_4PI 12.566371f
    #define M_INV_4PI 0.079577468f

    typedef float REALTYPE;
    typedef float2 REALTYPE2;
    typedef float3 REALTYPE3;
    typedef float4 REALTYPE4;
    typedef float8 REALTYPE8;
    typedef float16 REALTYPE16;
#endif

#if PRECISION == 1
    #define M_ZERO 0.0
    #define M_ONE 1.0
    #define M_TWO 2.0
    #define M_ZERO4 (double4)(0.0)
    #define M_ZERO8 (double8)(0.0)
    #define M_ZERO16 (double16)(0.0)
    #define M_4PI 12.566370614359172
    #define M_INV_4PI 0.07957747154594767

    typedef double REALTYPE;
    typedef double2 REALTYPE2;
    typedef double3 REALTYPE3;
    typedef double4 REALTYPE4;
    typedef double8 REALTYPE8;
    typedef double16 REALTYPE16;
#endif

typedef struct Geometry
{
    REALTYPE3 corners[3];

    REALTYPE3 jac[2];
    REALTYPE3 jac_inv_trans[2];

    REALTYPE3 normal;
    REALTYPE int_elem;
    REALTYPE volume;
} Geometry;

#define VEC_ELEMENT(A, INDEX) ((REALTYPE*)&A)[INDEX]


#endif
