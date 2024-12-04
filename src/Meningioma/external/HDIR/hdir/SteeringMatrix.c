#include "mex.h"
#include "Mates.h"
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <stdlib.h>

/* 

In order to compile this MEX function, type the following at the MATLAB prompt:
32-bit Windows:
mex SteeringMatrix.c MatesLap.c Debugging.c lapack.a blas.a libf2c.a
64-bit Windows:
mex LINKFLAGS="$LINKFLAGS /NODEFAULTLIB:LIBCMT" SteeringMatrix.c MatesLap.c Debugging.c clapack_nowrap.lib BLAS_nowrap.lib libf2c.lib

Usage:
C = SteeringMatrix(GradX, GradY, Resp, wsize, lambda, alpha);

Notes:
GradX=Gradient in the horizontal direction
GradY=Gradient in the vertical direction
Resp=Probability that a given pixel is not impulse corrupted
wsize=size of local analysis window
lambda=regularization for the elongation parameter
alpha=structure sensitive parameter
C=map of steering matrices


*/

/* Extend a 2D matrix (mirror extension at the edges) */
void Extend(double *Dest, double *Orig, mwSize N, mwSize M, mwSize radius);

void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{  
    double *Responsibilities, *GradX, *GradY, *C, *K;
    double *GradXExt, *GradYExt, *G, *U;
    double S[2];
    double V[4];
    double lambda, alpha, aux, SumK, S1, S2;
    double AuxMat[4];
    double AuxMat2[4];
    mwSize radius, wsize, M, N, NdxWin, NdxCol, NdxRow;
    mwSize Dims[4];
    mxArray *My_plhs[1];
    mxArray *My_prhs[2];

    /* Get input data */
    wsize = (mwSize)mxGetScalar(prhs[3]);  /* Ensure wsize is mwSize */
    lambda = mxGetScalar(prhs[4]);
    alpha = mxGetScalar(prhs[5]);
    GradX = mxGetPr(prhs[0]);
    GradY = mxGetPr(prhs[1]);
    Responsibilities = mxGetPr(prhs[2]);

    /* Create output array */
    N = (mwSize)mxGetM(prhs[0]);  /* Number of rows in the input image */
    M = (mwSize)mxGetN(prhs[0]);  /* Number of columns in the input image */
    Dims[0] = 2;
    Dims[1] = 2;
    Dims[2] = N;
    Dims[3] = M;
    plhs[0] = mxCreateNumericArray(4, Dims, mxDOUBLE_CLASS, mxREAL);
    C = mxGetPr(plhs[0]);

    /* Get working variables */
    if (wsize % 2 == 0) {
        wsize++;
    }
    radius = (mwSize)(wsize / 2);

    /* Get the filter */
    My_prhs[0] = mxCreateString("disk");
    My_prhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
    (*mxGetPr(My_prhs[1])) = (double)radius;
    mexCallMATLAB(1, My_plhs, 2, My_prhs, "fspecial");
    K = mxGetPr(My_plhs[0]);
    aux = K[radius * wsize + radius];
    SumK = 0.0;
    for (NdxWin = 0; NdxWin < wsize * wsize; NdxWin++) {
        K[NdxWin] /= aux;
        SumK += K[NdxWin];
    }

    /* Prepare auxiliary arrays */
    GradXExt = mxMalloc((N + 2 * radius) * (M + 2 * radius) * sizeof(double));
    GradYExt = mxMalloc((N + 2 * radius) * (M + 2 * radius) * sizeof(double));
    G = mxMalloc((wsize * wsize) * 2 * sizeof(double));
    U = mxMalloc((wsize * wsize) * 2 * sizeof(double));

    /* Extend the gradient maps */
    Extend(GradXExt, GradX, N, M, radius);
    Extend(GradYExt, GradY, N, M, radius);

    /* Estimate the steering matrices */
    for (NdxCol = 0; NdxCol < M; NdxCol++) {
        for (NdxRow = 0; NdxRow < N; NdxRow++) {
            /* Get the window with the relevant gradient values */
            for (NdxWin = 0; NdxWin < wsize; NdxWin++) {
                memcpy(G + NdxWin * wsize, GradXExt + (NdxCol + NdxWin) * (N + 2 * radius) + NdxRow,
                       wsize * sizeof(double));
                memcpy(G + wsize * wsize + NdxWin * wsize, GradYExt + (NdxCol + NdxWin) * (N + 2 * radius) + NdxRow,
                       wsize * sizeof(double));
            }

            /* Multiply by the filter */
            for (NdxWin = 0; NdxWin < wsize * wsize; NdxWin++) {
                G[NdxWin] *= K[NdxWin];
                G[wsize * wsize + NdxWin] *= K[NdxWin];
            }

            /* Compute Singular Value Decomposition */
            EconomySVD(G, S, U, V, wsize * wsize, 2);

            /* Regularized values */
            S1 = (S[0] + lambda) / (S[1] + lambda);
            S2 = (S[1] + lambda) / (S[0] + lambda);

            /* Compute the steering matrix */
            MatrixProduct(V, V, AuxMat, 2, 1, 2);
            ScalarMatrixProduct(S1, AuxMat, AuxMat, 2, 2);
            MatrixProduct(V + 2, V + 2, AuxMat2, 2, 1, 2);
            ScalarMatrixProduct(S2, AuxMat2, AuxMat2, 2, 2);
            MatrixSum(AuxMat, AuxMat2, AuxMat, 2, 2);

            /* Apply adaptive weighting and write on output */
            aux = pow((S[0] * S[1] + 0.0000001) / SumK, alpha);
            ScalarMatrixProduct(aux * Responsibilities[NdxCol * N + NdxRow], AuxMat,
                                C + 4 * (NdxCol * N + NdxRow), 2, 2);
        }
    }

    /* Release memory */
    mxDestroyArray(My_plhs[0]);
    mxDestroyArray(My_prhs[0]);
    mxDestroyArray(My_prhs[1]);
    mxFree(GradXExt);
    mxFree(GradYExt);
    mxFree(G);
    mxFree(U);
}

/* Extend a 2D matrix (mirror extension at the edges) */
void Extend(double *Dest, double *Orig, mwSize N, mwSize M, mwSize radius)
{
    mwSize Offset, NdxCol, NdxRow, cnt;

    /* Copy the original matrix to the destination */
    Offset = radius * (N + 2 * radius) + radius;
    for (NdxCol = 0; NdxCol < M; NdxCol++) {
        memcpy(Dest + Offset + NdxCol * (N + 2 * radius), Orig + NdxCol * N,
               N * sizeof(double));
    }

    /* Extend to the left */
    for (NdxCol = 0; NdxCol < radius; NdxCol++) {
        memcpy(Dest + radius + NdxCol * (N + 2 * radius), Orig + (radius - NdxCol) * N,
               N * sizeof(double));
    }

    /* Extend to the right */
    Offset = (M + radius) * (N + 2 * radius) + radius;
    for (NdxCol = 0; NdxCol < radius; NdxCol++) {
        memcpy(Dest + Offset + NdxCol * (N + 2 * radius), Orig + (M - 2 - NdxCol) * N,
               N * sizeof(double));
    }

    /* Extend to the top */
    for (NdxCol = 0; NdxCol < M + 2 * radius; NdxCol++) {
        for (NdxRow = 0; NdxRow < radius; NdxRow++) {
            Dest[NdxRow + NdxCol * (N + 2 * radius)] = Dest[2 * radius - NdxRow + NdxCol * (N + 2 * radius)];
        }
    }

    /* Extend to the bottom */
    for (NdxCol = 0; NdxCol < M + 2 * radius; NdxCol++) {
        for (NdxRow = radius + N, cnt = 0; NdxRow < N + 2 * radius; NdxRow++, cnt++) {
            Dest[NdxRow + NdxCol * (N + 2 * radius)] = Dest[N + radius - 2 - cnt + NdxCol * (N + 2 * radius)];
        }
    }
}
