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
mex SteeringKernelRegression.c MatesLap.c Debugging.c lapack.a blas.a libf2c.a
64-bit Windows:
mex LINKFLAGS="$LINKFLAGS /NODEFAULTLIB:LIBCMT" SteeringKernelRegression.c MatesLap.c Debugging.c clapack_nowrap.lib BLAS_nowrap.lib libf2c.lib

Usage:
[RecImg, GradX, GradY] = SteeringKernelRegression(NoisyImg, Resp, h, C, ksize);

Notes:
RecImg=Reconstructed image
GradX=Gradient in the horizontal direction
GradY=Gradient in the vertical direction
NoisyImg=Input (noisy) image
Resp=Probability that a given pixel is not impulse corrupted
h=Kernel regression global smoothing parameter
C=steering matrices map
ksize=Kernel size


*/


/* Extend a 2D matrix (mirror extension at the edges) */
void Extend(double *Dest,double *Orig,int N, int M,int radius);

void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{  
	double *Responsibilities,*NoisyImg,*RecImg,*GradX,*GradY,*C;
	double *Xx,*XxT,*Xw,*XwT,*XxTXw,*Inv,*A,*AT,*XCoord,*YCoord,*W;
	double *NoisyImgExt,*ResponsibilitiesExt,*Window,*WindowResp;
	double *WindowC11,*WindowC12,*WindowC22,*WindowSqrtDetC;
	double *C11,*C12,*C22,*C11Ext,*C12Ext,*C22Ext,*SqrtDetC,*SqrtDetCExt;
	double h;
	int radius,ksize,M,N,NdxWin,Ndx,Cnt,NdxCol,NdxRow;

    /* Get input data */
    h=mxGetScalar(prhs[2]);    
	ksize=(int)mxGetScalar(prhs[4]);    
	NoisyImg=mxGetPr(prhs[0]);
	Responsibilities=mxGetPr(prhs[1]);
	C=mxGetPr(prhs[3]);
    
    
    /* Create output arrays */
	N=mxGetM(prhs[0]);  /* Number of rows in the input image */
	M=mxGetN(prhs[0]);  /* Number of columns in the input image */
    plhs[0]=mxCreateDoubleMatrix(N,M,mxREAL);
	RecImg=mxGetPr(plhs[0]);
	plhs[1]=mxCreateDoubleMatrix(N,M,mxREAL);
	GradX=mxGetPr(plhs[1]);
	plhs[2]=mxCreateDoubleMatrix(N,M,mxREAL);
	GradY=mxGetPr(plhs[2]);
    
    
    /* Get working variables */
	radius = (ksize-1)/2;
         
    /* Prepare auxiliary arrays */
    Xx=mxMalloc((ksize*ksize)*6*sizeof(double));
	XxT=mxMalloc(6*(ksize*ksize)*sizeof(double));
	Xw=mxMalloc((ksize*ksize)*6 *sizeof(double));
	XwT=mxMalloc(6*(ksize*ksize)*sizeof(double));
	XxTXw=mxMalloc(6*6*sizeof(double));
	Inv=mxMalloc(6*6*sizeof(double));
	A=mxMalloc(6*(ksize*ksize)*sizeof(double));
	AT=mxMalloc((ksize*ksize)*6*sizeof(double));
	XCoord=mxMalloc(ksize*ksize*sizeof(double));
	YCoord=mxMalloc(ksize*ksize*sizeof(double));	
	W=mxMalloc(ksize*ksize*sizeof(double));
	NoisyImgExt=mxMalloc((N+2*radius)*(M+2*radius)*sizeof(double));
	ResponsibilitiesExt=mxMalloc((N+2*radius)*(M+2*radius)*sizeof(double));
	C11=mxMalloc(N*M*sizeof(double));
	C12=mxMalloc(N*M*sizeof(double));
	C22=mxMalloc(N*M*sizeof(double));
	SqrtDetC=mxMalloc(N*M*sizeof(double));
	C11Ext=mxMalloc((N+2*radius)*(M+2*radius)*sizeof(double));	
	C12Ext=mxMalloc((N+2*radius)*(M+2*radius)*sizeof(double));	
	C22Ext=mxMalloc((N+2*radius)*(M+2*radius)*sizeof(double));	
	SqrtDetCExt=mxMalloc((N+2*radius)*(M+2*radius)*sizeof(double));	
	Window=mxMalloc(ksize*ksize*sizeof(double));
	WindowResp=mxMalloc(ksize*ksize*sizeof(double));
	WindowC11=mxMalloc(ksize*ksize*sizeof(double));
	WindowC12=mxMalloc(ksize*ksize*sizeof(double));
	WindowC22=mxMalloc(ksize*ksize*sizeof(double));
	WindowSqrtDetC=mxMalloc(ksize*ksize*sizeof(double));


	/* Prepare window coordinates */
	for(NdxWin=0;NdxWin<ksize*ksize;NdxWin++)
	{
		XCoord[NdxWin]=(NdxWin%ksize)-radius;
		YCoord[NdxWin]=(NdxWin/ksize)-radius;
	}

	

	/* Prepare feature matrix and its traspose */
	for(NdxWin=0;NdxWin<ksize*ksize;NdxWin++)
	{
		Xx[NdxWin]=1.0;
		Xx[ksize*ksize+NdxWin]=XCoord[NdxWin];
		Xx[2*ksize*ksize+NdxWin]=YCoord[NdxWin];
		Xx[3*ksize*ksize+NdxWin]=XCoord[NdxWin]*XCoord[NdxWin];
		Xx[4*ksize*ksize+NdxWin]=XCoord[NdxWin]*YCoord[NdxWin];
		Xx[5*ksize*ksize+NdxWin]=YCoord[NdxWin]*YCoord[NdxWin];
	}	
	Traspose(Xx,XxT,ksize*ksize,6);

	/* Prepare steering matrices */
	for(Ndx=0;Ndx<N*M;Ndx++)
	{
		C11[Ndx]=C[4*Ndx];
		C12[Ndx]=C[4*Ndx+1];
		C22[Ndx]=C[4*Ndx+3];
		SqrtDetC[Ndx]=sqrt(C[4*Ndx]*C[4*Ndx+3]-C[4*Ndx+1]*C[4*Ndx+1]);
	}

	/* Extend the noisy image, the responsibilities map and the maps for steering matrices*/
	Extend(NoisyImgExt,NoisyImg,N,M,radius);	
	Extend(ResponsibilitiesExt,Responsibilities,N,M,radius);	
	Extend(C11Ext,C11,N,M,radius);
	Extend(C12Ext,C12,N,M,radius);
	Extend(C22Ext,C22,N,M,radius);
	Extend(SqrtDetCExt,SqrtDetC,N,M,radius);

	/*---------*/

	/* Estimate the original image and the gradients */
	for(NdxCol=0;NdxCol<M;NdxCol++)
	{
		for(NdxRow=0;NdxRow<N;NdxRow++)
		{
			/* Get the window with the relevant noisy samples and the rest of parameters */
			for(NdxWin=0;NdxWin<ksize;NdxWin++)
			{
				memcpy(Window+NdxWin*ksize,NoisyImgExt+(NdxCol+NdxWin)*(N+2*radius)+NdxRow,
					ksize*sizeof(double));
				memcpy(WindowC11+NdxWin*ksize,C11Ext+(NdxCol+NdxWin)*(N+2*radius)+NdxRow,
					ksize*sizeof(double));
				memcpy(WindowC12+NdxWin*ksize,C12Ext+(NdxCol+NdxWin)*(N+2*radius)+NdxRow,
					ksize*sizeof(double));
				memcpy(WindowC22+NdxWin*ksize,C22Ext+(NdxCol+NdxWin)*(N+2*radius)+NdxRow,
					ksize*sizeof(double));
				memcpy(WindowSqrtDetC+NdxWin*ksize,SqrtDetCExt+(NdxCol+NdxWin)*(N+2*radius)+NdxRow,
					ksize*sizeof(double));
				memcpy(WindowResp+NdxWin*ksize,ResponsibilitiesExt+(NdxCol+NdxWin)*(N+2*radius)+NdxRow,
					ksize*sizeof(double));
			}
			
			/* Prepare weight matrix */	
			for(NdxWin=0;NdxWin<ksize*ksize;NdxWin++)
			{	
				W[NdxWin]=Xx[ksize*ksize+NdxWin]*
					(WindowC11[NdxWin]*Xx[ksize*ksize+NdxWin]+WindowC12[NdxWin]*Xx[2*ksize*ksize+NdxWin])
					+Xx[2*ksize*ksize+NdxWin]*
					(WindowC12[NdxWin]*Xx[ksize*ksize+NdxWin]+WindowC22[NdxWin]*Xx[2*ksize*ksize+NdxWin]);
				W[NdxWin]=exp(-(0.5/(h*h)) * W[NdxWin])*WindowSqrtDetC[NdxWin];
			}

			/* The weights multiplied by the responsibilities */			
			for(NdxWin=0;NdxWin<ksize*ksize;NdxWin++)
			{
				WindowResp[NdxWin]*=W[NdxWin];
			}
			

			/* Equivalent kernel and its traspose */
			Ndx=0;
			for(Cnt=0;Cnt<6;Cnt++)
			{
				for(NdxWin=0;NdxWin<ksize*ksize;NdxWin++)
				{
					Xw[Ndx]=Xx[Ndx]*WindowResp[NdxWin];
					Ndx++;
				}					
			}			
			Traspose(Xw,XwT,ksize*ksize,6);

			/* XxTXw = Xx^T * Xw */
			MatrixProduct(XxT,Xw,XxTXw,6,ksize*ksize,6);
			/* Add 0.00001 to the elements of the diagonal */
			SumDiagonalConstant(XxTXw,0.00001,NULL,6);
			/* Invert it */
			Inverse(XxTXw,Inv,6);

			/* A=Inv*(Xw^T)*/
			MatrixProduct(Inv,XwT,A,6,6,ksize*ksize);

			Traspose(A,AT,6,ksize*ksize);

			/* Estimate the original image and the gradients at this position */
			(*RecImg)=0.0;
			(*GradX)=0.0;
			(*GradY)=0.0;
			for(NdxWin=0;NdxWin<ksize*ksize;NdxWin++)
			{
				(*RecImg)+=AT[NdxWin]*Window[NdxWin];
				(*GradX)+=AT[ksize*ksize+NdxWin]*Window[NdxWin];
				(*GradY)+=AT[2*ksize*ksize+NdxWin]*Window[NdxWin];
			}
						
			/* Advance output pointers */
			RecImg++;
			GradX++;
			GradY++;

			

		}

	}

	/* Release memory */
	mxFree(Xx);
	mxFree(XxT);
	mxFree(Xw);
	mxFree(XwT);
	mxFree(XxTXw);
	mxFree(Inv);
	mxFree(A);
	mxFree(AT);
	mxFree(XCoord);
	mxFree(YCoord);
	mxFree(W);
	mxFree(NoisyImgExt);
	mxFree(ResponsibilitiesExt);
	mxFree(Window);
	mxFree(WindowResp);
	mxFree(C11);
	mxFree(C12);
	mxFree(C22);
	mxFree(SqrtDetC);
	mxFree(C11Ext);
	mxFree(C12Ext);
	mxFree(C22Ext);
	mxFree(SqrtDetCExt);
	mxFree(WindowC11);
	mxFree(WindowC12);
	mxFree(WindowC22);
	mxFree(WindowSqrtDetC);
    
}    



/* Extend a 2D matrix (mirror extension at the edges) */
void Extend(double *Dest,double *Orig,int N, int M,int radius)
{
	int Offset,NdxCol,NdxRow,cnt;

	/* Copy the original matrix to the destination */
	Offset=radius*(N+2*radius)+radius;
	for(NdxCol=0;NdxCol<M;NdxCol++)
	{		
		memcpy(Dest+Offset+NdxCol*(N+2*radius),Orig+NdxCol*N,
			N*sizeof(double));
	}

	/* Extend to the left */
	for(NdxCol=0;NdxCol<radius;NdxCol++)
	{		
		memcpy(Dest+radius+NdxCol*(N+2*radius),Orig+(radius-NdxCol)*N,
			N*sizeof(double));
	}

	/* Extend to the right */
	Offset=(M+radius)*(N+2*radius)+radius;
	for(NdxCol=0;NdxCol<radius;NdxCol++)
	{		
		memcpy(Dest+Offset+NdxCol*(N+2*radius),Orig+(M-2-NdxCol)*N,
			N*sizeof(double));
	}

	/* Extend to the top */	
	for(NdxCol=0;NdxCol<M+2*radius;NdxCol++)
	{		
		for(NdxRow=0;NdxRow<radius;NdxRow++)
		{
			Dest[NdxRow+NdxCol*(N+2*radius)]=Dest[2*radius-NdxRow+NdxCol*(N+2*radius)];
		}		
	}

	/* Extend to the bottom */	
	for(NdxCol=0;NdxCol<M+2*radius;NdxCol++)
	{		
		for(NdxRow=radius+N,cnt=0;NdxRow<N+2*radius;NdxRow++,cnt++)
		{
			Dest[NdxRow+NdxCol*(N+2*radius)]=Dest[N+radius-2-cnt+NdxCol*(N+2*radius)];
		}		
	}

}
