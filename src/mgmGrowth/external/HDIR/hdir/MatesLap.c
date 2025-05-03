#include "Mates.h"
#include "Debugging.h"
#include <string.h>
#include <float.h>
#include <stddef.h>
#include "mex.h"
#include "f2c.h"
#include "clapack.h"


 
/* Project orthogonally the column vector Vector on the vector basis Matrix,
and store the resulting projection vector in ResultVector  */

void Project(double * const Vector,double * const Matrix,
        double * const ResultVector,
        int Dimension,int NumBasisVectors)
{
    register double *Limit;
    register double *MyComponent;
    register double *MyElement;
    int ndx;
    register double Result;
    
    memset(ResultVector,0,sizeof(double)*Dimension);    
    MyElement=Matrix;
    for(ndx=0;ndx<NumBasisVectors;ndx++)
    {
        /* Find the dot product of the input vector and this basis vector */
        MyComponent=Vector;
        Limit=MyElement+Dimension;
        Result=0.0;
        while (MyElement<Limit)
        {
            Result+=(*MyElement)*(*MyComponent);
            MyElement++;
            MyComponent++;
        }    
        /* Find the contribution of this basis vector to the projection vector */
        MyComponent=ResultVector;
        MyElement-=Dimension;
        while (MyElement<Limit)
        {
            (*MyComponent)+=Result*(*MyElement);
            MyComponent++;
            MyElement++;
        }
    }          
}
  
    
/* Project orthogonally the column vector Vector on the vector basis Matrix,
store the resulting projection vector in ResultVector, and
the expression of the projection vector in basis coordinates in 
 ResultVectorInBase.  */

void ProjectExtra(double * const Vector,double * const Matrix,
        double * const ResultVector,
        double * const ResultVectorInBase,
        int Dimension,int NumBasisVectors)
{
    register double *Limit;
    register double *MyComponent;
    register double *MyElement;
    register double *MyResultEnBase;
    int ndx;
    register double Result;
    

    memset(ResultVector,0,sizeof(double)*Dimension);
    MyElement=Matrix;
    MyResultEnBase=ResultVectorInBase;
    for(ndx=0;ndx<NumBasisVectors;ndx++)
    {
        /* Find the dot product of the input vector and this basis vector */
        MyComponent=Vector;
        Limit=MyElement+Dimension;
        Result=0.0;
        while (MyElement<Limit)
        {
            Result+=(*MyElement)*(*MyComponent);
            MyElement++;
            MyComponent++;
        }    
        (*MyResultEnBase)=Result;
        MyResultEnBase++;
        /* Find the contribution of this basis vector to the projection vector */
        MyComponent=ResultVector;
        MyElement-=Dimension;
        while (MyElement<Limit)
        {
            (*MyComponent)+=Result*(*MyElement);
            MyComponent++;
            MyElement++;
        }
    }           
}  

/* Find the difference vector between two vectors*/
void Difference(double * const InputVector1,double * const InputVector2,
    double * const ResultVector,int Dimension)
{
    register double *MyComponentInput1;
    register double *MyComponentInput2;
    register double *MyComponentResult;
    register int ndx;
    
    MyComponentInput1=InputVector1;
    MyComponentInput2=InputVector2;
    MyComponentResult=ResultVector;
    for (ndx=0;ndx<Dimension;ndx++)
    {
        (*MyComponentResult)=(*MyComponentInput1)-
                (*MyComponentInput2);
        MyComponentInput1++;
        MyComponentInput2++;
        MyComponentResult++;                
    }      
}    
    
/* Find the squared Euclidean norm of a vector */
void SquaredNorm(double * const Vector,double * const Result,int Dimension)
{
    register double *MyComponent;
    register int ndx;
    register double MyResult;    
    
    MyComponent=Vector;
    MyResult=0.0;
    for (ndx=0;ndx<Dimension;ndx++)
    {
        MyResult+=(*MyComponent)*(*MyComponent);
        MyComponent++;                
    }    
    (*Result)=MyResult;  
}
    
/* Find the eigenvalues and the eigenvectors of a real symmetric matrix,
sorted by the descending order of the eigenvalues */
void EigValVec(double * const Matrix,double * const EigenValues,
    double * const EigenVectors,int Dimension)
{
    double *work;
    long int lwork;
    long int *iwork;
    long int liwork;
    long int info;
    int NdxCol;
    long int MyDimension;
    
    
    /* Copy Matrix to EigenVectors */
    memcpy(EigenVectors,Matrix,Dimension*Dimension*sizeof(double));
    
    MyDimension=Dimension;
    lwork=1 + 6*Dimension + 2*Dimension*Dimension;
    work=mxMalloc(lwork*sizeof(double));
    liwork=3 + 5*Dimension;
    iwork=mxMalloc(liwork*sizeof(long int));
    
    dsyevd_("V", "U", &MyDimension, EigenVectors, &MyDimension, EigenValues,
       work, &lwork, iwork, &liwork, &info); 
    
    
    
    if (info!=0)
        {
            PrintMatrix(EigenVectors,Dimension,Dimension);
            mexErrMsgTxt("LAPACK: Error while calling dsyevd in EigValVec\n");
        }     
    
    /* Sort by the descending order of the eigenvalues */
    
    /* Reverse the order of the eigenvectors using work as auxiliar */
    for(NdxCol=0;NdxCol<Dimension;NdxCol++)
    {
        memcpy(work+NdxCol*Dimension,EigenVectors+(Dimension-NdxCol-1)*Dimension,
                Dimension*sizeof(double));
    }    
    memcpy(EigenVectors,work,Dimension*Dimension*sizeof(double));
    
    /* Reverse the order of the eigenvalues using work as auxiliar */
    for(NdxCol=0;NdxCol<Dimension;NdxCol++)
    {
        work[NdxCol]=EigenValues[Dimension-NdxCol-1];
    }    
    memcpy(EigenValues,work,Dimension*sizeof(double));
        

    
    
    mxFree(work);    
    mxFree(iwork);
}

/* Orthonormalize a vector basis */
void Orthonormalize(double *InputBasis,double *OrthBasis,
    int Dimension,int NumBasisVectors)
{
  long int MyNumRows,MyNumCols;
  double *CopyA;
  double *S,*U,*VT;
  double *work;
  long int *iwork;
  long int lwork;
  long int info,NumSingularValues;
  int NdxCol;
  
  
  MyNumRows=Dimension;
  MyNumCols=NumBasisVectors;
  
  /* To store the singular values, the matrix U, and the traspose of the matrix V, respectively */
  if (MyNumRows>MyNumCols)
  {
      NumSingularValues=MyNumCols;
  }
  else
  {
      NumSingularValues=MyNumRows;
  }
          
  S=mxMalloc(NumSingularValues*sizeof(double));
  U=mxMalloc(MyNumRows*MyNumRows*sizeof(double));
  VT=mxMalloc(MyNumCols*MyNumCols*sizeof(double));
  
  /* Make a copy of A (the input vector basis) */
  CopyA=mxMalloc(MyNumRows*MyNumCols*sizeof(double));
  memcpy(CopyA,InputBasis,MyNumRows*MyNumCols*sizeof(double));
  
  /* Ask how much auxiliary memory we need */
  lwork=-1;
  work=mxMalloc(sizeof(double));
  iwork=mxMalloc(8*sizeof(long int));
  dgesdd_("A", &MyNumRows, &MyNumCols, CopyA, &MyNumRows, S, U, &MyNumRows, 
	VT, &MyNumCols, work, &lwork, iwork, &info);

  if (info!=0)
        {
            PrintMatrix(InputBasis,MyNumRows,MyNumCols);
            PrintMatrix(U,MyNumRows,MyNumRows);
            PrintMatrix(VT,MyNumCols,MyNumCols);
            PrintMatrix(S,1,NumSingularValues);
            mexPrintf("%d\n",lwork);
            mexErrMsgTxt("LAPACK: Error while calling dgesdd for the first time in Orthonormalize\n");
        } 
  
  lwork=(long int)work[0];
  mxFree(work);
  work=mxMalloc(lwork*sizeof(double));
  
  /* SVD computation*/ 
  dgesdd_("A", &MyNumRows, &MyNumCols, CopyA, &MyNumRows, S, U, &MyNumRows, 
	VT, &MyNumCols, work, &lwork, iwork, &info);
	
  if (info!=0)
        {
            PrintMatrix(InputBasis,MyNumRows,MyNumCols);
            PrintMatrix(U,MyNumRows,MyNumRows);
            PrintMatrix(VT,MyNumCols,MyNumCols);
            PrintMatrix(S,1,NumSingularValues);
            mexPrintf("%d\n",lwork);
            mexErrMsgTxt("LAPACK: Error while calling dgesdd for the second time in Orthonormalize\n");
        } 
  
  /* Given the SVD decomposition, A=U*S*V', copy the result to the output, 
  which are the NumSingularValues first columns of U */
  for(NdxCol=0;NdxCol<NumSingularValues;NdxCol++)
  {
      memcpy(OrthBasis,U+NdxCol*MyNumRows,MyNumRows*sizeof(double));
  }        
  
  mxFree(CopyA);
  mxFree(S);
  mxFree(U);
  mxFree(VT);
  mxFree(work);
  mxFree(iwork);

}    

/* Compute the inverse of a square matrix A  */
void Inverse(double *A,double *InverseA,int Dimension)
{
    double *work;
    long int lwork,NdxRow,NdxCol;
    integer info;
    long int *ipiv;
    long int MyDimension;
    
    MyDimension=Dimension;
        
    /* Copy the matrix A on InverseA*/
    memcpy(InverseA,A,Dimension*Dimension*sizeof(double));
    
    ipiv=mxMalloc(Dimension*sizeof(long int));
    
    /* Ask for the optimum size for the working matrix */
    lwork=-1;
    work=mxMalloc(sizeof(double));
    dsytrf_("U", &MyDimension, InverseA, &MyDimension, 
        ipiv, work, &lwork, &info);
    if (info!=0)
        {
            PrintMatrix(A,Dimension,Dimension);
            PrintMatrix(InverseA,Dimension,Dimension);
            mexPrintf("%d",lwork);
            mexErrMsgTxt("LAPACK: Error while calling dsytrf for the first time in Inverse\n");
        }   
    lwork=(int)work[0];
    mxFree(work);
    work=mxMalloc(lwork*sizeof(double));
    
    /* Decompose */
    dsytrf_("U", &MyDimension, InverseA, &MyDimension, 
        ipiv, work, &lwork, &info);
    mxFree(work);
	
	if (info!=0)
        {
            PrintMatrix(A,Dimension,Dimension);
            PrintMatrix(InverseA,Dimension,Dimension);
            mexPrintf("%d",lwork);
            mexErrMsgTxt("LAPACK: Error while calling dsytrf for the second time in Inverse\n");
        }     
       
    
	work=mxMalloc(Dimension*sizeof(double));
    dsytri_("U", &MyDimension, InverseA, &MyDimension, ipiv,
         work, &info);
	
	if (info!=0)
        {
            PrintMatrix(InverseA,Dimension,Dimension);
            mexErrMsgTxt("LAPACK: Error while calling dsytri in Inverse\n");
        }
 
    /* Copy the upper triangle on the lower triangle */
    for(NdxRow=0;NdxRow<Dimension;NdxRow++)
    {
        for(NdxCol=NdxRow+1;NdxCol<Dimension;NdxCol++)
        {
            InverseA[NdxCol+NdxRow*Dimension]=
                        InverseA[NdxRow+NdxCol*Dimension];
        }    
    }  
    
              
	mxFree(work);    
    mxFree(ipiv);
    
}    

/* Compute the inverse of a square matrix A, and its L2-norm (or spectral norm).
 Either the norm or the inverse are not computed if we set the corresponding
input parameter to NULL */
void InverseNorm(double *A,double *InverseA,double *NormA,double *NormInverseA,int NumRowsA,
    int NumColsA)
{
  long int MyNumRows,MyNumCols;
  double *CopyA,*V;
  double *S,*U,*VT;
  double *work,*ptrElement;
  long int *iwork;
  long int lwork;
  long int info,NumSingularValues;
  int i,j,k;
  double Sum;
  
  MyNumRows=NumRowsA;
  MyNumCols=NumColsA;
  
  /* To store the singular values, the matrix U, and the traspose of the matrix V, respectively */
  if (NumRowsA>NumColsA)
  {
      NumSingularValues=NumColsA;
  }
  else
  {
      NumSingularValues=NumRowsA;
  }
          
  S=mxMalloc(NumSingularValues*sizeof(double));
  U=mxMalloc(MyNumRows*MyNumRows*sizeof(double));
  VT=mxMalloc(MyNumCols*MyNumCols*sizeof(double));
  
  /* Make a copy of the matrix A */
  CopyA=mxMalloc(MyNumRows*MyNumCols*sizeof(double));
  memcpy(CopyA,A,MyNumRows*MyNumCols*sizeof(double));
  
  /* Ask how much auxiliary memory we need */
  lwork=-1;
  work=mxMalloc(sizeof(double));
  iwork=mxMalloc(8*sizeof(long int));
  dgesdd_("A", &MyNumRows, &MyNumCols, CopyA, &MyNumRows, S, U, &MyNumRows, 
	VT, &MyNumCols, work, &lwork, iwork, &info);

  if (info!=0)
        {
            PrintMatrix(A,NumRowsA,NumColsA);
            PrintMatrix(U,NumRowsA,NumRowsA);
            PrintMatrix(VT,NumColsA,NumColsA);
            PrintMatrix(S,1,NumSingularValues);
            mexPrintf("%d\n",lwork);
            mexErrMsgTxt("LAPACK: Error while calling dgesdd for the first time in InverseNorm\n");
        } 
  
  lwork=(long int)work[0];
  mxFree(work);
  work=mxMalloc(lwork*sizeof(double));
  
  /* SVD computation */ 
  dgesdd_("A", &MyNumRows, &MyNumCols, CopyA, &MyNumRows, S, U, &MyNumRows, 
	VT, &MyNumCols, work, &lwork, iwork, &info);
	
  if (info!=0)
        {
            PrintMatrix(A,NumRowsA,NumColsA);
            PrintMatrix(U,NumRowsA,NumRowsA);
            PrintMatrix(VT,NumColsA,NumColsA);
            PrintMatrix(S,1,NumSingularValues);
            mexPrintf("%d\n",lwork);
            mexErrMsgTxt("LAPACK: Error while calling dgesdd for the second time in InverseNorm\n");
        }   


  /* The norm of A is computed as the largest singular value */
  if (NormA!=NULL)
  {
      (*NormA)=S[0];
  }    
  
  
  /* The norm of inv(A) is computed as the inverse of the smallest singular value 
  of A */
  if (NormInverseA!=NULL)
  {
      (*NormInverseA)=1.0/S[NumSingularValues-1];
  }    
  
  /* Given the SVD decomposition, A=U*S*V', the inverse matrix is obtained as
  inv(A)=V*inv(S)*U', where inv(S) is computed by inverting each of the elements
  of S. This is because both V and U are orthogonal matrices, and this implies that
  inv(V)=V', inv(U)=U'. See for example 
  http://kwon3d.com/theory/jkinem/svd.html */
  
  if (InverseA!=NULL)
  {
      if (NumRowsA!=NumColsA)
      {
          mexErrMsgTxt("InverseNorm: Unable to find the inverse of a non square matrix\n");
      }   
       
      V=mxMalloc(NumRowsA*NumRowsA*sizeof(double));
      Traspose(VT,V,NumRowsA,NumRowsA);
      
      /* Compute V*inv(S) */
      for (j=0;j<NumRowsA;j++)
      {
          for (i=0;i<NumRowsA;i++)
          {
              V[i+j*NumRowsA]/=S[j];
          }
      }   
      
      /* Compute InverseA=(V*inv(S))*U' */
      ptrElement=InverseA;
      for (j=0;j<NumRowsA;j++)
      {
          for (i=0;i<NumRowsA;i++)
          {
              Sum=0.0;
              for(k=0;k<NumRowsA;k++)
              {
                  /* Note that we have U, not U' */
                  Sum+=V[i+k*NumRowsA]*U[j+k*NumRowsA];
              }    
              (*ptrElement)=Sum;
              ptrElement++;
          }
      }  
      mxFree(V);
  }     
  
  mxFree(CopyA);
  mxFree(S);
  mxFree(U);
  mxFree(VT);
  mxFree(work);
  mxFree(iwork);
  
}

/* Product of an scalar by a matrix. It supports Matrix==Result */
void ScalarMatrixProduct(double Escalar,double *Matrix,double *Result,
    int NumRows,int NumCols)
{
    register double Factor;
    register double *ptr;
    register double *ptrres;
    register int ndx;
    register int NumElements;
    
    ptrres=Result;
    ptr=Matrix;
    Factor=Escalar;
    NumElements=NumRows*NumCols;
    for(ndx=0;ndx<NumElements;ndx++)
    {
        (*ptrres)=Factor*(*ptr);
        ptrres++;
        ptr++;
    }    
    
}    
/* Matrix sum. It supports that one of the operands is also the result*/
void MatrixSum(double *A,double *B,double *Result,int NumRows,int NumCols)
{
    register double *ptra;
    register double *ptrb;
    register double *ptrres;
    register int ndx;
    register int NumElements;
    
    ptra=A;
    ptrb=B;
    ptrres=Result;
    NumElements=NumRows*NumCols;
    for(ndx=0;ndx<NumElements;ndx++)
    {
        (*ptrres)=(*ptra)+(*ptrb);
        ptrres++;
        ptra++;
        ptrb++;
    }    
}

/* Matrix difference */
void MatrixDifference(double *A,double *B,double *Result,int NumRows,int NumCols)
{
    register double *ptra;
    register double *ptrb;
    register double *ptrres;
    register int ndx;
    register int NumElements;
    
    ptra=A;
    ptrb=B;
    ptrres=Result;
    NumElements=NumRows*NumCols;
    for(ndx=0;ndx<NumElements;ndx++)
    {
        (*ptrres)=(*ptra)-(*ptrb);
        ptrres++;
        ptra++;
        ptrb++;
    }    
}

/* Matrix product */
void MatrixProduct(double *A,double *B,double *Result,int NumRowsA,
    int NumColsA,int NumColsB)
{
    register double *ptra;
    register double *ptrb;
    register double *ptrres;
    register int i;
    register int j;
    register int k;
    register double Sum;
    
    ptrres=Result;
    for(j=0;j<NumColsB;j++)
    {
        for(i=0;i<NumRowsA;i++)
        {
            Sum=0.0;
            ptrb=B+NumColsA*j;
            ptra=A+i;
            for(k=0;k<NumColsA;k++)
            {
                Sum+=(*ptra)*(*ptrb);
                ptra+=NumRowsA;
                ptrb++;
            }    
            (*ptrres)=Sum;
            ptrres++;
        }
    }            
}   

/* Find the diagonal of the product of A and B, that is,
 Result = diag ( A * B ), where Result is a vector. It is needed that 
 the number of rows of A is the same as the number of columns of B
 */
void DiagonalMatrixProduct(double *A,double *B,double *Result,
    int NumRowsA,int NumColsA)
{
    register double *ptra;
    register double *ptrb;
    register double *ptrres;
    register int i;
    register int k;
    register double Sum;
    
    ptrres=Result;
    for(i=0;i<NumRowsA;i++)
    {
        Sum=0.0;
        ptrb=B+NumColsA*i;
        ptra=A+i;
        for(k=0;k<NumColsA;k++)
        {
            Sum+=(*ptra)*(*ptrb);
            ptra+=NumRowsA;
            ptrb++;
        }    
        (*ptrres)=Sum;
        ptrres++;
    }
         
}   

/* Traspose of a matrix*/
void Traspose(double *A,double *TrasposeA,int NumRowsA,int NumColsA)
{
    register int NdxRow;
    register int NdxCol;
    register double *ptrA;
    
    ptrA=A;
    for(NdxCol=0;NdxCol<NumColsA;NdxCol++)
    {
        for(NdxRow=0;NdxRow<NumRowsA;NdxRow++)
        {
            (*(TrasposeA+NdxRow*NumColsA+NdxCol))=(*ptrA);
            ptrA++;
        }
    }        
}    

/* Sum a diagonal matrix with a square matrix A. If Result==NULL,
the computation is performed on A */
void SumMatrixDiagonal(double *A,double *MatrixDiagonal,double *Result,int Dimension)
{
    register int NdxElement;
    register double *ptrDiagonal;
    register double *ptrResult;
    
    /* Copy the matrix A in the output, if necessary */
    if (Result!=NULL)
    {
        memcpy(Result,A,sizeof(double)*Dimension*Dimension);
    }
    else
    {
        Result=A;
    }        
    
    /* Add the diagonal matrix to the result */
    ptrDiagonal=MatrixDiagonal;
    ptrResult=Result;
    for(NdxElement=0;NdxElement<Dimension;NdxElement++)
    {
        (*ptrResult)+=(*ptrDiagonal);
        ptrResult+=(Dimension+1);
        ptrDiagonal++;
    }  
}   

/* Sum a constant to all the diagonal elements of the square matrix A. If Result==NULL,
the computation is performed on A */
void SumDiagonalConstant(double *A,double Value,double *Result,int Dimension)
{
    register int NdxElement;
    register double *ptrResult;
    
   /* Copy the matrix A in the output, if necessary */
    if (Result!=NULL)
    {
        memcpy(Result,A,sizeof(double)*Dimension*Dimension);
    }
    else
    {
        Result=A;
    }
    
    /* Add the constant to the diagonal of the output */
    ptrResult=Result;
    for(NdxElement=0;NdxElement<Dimension;NdxElement++)
    {
        (*ptrResult)+=Value;
        ptrResult+=(Dimension+1);
    }  
}    

/* Extract the main diagonal of the square matrix A */
void ExtractDiagonal(double *A,double *DiagonalA,int Dimension)
{
    register int NdxElement;
    register double *ptrResult;
    register double *ptrA;
    
    ptrResult=DiagonalA;
    ptrA=A;
    for(NdxElement=0;NdxElement<Dimension;NdxElement++)
    {
        (*ptrResult)=(*ptrA);
        ptrA+=(Dimension+1);
        ptrResult++;
    }  
}

/* Compute the "economy size" Singular Value Decomposition of matrix A. 
The sizes of the matrices are:
A is NumRows x NumCols
S is min(NumRows,NumCols) x 1
U is NumRows x min(NumRows,NumCols)
V is NumCols x min(NumRows,NumCols)
We have:
A = U * diag(S) * V'
*/
void EconomySVD(double *A,double *S,double *U,double *V,
    int NumRows,int NumCols)
{
  long int MyNumRows,MyNumCols;
  double *CopyA;
  double *VT;
  double *work,*ptrElement;
  long int *iwork;
  long int lwork;
  long int liwork,info,NumSingularValues;
  
  
  MyNumRows=NumRows;
  MyNumCols=NumCols;
  
  /* To store the singular values, the matrix U, the matrix V, and the traspose of the matrix V, respectively */
  if (MyNumRows>MyNumCols)
  {
      NumSingularValues=MyNumCols;
  }
  else
  {
      NumSingularValues=MyNumRows;
  }
            
  VT=mxMalloc(MyNumCols*MyNumCols*sizeof(double));
  
  /* Make a copy of A (the input matrix) */
  CopyA=mxMalloc(MyNumRows*MyNumCols*sizeof(double));
  memcpy(CopyA,A,MyNumRows*MyNumCols*sizeof(double));
  
  /* Ask how much auxiliary memory we need */
  lwork=-1;
  work=mxMalloc(sizeof(double));
  iwork=mxMalloc(8*sizeof(long int));
  dgesdd_("S", &MyNumRows, &MyNumCols, CopyA, &MyNumRows, S, U, &MyNumRows, 
	VT, &MyNumCols, work, &lwork, iwork, &info);

  if (info!=0)
        {
            PrintMatrix(A,MyNumRows,MyNumCols);
            PrintMatrix(U,MyNumRows,MyNumRows);
            PrintMatrix(VT,MyNumCols,MyNumCols);
            PrintMatrix(S,1,NumSingularValues);
            mexPrintf("%d\n",lwork);
            mexErrMsgTxt("LAPACK: Error while calling dgesdd for the first time in EconomySVD\n");
        } 
  
  lwork=(long int)work[0];
  mxFree(work);
  work=mxMalloc(lwork*sizeof(double));
  
  /* SVD computation*/ 
  dgesdd_("S", &MyNumRows, &MyNumCols, CopyA, &MyNumRows, S, U, &MyNumRows, 
	VT, &MyNumCols, work, &lwork, iwork, &info);
	
  if (info!=0)
        {
            PrintMatrix(A,MyNumRows,MyNumCols);
            PrintMatrix(U,MyNumRows,MyNumRows);
            PrintMatrix(VT,MyNumCols,MyNumCols);
            PrintMatrix(S,1,NumSingularValues);
            mexPrintf("%d\n",lwork);
            mexErrMsgTxt("LAPACK: Error while calling dgesdd for the second time in EconomySVD\n");
        } 
  
  /* Given the SVD decomposition, A=U*S*V', obtain V from V' */
  Traspose(VT,V,NumSingularValues,NumCols);      
  
  mxFree(CopyA);  
  mxFree(VT);
  mxFree(work);
  mxFree(iwork);

}