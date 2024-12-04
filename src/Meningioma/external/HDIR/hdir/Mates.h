#ifndef _MATES_H

#define _MATES_H
 
/* Project orthogonally the column vector Vector on the vector basis Matrix,
and store the resulting projection vector in ResultVector  */

void Project(double * const Vector,double * const Matrix,
        double * const ResultVector,
        int Dimension,int NumBasisVectors);

/* Project orthogonally the column vector Vector on the vector basis Matrix,
store the resulting projection vector in ResultVector, and
the expression of the projection vector in basis coordinates in 
 ResultVectorInBase.  */

void ProjectExtra(double * const Vector,double * const Matrix,
        double * const ResultVector,
        double * const ResultVectorInBase,
        int Dimension,int NumBasisVectors);
        
/* Find the difference vector between two vectors*/
void Difference(double * const InputVector1,double * const InputVector2,
    double * const ResultVector,int Dimension);
    
/* Find the squared Euclidean norm of a vector */
void SquaredNorm(double * const Vector,double * const Result,int Dimension);

/* Find the eigenvalues and the eigenvectors of a real symmetric matrix,
sorted by the descending order of the eigenvalues */
void EigValVec(double * const Matrix,double * const EigenValues,
    double * const EigenVectors,int Dimension);

/* Orthonormalize a vector basis */
void Orthonormalize(double *InputBasis,double *OrthBasis,
    int Dimension,int NumBasisVectors);

/* Compute the inverse of a square matrix A  */
void Inverse(double *A,double *InverseA,int Dimension);

/* Compute the inverse of a square matrix A, and its L2-norm (or spectral norm).
Either the norm or the inverse are not computed if we set the corresponding
input parameter to NULL */
void InverseNorm(double *A,double *InverseA,double *NormA,double *NormInverseA,int NumRowsA,
    int NumColsA);

/* Product of an scalar by a matrix. It supports Matrix==Result */
void ScalarMatrixProduct(double Escalar,double *Matrix,double *Result,
    int NumRows,int NumCols);
        
/* Matrix sum. It supports that one of the operands is also the result*/
void MatrixSum(double *A,double *B,double *Result,int NumRows,int NumCols);

/* Matrix difference */
void MatrixDifference(double *A,double *B,double *Result,int NumRows,int NumCols);

/* Matrix product */
void MatrixProduct(double *A,double *B,double *Result,int NumRowsA,
    int NumColsA,int NumColsB);  

/* Find the diagonal of the product of A and B, that is,
 Result = diag ( A * B ), where Result is a vector. It is needed that 
 the number of rows of A is the same as the number of columns of B
 */
void DiagonalMatrixProduct(double *A,double *B,double *Result,
    int NumRowsA,int NumColsA);
    
/* Traspose of a matrix*/
void Traspose(double *A,double *TrasposeA,int NumRowsA,int NumColsA);  

/* Sum a diagonal matrix with a square matrix A. If Result==NULL,
the computation is performed on A */
void SumMatrixDiagonal(double *A,double *MatrixDiagonal,double *Result,int Dimension);

/* Sum a constant to all the diagonal elements of the square matrix A. If Result==NULL,
the computation is performed on A */
void SumDiagonalConstant(double *A,double Value,double *Result,int Dimension);

/* Extract the main diagonal of the square matrix A */
void ExtractDiagonal(double *A,double *DiagonalA,int Dimension);

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
    int NumRows,int NumCols);

#endif

