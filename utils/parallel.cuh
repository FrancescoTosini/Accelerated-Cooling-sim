#ifndef PARALLEL_CUH
#define PARALLEL_CUH

#include <stdio.h>
#include <stdlib.h>

#include "constants.h"
#include "cuda_runtime.h"
#include "cusolverDn.h"
#include "factor.h"
#include "helper_cuda.h"
#include "util.cuh"

// Global GPU variables
// extern __device__ int maxRow, maxCol;
// extern __device__ double temp;
// extern __device__ double globalV;
// extern __device__ double rMax;
// extern __device__ double rMin;
// extern __device__ double rMean;
// extern __device__ double rStd;

// Parameters to compute point sensitiveness - values read from input file
extern double Sreal, Simag, Rreal, Rimag;
extern int MaxIters;
extern int TimeSteps;          // Evolution time steps
extern double *MeasuredValues; // 2-D array - (NumInputValues,3) - Values read in
                               // input file
extern int NumInputValues;     // Number of values read in input file
extern double *TheorSlope;     // 2-D array - Theoretical value distribution
extern int TSlopeLength;       // TheorSlope grid dimensions
extern int *FieldWeight;       // 2-D array - (Xdots,Ydots) - Degree of sensitiveness to
                               // perturbing field
extern double *FieldCoord;     // 3-D array - X, Y coordinates in field
extern double *FieldValues;    // 3-D array - X, Y coordinates in field

void gpuInitGrid(char *InputFile);
void gpuFieldDistribution();
void gpuGridDef(double x0, double x1, double y0, double y1, int N, double *Pts);
__global__ void gpuGridDefKernel(double x0, double y0, double dx, double dy, double *Pts, int Nm1, int len, int TSlopeLength);
void gpuEqsDef(double x0, double x1, double y0, double y1, int N, int LA, double *A, double *Rhs, double *Pts);
__global__ void gpuEqsDefKernel(double x0, double x1, double y0, double y1, int Nm1, double dx, double dy, int LA, double *A, double *Rhs, double *Pts, int TSlopeLength);
__device__ double gpuSolution(double x, double y);

int gpuLinEquSolve(double *a, int n, double *b);
__global__ void gpuLinEquSolveKernel(double *maxima, int *maxIndex, double *a, double *b, int *indrow, int *indcol, int *ipiv, int n);

void gpuSensiblePoints(double Ir, double Ii, double Sr, double Si, int MaxIt);
__global__ void gpuSensiblePointsKernel(double Ir, double Ii, double Xinc, double Yinc, int MaxIt, double *FieldCoord, int *FieldWeight);

void gpuFieldInit();
double gpuNearestValue(double xc, double yc, int ld, double *Values);
__global__ void gpuNearestValueKernel(double xc, double yc, double *Values, double *dist, int *mask, double *partialResult, int n);

void MinMaxIntVal(int *Values, int len);
__global__ void MinMaxIntValKernel(int *Values, int len, int *tmpMin, int *tmpMax);
void gpuFieldPoints(double Diff);
__global__ void gpuFieldPointsKernel(double *FieldCoord, int *FieldWeigth, double *FieldValues, double Diff, int TSlopeLength, double *TheorSlope);
__device__ double deviceNearestValue(double xc, double yc, int ld, double *Values);

void gpuCooling(int steps);
void gpuStatistics(int s1, int s2, double *rdata, double *tmp, int step);
__global__ void gpuStatisticsKernel(double *tmpMin, double *tmpMax, double *tmpMean, double *tmpStd, double *Values, int len, int reduceLayer);
void gpuUpdate(int xdots, int ydots, double *u1, double *u2);
__global__ void gpuUpdateKernel(int xdots, int ydots, double *u1, double *u2, double CX, double CY, double dgx, double dgy);
int LinEquSolve_ACC(double *d_A,  // dense coefficient matrix (on device)
                    int n,        // size (square)
                    double *d_b); // A*x = b  (on device)

#endif