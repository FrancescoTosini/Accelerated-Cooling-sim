#ifndef __CUDACC__ 
    #define __CUDACC__
#endif

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "util.cu"

#define index2D(i,j,LD1) i + ((j)*LD1)    // element position in 2-D arrays
#define index3D(i,j,k,LD1,LD2) i + ((j)*LD1) + ((k)*LD1*LD2)   // element position in 3-D arrays

#define Xdots 1400   // Plate grid resolution in 2 dimensions
#define Ydots 1400   // May be changed to 1000x1000

// Parameters to compute point sensitiveness - values read from input file
double Sreal, Simag, Rreal, Rimag;
int MaxIters;
int TimeSteps;   // Evolution time steps

double* MeasuredValues;    // 2-D array - (NumInputValues,3) - Values read in input file
int NumInputValues;        // Number of values read in input file
double* TheorSlope;        // 2-D array - Theoretical value distribution
int TSlopeLength;          // TheorSlope grid dimensions
int* FieldWeight;          // 2-D array - (Xdots,Ydots) - Degree of sensitiveness to perturbing field 
double* FieldCoord;        // 3-D array - X, Y coordinates in field
double* FieldValues;       // 3-D array - X, Y coordinates in field

// Global GPU variables

__device__ int maxRow, maxCol;
__device__ double temp;
__device__ double globalV;
__device__ double rMax;
__device__ double rMin;
__device__ double rMean;
__device__ double rStd;

/* CODE */

// Functions  prototypes

    void InitGrid(char* InputFile);
    int LinEquSolve(double* a, int n, double* b);
    void EqsDef(double x0, double x1, double y0, double y1, int N, int LA, double* A, double* Rhs, double* Pts);
    double Solution(double x, double y);
    void FieldDistribution();
    void GridDef(double x0, double x1, double y0, double y1, int N, double* Pts);
    void SensiblePoints(double Ir, double Ii, double Sr, double Si, int MaxIt);
    void FieldInit();
    double NearestValue(double xc, double yc, int ld, double* Values);
    void FieldPoints(double Diff);
    void Cooling(int steps);
    void RealData2ppm(int s1, int s2, double* rdata, double* vmin, double* vmax, char* name);
    void Statistics(int s1, int s2, double* rdata, int s);
    void Update(int xdots, int ydots, double* u1, double* u2);

// Accelerated functions

    void gpuInitGrid(char* InputFile);
    void gpuFieldDistribution();
    void gpuGridDef(double x0, double x1, double y0, double y1, int N, double* Pts);
    __global__ void gpuGridDefKernel(double x0, double y0, double dx, double dy, double* Pts, int Nm1, int len, int TSlopeLength);
    void gpuEqsDef(double x0, double x1, double y0, double y1, int N, int LA, double* A, double* Rhs, double* Pts);
    __global__ void gpuEqsDefKernel(double x0, double x1, double y0, double y1, int Nm1, double dx, double dy, int LA, double* A, double* Rhs, double* Pts, int TSlopeLength);
    __device__ double gpuSolution(double x, double y);

    int gpuLinEquSolve(double* a, int n, double* b);
    __global__ void gpuLinEquSolveKernel(double* maxima, int* maxIndex, double* a, double* b, int* indrow, int* indcol, int* ipiv, int n);

    void gpuSensiblePoints(double Ir, double Ii, double Sr, double Si, int MaxIt);
    __global__ void gpuSensiblePointsKernel(double Ir, double Ii, double Xinc, double Yinc, int MaxIt, double* FieldCoord, int* FieldWeight);

    void gpuFieldInit();
    double gpuNearestValue(double xc, double yc, int ld, double* Values);
    __global__ void gpuNearestValueKernel(double xc, double yc, double* Values, double* dist, int* mask, double* partialResult, int n);

    void MinMaxIntVal(int* Values, int len);
    __global__ void MinMaxIntValKernel(int* Values, int len, int* tmpMin, int* tmpMax);
    void gpuFieldPoints(double Diff);
    __global__ void gpuFieldPointsKernel(double* FieldCoord, int* FieldWeigth, double* FieldValues, double Diff, int TSlopeLength, double* TheorSlope);
    __device__ double deviceNearestValue(double xc, double yc, int ld, double* Values);

    void gpuCooling(int steps);
    void gpuStatistics(int s1, int s2, double* rdata, double* tmp, int step);
    __global__ void gpuStatisticsKernel(double* tmpMin, double* tmpMax, double* tmpMean, double* tmpStd, double* Values, int len, int reduceLayer);
    void gpuUpdate(int xdots, int ydots, double* u1, double* u2);
    __global__ void gpuUpdateKernel(int xdots, int ydots, double* u1, double* u2, double CX, double CY, double dgx, double dgy);

int main(int argc, char* argv[])
{
    clock_t t0, t1, p0, p1;

    t0 = clock();
    printf(">> Starting\n");

    // Read input file
    p0 = clock();
    gpuInitGrid("Cooling.inp");
    p1 = clock();
    fprintf(stdout, ">> InitGrid ended in %lf seconds\n", (double)(p1 - p0)/CLOCKS_PER_SEC);

    // TheorSlope(TSlopeLength,3)
    p0 = clock();
    gpuFieldDistribution();
    p1 = clock();
    fprintf(stdout, ">> FieldDistribution ended in %lf seconds\n", (double)(p1 - p0) / CLOCKS_PER_SEC);

    // FieldCoord(Xdots,Ydots,2), FieldWeight(Xdots,Ydots)
    p0 = clock();
    gpuSensiblePoints(Sreal, Simag, Rreal, Rimag, MaxIters);
    p1 = clock();
    fprintf(stdout, ">> SensiblePoints ended in %lf seconds\n", (double)(p1 - p0) / CLOCKS_PER_SEC);

    // MeasuredValues(:,3), FieldWeight(Xdots,Ydots) -> FieldValues(Xdots,Ydots,2)
    p0 = clock();
    gpuFieldInit();
    p1 = clock();
    fprintf(stdout, ">> FieldInit ended in %lf seconds\n", (double)(p1 - p0) / CLOCKS_PER_SEC);

    // FieldValues(Xdots,Ydots,2)
    p0 = clock();
    gpuCooling(TimeSteps);
    p1 = clock();
    fprintf(stdout, ">> Cooling ended in %lf seconds\n", (double)(p1 - p0) / CLOCKS_PER_SEC);

    t1 = clock();
    fprintf(stdout, ">> Computations ended in %lf seconds\n", (double)(t1 - t0) / CLOCKS_PER_SEC);
    
    // End Program

    free(MeasuredValues);
    cudaFree(FieldWeight);
    cudaFree(FieldCoord);
    cudaFree(TheorSlope);
    cudaFree(FieldValues);

    return 0;

    /*time_t t0, t1, p0, p1;

    time(&t0);
    //fprintf(stdout, ">> Starting %s at: %s", argv[0], asctime(localtime(&t0)));
    printf(">> Starting\n");

    // Read input file
    time(&p0);
    InitGrid("Cooling.inp");
    time(&p1);
    fprintf(stdout, ">> InitGrid ended in %lf seconds\n", difftime(p1, p0));

    // TheorSlope(TSlopeLength,3)
    time(&p0);
    FieldDistribution();
    time(&p1);
    fprintf(stdout, ">> FieldDistribution ended in %lf seconds\n", difftime(p1, p0));

    // FieldCoord(Xdots,Ydots,2), FieldWeight(Xdots,Ydots)
    time(&p0);
    SensiblePoints(Sreal, Simag, Rreal, Rimag, MaxIters);
    time(&p1);
    fprintf(stdout, ">> SensiblePoints ended in %lf seconds\n", difftime(p1, p0));

    // MeasuredValues(:,3), FieldWeight(Xdots,Ydots) -> FieldValues(Xdots,Ydots,2)
    time(&p0);
    FieldInit();
    time(&p1);
    fprintf(stdout, ">> FieldInit ended in %lf seconds\n", difftime(p1, p0));

    // FieldValues(Xdots,Ydots,2)
    time(&p0);
    Cooling(TimeSteps);
    time(&p1);
    fprintf(stdout, ">> Cooling ended in %lf seconds\n", difftime(p1, p0));

    time(&t1);
    fprintf(stdout, ">> Ending at: %s", asctime(localtime(&t1)));
    fprintf(stdout, ">> Computations ended in %lf seconds\n", difftime(t1, t0));

    // End Program

    free(MeasuredValues);
    free(FieldWeight);
    free(FieldCoord);
    free(TheorSlope);
    free(FieldValues);

	return 0;*/
}

/* FUNCTIONS */

void InitGrid(char* InputFile)
{
    /* Output:
    !  MeasuredValues(:,3) - values read from input file
    !  Initialization of FieldWeight(Xdots,Ydots) and FieldCoord(Xdots,Ydots,2)
    */

    int valrows, st;
    char filerow[80];
    FILE* inpunit;

    fprintf(stdout, ">> Initializing grid ...\n");

    inpunit = fopen(InputFile, "r");
    if (!inpunit) 
    {
        fprintf(stderr, "(Error) >>> Cannot access file %s\n", InputFile);
        exit(-1);
    }

    // Read measured values
    NumInputValues = 0;
    valrows = 0;
    while (1)
    {
        st = readrow(filerow, 80, inpunit);
        if (filerow[0] == '#') continue;
        if (NumInputValues <= 0) 
        {
            if (sscanf(filerow, "  %d", &NumInputValues) < 1) 
            {
                if (NumInputValues <= 0) 
                {
                    fprintf(stderr, "(Error) >> there seems to be %d input values...\n", NumInputValues);
                    exit(-1);
                }
            }
            else 
            {
                MeasuredValues = (double*)malloc(sizeof(double) * NumInputValues * 3);
                if (MeasuredValues == NULL) 
                {
                    fprintf(stderr, "(Error) >> Cannot allocate MeasuredValues[%d,3] :(\n", NumInputValues);
                    exit(-1);
                }
            }
        }
        else 
        {
            if (sscanf(filerow, "%lf %lf %lf",
                &MeasuredValues[index2D(valrows, 0, NumInputValues)],  // X coord
                &MeasuredValues[index2D(valrows, 1, NumInputValues)],  // Y coord
                &MeasuredValues[index2D(valrows, 2, NumInputValues)])  // Measured value
                < 3) 
            {
                fprintf(stderr, "(Error) >>> something went wrong while reading MeasuredValues(%d,*)", valrows);
                exit(-1);
            }
            valrows++;
            if (valrows >= NumInputValues) break;
        }
    }

    /* Create and initialize FieldWeight */
    FieldWeight = (int*)malloc(sizeof(int) * Xdots * Ydots);
    if (FieldWeight == NULL) 
    {
        fprintf(stderr, "(Error) >> Cannot allocate FieldWeight[%d,%d]\n", Xdots, Ydots);
        exit(-1);
    }
    SetIntValue(FieldWeight, Xdots * Ydots, 0); // OPP: you can use calloc?

    /* Create and initialize FieldCoord */
    FieldCoord = (double*)malloc(sizeof(double) * Xdots * Ydots * 2);
    if (FieldCoord == NULL) 
    {
        fprintf(stderr, "(Error) >> Cannot allocate FieldCoord[%d,%d,2]\n", Xdots, Ydots);
        exit(-1);
    }
    SetDoubleValue(FieldCoord, Xdots * Ydots * 2, (double)0); // OPP: you can use calloc?

    /* Now read Sreal, Simag, Rreal, Rimag */
    Sreal = Simag = Rreal = Rimag = 0.0;
    while (1)
    {
        if (readrow(filerow, 80, inpunit) < 1) 
        {
            fprintf(stderr, "(Error) >> Cannot read Sreal from input file.\n");
            exit(-1);
        }
        if (filerow[0] == '#') continue;
        if (sscanf(filerow, "%lf", &Sreal) < 1) 
        {
            fprintf(stderr, "(Error) >> Cannot read Sreal from string.\n");
            exit(-1);
        }
        if (fscanf(inpunit, "%lf", &Simag) < 1) 
        {
            fprintf(stderr, "(Error) >> Cannot read Simag from input file.\n");
            exit(-1);
        }
        if (fscanf(inpunit, "%lf", &Rreal) < 1) 
        {
            fprintf(stderr, "(Error) >> Cannot read Rreal from input file.\n");
            exit(-1);
        }
        if (fscanf(inpunit, "%lf", &Rimag) < 1) 
        {
            fprintf(stderr, "(Error) >> Cannot read Rimag from input file.\n");
            exit(-1);
        }
        break;
    }

    /* Now read MaxIters */
    MaxIters = 0;
    while (1)
    {
        if (readrow(filerow, 80, inpunit) < 1) 
        {
            fprintf(stderr, "(Error) >> Cannot read MaxIters from input file.\n");
            exit(-1);
        }
        if (filerow[0] == '#' || rowlen(filerow) < 1) continue;
        if (sscanf(filerow, "%d", &MaxIters) < 1) 
        {
            fprintf(stderr, "(Error) >> Cannot read MaxIters from string.\n");
            exit(-1);
        }
        break;
    }

    /* Now read TimeSteps */
    TimeSteps = 0;
    while (1)
    {
        if (readrow(filerow, 80, inpunit) < 1) 
        {
            fprintf(stderr, "(Error) >> Cannot read MaxIters from input file.\n");
            exit(-1);
        }
        if (filerow[0] == '#' || rowlen(filerow) < 1) continue;
        if (sscanf(filerow, "%d", &TimeSteps) < 1) 
        {
            fprintf(stderr, "(Error) >> Cannot read TimeSteps from string.\n");
            exit(-1);
        }
        break;
    }

    fclose(inpunit);
    return;
}

void gpuInitGrid(char* InputFile)
{
    /* Output:
    !  MeasuredValues(:,3) - values read from input file
    !  Initialization of FieldWeight(Xdots,Ydots) and FieldCoord(Xdots,Ydots,2)
    */

    int valrows, st;
    char filerow[80];
    FILE* inpunit;

    cudaError_t err;

    fprintf(stdout, "(NO CPU) >> Initializing grid ...\n");

    inpunit = fopen(InputFile, "r");
    if (!inpunit)
    {
        fprintf(stderr, "(Error) >>> Cannot access file %s\n", InputFile);
        exit(-1);
    }

    // Read measured values
    NumInputValues = 0;
    valrows = 0;
    while (1)
    {
        st = readrow(filerow, 80, inpunit);
        if (filerow[0] == '#') continue;
        if (NumInputValues <= 0)
        {
            if (sscanf(filerow, "  %d", &NumInputValues) < 1)
            {
                if (NumInputValues <= 0)
                {
                    fprintf(stderr, "(Error) >> there seems to be %d input values...\n", NumInputValues);
                    exit(-1);
                }
            }
            else
            {
                MeasuredValues = (double*)malloc(sizeof(double) * NumInputValues * 3);
                if (MeasuredValues == NULL)
                {
                    fprintf(stderr, "(Error) >> Cannot allocate tmpMeasuredValues[%d,3] :(\n", NumInputValues);
                    exit(-1);
                }
            }
        }
        else
        {
            if (sscanf(filerow, "%lf %lf %lf",
                &MeasuredValues[index2D(valrows, 0, NumInputValues)],  // X coord
                &MeasuredValues[index2D(valrows, 1, NumInputValues)],  // Y coord
                &MeasuredValues[index2D(valrows, 2, NumInputValues)])  // Measured value
                < 3)
            {
                fprintf(stderr, "(Error) >>> something went wrong while reading MeasuredValues(%d,*)", valrows);
                exit(-1);
            }
            valrows++;
            if (valrows >= NumInputValues) break;
        }
    }

    /* Create and initialize FieldWeight */
    err = cudaMalloc(&FieldWeight, sizeof(int) * Xdots * Ydots);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "(Error) >> Cannot allocate FieldWeight[%d,%d] on GPU\n", Xdots, Ydots);
        exit(-1);
    }
    cudaMemset(FieldWeight, 0, sizeof(int) * Xdots * Ydots);

    /* Create and initialize FieldCoord */
    err = cudaMalloc(&FieldCoord, sizeof(double) * Xdots * Ydots * 2);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "(Error) >> Cannot allocate FieldCoord[%d,%d,2] on GPU\n", Xdots, Ydots);
        exit(-1);
    }
    cudaMemset(FieldCoord, 0, sizeof(double) * Xdots * Ydots * 2);

    /* Now read Sreal, Simag, Rreal, Rimag */
    Sreal = Simag = Rreal = Rimag = 0.0;
    while (1)
    {
        if (readrow(filerow, 80, inpunit) < 1)
        {
            fprintf(stderr, "(Error) >> Cannot read Sreal from input file.\n");
            exit(-1);
        }
        if (filerow[0] == '#') continue;
        if (sscanf(filerow, "%lf", &Sreal) < 1)
        {
            fprintf(stderr, "(Error) >> Cannot read Sreal from string.\n");
            exit(-1);
        }
        if (fscanf(inpunit, "%lf", &Simag) < 1)
        {
            fprintf(stderr, "(Error) >> Cannot read Simag from input file.\n");
            exit(-1);
        }
        if (fscanf(inpunit, "%lf", &Rreal) < 1)
        {
            fprintf(stderr, "(Error) >> Cannot read Rreal from input file.\n");
            exit(-1);
        }
        if (fscanf(inpunit, "%lf", &Rimag) < 1)
        {
            fprintf(stderr, "(Error) >> Cannot read Rimag from input file.\n");
            exit(-1);
        }
        break;
    }

    /* Now read MaxIters */
    MaxIters = 0;
    while (1)
    {
        if (readrow(filerow, 80, inpunit) < 1)
        {
            fprintf(stderr, "(Error) >> Cannot read MaxIters from input file.\n");
            exit(-1);
        }
        if (filerow[0] == '#' || rowlen(filerow) < 1) continue;
        if (sscanf(filerow, "%d", &MaxIters) < 1)
        {
            fprintf(stderr, "(Error) >> Cannot read MaxIters from string.\n");
            exit(-1);
        }
        break;
    }

    /* Now read TimeSteps */
    TimeSteps = 0;
    while (1)
    {
        if (readrow(filerow, 80, inpunit) < 1)
        {
            fprintf(stderr, "(Error) >> Cannot read MaxIters from input file.\n");
            exit(-1);
        }
        if (filerow[0] == '#' || rowlen(filerow) < 1) continue;
        if (sscanf(filerow, "%d", &TimeSteps) < 1)
        {
            fprintf(stderr, "(Error) >> Cannot read TimeSteps from string.\n");
            exit(-1);
        }
        break;
    }

    fclose(inpunit);
    return;
}

void FieldDistribution()
{
    /*
    !  Compute theoretical value distribution of the perturbing field
    !  Output: TheorSlope(TSlopeLength,3) - theoretical field distribution function
    */
    double *CoeffMatrix, *B;
    double x0, y0, x1, y1;

    int M, Mm1, N, Nm1, LA;
    int i, rc;

    fprintf(stdout, "\t>> Computing theoretical perturbing field...\n");

    x0 = Sreal; 
    y0 = Simag; 
    x1 = x0 + Rreal; 
    y1 = y0 + Rimag;

    // How many intervals? It should be safe to use SQRT(Xdots)
    M = sqrt((double)Xdots);
    N = sqrt((double)Ydots);

    Nm1 = N - 1;  // Grid points minus boundary
    Mm1 = M - 1;  // Grid points minus boundary

    LA = Mm1 * Nm1; // unknown points
    TSlopeLength = LA;

    /* Allocate CoeffMatrix */
    CoeffMatrix = (double*)malloc(sizeof(double) * LA * LA);
    if (CoeffMatrix == NULL) 
    {
        fprintf(stderr, "(Error) >> Cannot allocate CoeffMatrix[%d,%d]\n", LA, LA);
        exit(-1);
    }

    /* Allocate TheorSlope */
    TheorSlope = (double*)malloc(sizeof(double) * TSlopeLength * 3);
    if (TheorSlope == NULL) 
    {
        fprintf(stderr, "(Error) >> Cannot allocate TheorSlope[%d,3]\n", TSlopeLength);
        exit(-1);
    }

    /* Allocate B */
    B = (double*)malloc(sizeof(double) * LA);
    if (B == NULL) 
    {
        fprintf(stderr, "(Error) >> Cannot allocate B[%d]\n", LA);
        exit(-1);
    }

    GridDef(x0, x1, y0, y1, N, TheorSlope);

    EqsDef(x0, x1, y0, y1, N, LA, CoeffMatrix, B, TheorSlope);

    rc = LinEquSolve(CoeffMatrix, LA, B);
    if (rc != 0) exit(-1);

    for (i = 0; i < LA; i++) TheorSlope[index2D(i, 2, TSlopeLength)] = B[i]; // OPP: why not use memcpy?

    free(CoeffMatrix);
    free(B);

    return;
}

void gpuFieldDistribution()
{
    /*
    !  Compute theoretical value distribution of the perturbing field
    !  Output: TheorSlope(TSlopeLength,3) - theoretical field distribution function
    */
    double* CoeffMatrix, * B;
    double x0, y0, x1, y1;

    int M, Mm1, N, Nm1, LA;
    int i, rc;

    cudaError_t err;

    fprintf(stdout, "\t>> Computing theoretical perturbing field...\n");

    x0 = Sreal;
    y0 = Simag;
    x1 = x0 + Rreal;
    y1 = y0 + Rimag;

    // How many intervals? It should be safe to use SQRT(Xdots)
    M = sqrt((double)Xdots);
    N = sqrt((double)Ydots);

    Nm1 = N - 1;  // Grid points minus boundary
    Mm1 = M - 1;  // Grid points minus boundary

    LA = Mm1 * Nm1; // unknown points
    TSlopeLength = LA;

    /* Allocate CoeffMatrix */
    err = cudaMalloc(&CoeffMatrix, sizeof(double) * LA * LA);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "(Error) >> Cannot allocate CoeffMatrix[%d,%d] on GPU\n", LA, LA);
        exit(-1);
    }

    /* Allocate TheorSlope */
    err = cudaMalloc(&TheorSlope, sizeof(double) * TSlopeLength * 3);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "(Error) >> Cannot allocate TheorSlope[%d,3] on GPU\n", TSlopeLength);
        exit(-1);
    }

    /* Allocate B */
    err = cudaMalloc(&B, sizeof(double) * LA);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "(Error) >> Cannot allocate B[%d] on GPU\n", LA);
        exit(-1);
    }

    gpuGridDef(x0, x1, y0, y1, N, TheorSlope);
    cudaDeviceSynchronize();

    gpuEqsDef(x0, x1, y0, y1, N, LA, CoeffMatrix, B, TheorSlope);
    cudaDeviceSynchronize();

    gpuLinEquSolve(CoeffMatrix, LA, B);
    cudaDeviceSynchronize();

    cudaMemcpy(&TheorSlope[2 * TSlopeLength], B, sizeof(double) * LA, cudaMemcpyDeviceToDevice);

    cudaFree(CoeffMatrix);
    cudaFree(B);

    return;
}

void SensiblePoints(double Ir, double Ii, double Sr, double Si, int MaxIt)
{
    /*
    !  Compute "heated" points
    !  Output:
    !          FieldCoord(Xdots,Ydots,2)
    !          FieldWeight(Xdots,Ydots)
   */

    int ix, iy, iz;
    double ca, cb, za, zb;
    double rad, zan, zbn;
    double Xinc, Yinc;

    fprintf(stdout, "\t>> Computing sensitivity to field effects...\n");

    Xinc = Sr / (double)Xdots;
    Yinc = Si / (double)Ydots;

    for (iy = 0; iy < Ydots; iy++) 
    {
        for (ix = 0; ix < Xdots; ix++) 
        {
            ca = Xinc * ix + Ir;
            cb = Yinc * iy + Ii;
            FieldCoord[index3D(ix, iy, 0, Xdots, Ydots)] = ca;
            FieldCoord[index3D(ix, iy, 1, Xdots, Ydots)] = cb;
            rad = ca * ca * ((double)1.0 + (cb / ca) * (cb / ca));
            zan = 0.0;
            zbn = 0.0;
            for (iz = 1; iz <= MaxIt; iz++) 
            {
                if (rad > (double)4.0) break;
                za = zan;
                zb = zbn;
                zan = ca + (za - zb) * (za + zb);
                zbn = 2.0 * (za * zb + cb / 2.0);
                rad = zan * zan * ((double)1.0 + (zbn / zan) * (zbn / zan));
            }
            FieldWeight[index2D(ix, iy, Xdots)] = iz;
        }
    }

    return;
}

void gpuSensiblePoints(double Ir, double Ii, double Sr, double Si, int MaxIt)
{
    /*
    !  Compute "heated" points
    !  Output:
    !          FieldCoord(Xdots,Ydots,2)
    !          FieldWeight(Xdots,Ydots)
   */

    double Xinc, Yinc;

    cudaError_t err;

    fprintf(stdout, "\t>> Computing sensitivity to field effects...\n");

    Xinc = Sr / (double)Xdots;
    Yinc = Si / (double)Ydots;

    gpuSensiblePointsKernel<<<dim3(8,8,4), dim3(8,8,2)>>>(Ir, Ii, Xinc, Yinc, MaxIt, FieldCoord, FieldWeight);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) printf("(CUDA Error) >> %s\n", cudaGetErrorString(err));

    return;
}

__global__
void gpuSensiblePointsKernel(double Ir, double Ii, double Xinc, double Yinc, int MaxIt, double *FieldCoord, int *FieldWeight)
{
    double ca, cb, za, zb;
    double rad, zan, zbn;

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int idz = threadIdx.z + blockIdx.z * blockDim.z;

    int sizeX = gridDim.x * blockDim.x;
    int sizeY = gridDim.y * blockDim.y;
    int sizeZ = gridDim.z * blockDim.z;

    int ix, iy, iz;

    for (iy = idy; iy < Ydots; iy += sizeY)
    {
        for (ix = idx; ix < Xdots; ix += sizeX)
        {
            ca = Xinc * ix + Ir;
            cb = Yinc * iy + Ii;
            FieldCoord[index3D(ix, iy, 0, Xdots, Ydots)] = ca;
            FieldCoord[index3D(ix, iy, 1, Xdots, Ydots)] = cb;
            rad = ca * ca * ((double)1.0 + (cb / ca) * (cb / ca));
            zan = 0.0;
            zbn = 0.0;
            for (iz = idz; iz <= MaxIt; iz += sizeZ)
            {
                if (iz == 0) break;
                if (rad > (double)4.0) break;
                za = zan;
                zb = zbn;
                zan = ca + (za - zb) * (za + zb);
                zbn = 2.0 * (za * zb + cb / 2.0);
                rad = zan * zan * ((double)1.0 + (zbn / zan) * (zbn / zan));
            }
            FieldWeight[index2D(ix, iy, Xdots)] = iz;
        }
    }
}

void FieldInit()
{
    /*
    ! Initialize field values in the grid. Values are computed on the basis
    ! of the measured values read in subroutine InitGrid and the gross grid
    ! values computed in subroutine FieldDistribution. Moreover sensitiveness
    ! to field effects as computed in subroutine SensiblePoints are taken into
    ! account.
    !
    ! Input:
    !        MeasuredValues(:,3)
    !        FieldWeight(Xdots,Ydots)
    ! Output:
    !        FieldValues(Xdots,Ydots,2)
    */

    int rv;
    double xc, yc, ev, sv, sd, DiscrValue;
    double *DiffValues;

    fprintf(stdout, "\t>> Initializing entity of field effects...\n");

    /* Allocate FieldValues */
    FieldValues = (double*)malloc(sizeof(double) * Xdots * Ydots * 2);
    if (FieldValues == NULL) 
    {
        fprintf(stderr, "(Error@FieldInit) >> Cannot allocate FieldValues[%d,%d,2]\n", Xdots, Ydots);
        exit(-1);
    }
    SetDoubleValue(FieldValues, Xdots * Ydots * 2, (double)0); // OPP: you can use calloc?

    /* Allocate DiffValues */
    DiffValues = (double*)malloc(sizeof(double) * NumInputValues);
    if (DiffValues == NULL) 
    {
        fprintf(stderr, "(Error@FieldInit) >> Cannot allocate DiffValues[%d]\n", NumInputValues);
        exit(-1);
    }
    SetDoubleValue(DiffValues, NumInputValues, (double)0.0); // OPP: you can use calloc?

    /* Compute discrepancy between Measured and Theoretical value */
    DiscrValue = 0.0;
    for (rv = 0; rv < NumInputValues; rv++) 
    {
        xc = MeasuredValues[index2D(rv, 0, NumInputValues)];
        yc = MeasuredValues[index2D(rv, 1, NumInputValues)];

        // TheorSlope is computed on the basis of a coarser grid, so look for the best values near xc, yc coordinates
        sv = NearestValue(xc, yc, TSlopeLength, TheorSlope);
        ev = MeasuredValues[index2D(rv, 2, NumInputValues)];

        DiffValues[rv] = ev - sv;
        DiscrValue += ev - sv;
    }
    DiscrValue = DiscrValue / (double)NumInputValues;

    // Compute standard deviation
    sd = 0.0;
    for (rv = 0; rv < NumInputValues; rv++) sd = sd + (DiffValues[rv] - DiscrValue) * (DiffValues[rv] - DiscrValue);
    sd = sqrt(sd / (double)NumInputValues);

    // Print statistics
    fprintf(stdout, "\t...Number of Points, Mean value, Standard deviation = %d, %12.3e, %12.3e\n", NumInputValues, DiscrValue, sd);

    // Compute FieldValues stage 1
    FieldPoints(DiscrValue);

    free(DiffValues);

    return;
}

void gpuFieldInit()
{
    /*
    ! Initialize field values in the grid. Values are computed on the basis
    ! of the measured values read in subroutine InitGrid and the gross grid
    ! values computed in subroutine FieldDistribution. Moreover sensitiveness
    ! to field effects as computed in subroutine SensiblePoints are taken into
    ! account.
    !
    ! Input:
    !        MeasuredValues(:,3)
    !        FieldWeight(Xdots,Ydots)
    ! Output:
    !        FieldValues(Xdots,Ydots,2)
    */

    int rv;
    double xc, yc, ev, sv, sd, DiscrValue;
    double* DiffValues;

    cudaError_t err;

    fprintf(stdout, "\t>> Initializing entity of field effects...\n");

    /* Allocate FieldValues */
    err = cudaMalloc(&FieldValues, sizeof(double) * Xdots * Ydots * 2);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "(Error@FieldInit) >> Cannot allocate FieldValues[%d,%d,2] in GPU\n", Xdots, Ydots);
        exit(-1);
    }
    cudaMemset(FieldValues, 0, sizeof(double) * Xdots * Ydots * 2);

    /* Allocate DiffValues */
    DiffValues = (double*)malloc(sizeof(double) * NumInputValues);
    if (DiffValues == NULL)
    {
        fprintf(stderr, "(Error@FieldInit) >> Cannot allocate DiffValues[%d]\n", NumInputValues);
        exit(-1);
    }
    memset(DiffValues, 0, sizeof(double) * NumInputValues);

    /* Compute discrepancy between Measured and Theoretical value */

    DiscrValue = 0.0;
    for (rv = 0; rv < NumInputValues; rv++)
    {
        xc = MeasuredValues[index2D(rv, 0, NumInputValues)];
        yc = MeasuredValues[index2D(rv, 1, NumInputValues)];

        // TheorSlope is computed on the basis of a coarser grid, so look for the best values near xc, yc coordinates
        sv = gpuNearestValue(xc, yc, TSlopeLength, TheorSlope);
        ev = MeasuredValues[index2D(rv, 2, NumInputValues)];

        DiffValues[rv] = ev - sv;
        DiscrValue += ev - sv;
    }
    DiscrValue = DiscrValue / (double)NumInputValues;

    // Compute standard deviation
    sd = 0.0;
    for (rv = 0; rv < NumInputValues; rv++) sd = sd + (DiffValues[rv] - DiscrValue) * (DiffValues[rv] - DiscrValue);
    sd = sqrt(sd / (double)NumInputValues);

    // Print statistics
    fprintf(stdout, "\t...Number of Points, Mean value, Standard deviation = %d, %12.3e, %12.3e\n", NumInputValues, DiscrValue, sd);

    // Compute FieldValues stage 1
    
    gpuFieldPoints(DiscrValue);

    free(DiffValues);
}

void Cooling(int steps)
{
    /*
    !  Compute evolution of the effects of the field
    !  Input/Output:
    !                FieldValues(Xdots,Ydots,2)
    */

    int iz, it;
    char fname[80];
    double vmin, vmax;

    fprintf(stdout, "\t>> Computing cooling of field effects ...\n");
    fprintf(stdout, "\t... %d steps ...\n", steps);
    sprintf(fname, "FieldValues0000");

    vmin = vmax = 0.0;
    //RealData2ppm(Xdots, Ydots, &FieldValues[index3D(0, 0, 0, Xdots, Ydots)], &vmin, &vmax, fname);
    Statistics(Xdots, Ydots, &FieldValues[index3D(0, 0, 0, Xdots, Ydots)], 0);

    iz = 1;
    for (it = 1; it <= steps; it++) 
    {
        // Update the value of grid points
        Update(Xdots, Ydots, &FieldValues[index3D(0, 0, iz - 1, Xdots, Ydots)], &FieldValues[index3D(0, 0, 2 - iz, Xdots, Ydots)]);
        iz = 3 - iz;

        // Print and show results 
        sprintf(fname, "FieldValues%4.4d", it);
        //if (it % 4 == 0) RealData2ppm(Xdots, Ydots, &FieldValues[index3D(0, 0, iz - 1, Xdots, Ydots)], &vmin, &vmax, fname);
        Statistics(Xdots, Ydots, &FieldValues[index3D(0, 0, iz - 1, Xdots, Ydots)], it);
    }

    return;
}

void gpuCooling(int steps)
{
    /*
    !  Compute evolution of the effects of the field
    !  Input/Output:
    !                FieldValues(Xdots,Ydots,2)
    */

    int iz, it;
    char fname[80];
    double vmin, vmax;

    double* tmp;
    int reduceLayer = (Xdots * Ydots + 1) / 2;

    cudaError_t err;

    /* Allocate space for temporary results */
    err = cudaMalloc(&tmp, sizeof(double) * reduceLayer * 4);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "(CUDA Error) >> %s\n", cudaGetErrorString(err));
        return;
    }

    // -------------- 

    fprintf(stdout, "\t>> Computing cooling of field effects ...\n");
    fprintf(stdout, "\t... %d steps ...\n", steps);
    sprintf(fname, "FieldValues0000");

    vmin = vmax = 0.0;
    //RealData2ppm(Xdots, Ydots, &FieldValues[index3D(0, 0, 0, Xdots, Ydots)], &vmin, &vmax, fname);
    gpuStatistics(Xdots, Ydots, FieldValues, tmp, 0);

    iz = 1;
    for (it = 1; it <= steps; it++)
    {
        // Update the value of grid points
        gpuUpdate(Xdots, Ydots, &FieldValues[index3D(0, 0, iz - 1, Xdots, Ydots)], &FieldValues[index3D(0, 0, 2 - iz, Xdots, Ydots)]);
        cudaDeviceSynchronize();

        iz = 3 - iz;

        // Print and show results 
        sprintf(fname, "FieldValues%4.4d", it);
        //if (it % 4 == 0) RealData2ppm(Xdots, Ydots, &FieldValues[index3D(0, 0, iz - 1, Xdots, Ydots)], &vmin, &vmax, fname);
        gpuStatistics(Xdots, Ydots, &FieldValues[index3D(0, 0, iz - 1, Xdots, Ydots)], tmp, it);
    }

    cudaFree(tmp);

    return;
}

/* SUB-FUNCTIONS */

void GridDef(double x0, double x1, double y0, double y1, int N, double* Pts)
{
    double x, y, dx, dy;
    int i, j, np, Mm1, Nm1;

    Mm1 = sqrt((double)Xdots) - 1;
    Nm1 = sqrt((double)Ydots) - 1;
    dx = (x1 - x0) / (double)N; 
    dy = (y1 - y0) / (double)N;

    np = -1;
    for (i = 0; i < Mm1; i++) 
    {
        for (j = 0; j < Nm1; j++) 
        {
            np++;
            if (np > Mm1 * Nm1) 
            {
                fprintf(stderr, "(Error@GridDef) >> NP = %d > N*N = %d\n", np, Nm1 * Nm1);
                exit(-1);
            }
            x = x0 + dx * (double)(i + 1);
            y = y0 + dy * (double)(j + 1);
            Pts[index2D(np, 0, TSlopeLength)] = x;
            Pts[index2D(np, 1, TSlopeLength)] = y;
        }
    }
    return;
}

void gpuGridDef(double x0, double x1, double y0, double y1, int N, double* Pts)
{
    double dx, dy;
    int Mm1, Nm1;

    Mm1 = sqrt((double)Xdots) - 1;
    Nm1 = sqrt((double)Ydots) - 1;
    dx = (x1 - x0) / (double)N;
    dy = (y1 - y0) / (double)N;

    gpuGridDefKernel<<<6, 128>>>(x0, y0, dx, dy, Pts, Nm1, Nm1 * Mm1, TSlopeLength);

    return;
}

__global__
void gpuGridDefKernel(double x0, double y0, double dx, double dy, double* Pts, int Nm1, int len, int TSlopeLength)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int gridSize = blockDim.x * gridDim.x;

    int np, i, j;
    double x, y;

    for (np = id; np < len; np += gridSize)
    {
        i = np / Nm1;
        j = np % Nm1;

        x = x0 + dx * (double)(i + 1);
        y = y0 + dy * (double)(j + 1);
        Pts[index2D(i, 0, TSlopeLength)] = x;
        Pts[index2D(i, 1, TSlopeLength)] = y;
    }
}

void EqsDef(double x0, double x1, double y0, double y1, int N, int LA, double* A, double* Rhs, double* Pts)
{
    // Pts(LA,3) - inner grid point Coordinates
    // Rhs(LA)   - Linear equation Right Hand Side
    // A(LA,LA)  - Linear equation matrix

    double x, y, Eps, dx, dy;
    int np, Nm1, pos;

    //  Define A matrix and RHS

    Nm1 = N - 1;
    dx = (x1 - x0) / (double)N; 
    dy = (y1 - y0) / (double)N;

    SetDoubleValue(A, LA * LA, (double)0); // OPP: you can use calloc?
    SetDoubleValue(Rhs, LA, (double)0); // OPP: you can use calloc?

    for (np = 0; np < LA; np++) 
    {
        x = Pts[index2D(np, 0, TSlopeLength)];
        y = Pts[index2D(np, 1, TSlopeLength)];

        A[index2D(np, np, LA)] = -4.0;

        Rhs[np] = (x + y) * dx * dy;

        // define Eps function of grid dimensions 
        Eps = (dx + dy) / 20.0;

        // where is P(x-dx,y) ? 
        if (fabs((x - dx) - x0) < Eps) Rhs[np] = Rhs[np] - Solution(x0, y);
        else 
        {
            // Find pos = position of P(x-dx,y)
            pos = np - Nm1;
            if (fabs(Pts[index2D(pos, 0, TSlopeLength)] - (x - dx)) > Eps) 
            {
                fprintf(stderr, "(Error@EqsDef) >> x-dx: pos, np, d = %d %d %lf\n", pos, np, fabs(Pts[index2D(pos, 0, TSlopeLength)] - (x - dx)));
                exit(-1);
            }
            A[index2D(np, pos, LA)] = 1.0;
        }

        // where is P(x+dx,y) ? 
        if (fabs((x + dx) - x1) < Eps) Rhs[np] = Rhs[np] - Solution(x1, y);
        else 
        {
            // Find pos = position of P(x+dx,y)
            pos = np + Nm1;
            if (fabs(Pts[index2D(pos, 0, TSlopeLength)] - (x + dx)) > Eps) 
            {
                fprintf(stderr, "(Error@EqsDef) >> x+dx: %lf\n", fabs(Pts[index2D(pos, 0, TSlopeLength)] - (x + dx)));
                exit(-1);
            }
            A[index2D(np, pos, LA)] = 1.0;
        }

        // where is P(x,y-dy) ? 
        if (fabs((y - dy) - y0) < Eps) Rhs[np] = Rhs[np] - Solution(x, y0);
        else 
        {
            // Find pos = position of P(x,y-dy)
            pos = np - 1;
            if (fabs(Pts[index2D(pos, 1, TSlopeLength)] - (y - dy)) > Eps) 
            {
                fprintf(stderr, "(Error@EqsDef) >> y-dy: %lf\n", fabs(Pts[index2D(pos, 1, TSlopeLength)] - (y - dy)));
                exit(-1);
            }
            A[index2D(np, pos, LA)] = 1.0;
        }

        // where is P(x,y+dy) ? 
        if (fabs((y + dy) - y1) < Eps) Rhs[np] = Rhs[np] - Solution(x, y1);
        else 
        {
            // Find pos = position of P(x,y-dy)
            pos = np + 1;
            if (fabs(Pts[index2D(pos, 1, TSlopeLength)] - (y + dy)) > Eps) 
            {
                fprintf(stderr, "(Error@EqsDef) >> y+dy: %lf\n", fabs(Pts[index2D(pos, 1, TSlopeLength)] - (y + dy)));
                exit(-1);
            }
            A[index2D(np, pos, LA)] = 1.0;
        }
    }
    return;
}

void gpuEqsDef(double x0, double x1, double y0, double y1, int N, int LA, double* A, double* Rhs, double* Pts)
{
    // Pts(LA,3) - inner grid point Coordinates
    // Rhs(LA)   - Linear equation Right Hand Side
    // A(LA,LA)  - Linear equation matrix

    double x, y, Eps, dx, dy;
    int np, Nm1, pos;

    //  Define A matrix and RHS

    Nm1 = N - 1;
    dx = (x1 - x0) / (double)N;
    dy = (y1 - y0) / (double)N;

    cudaMemset(A, 0, sizeof(double) * LA * LA);
    cudaMemset(Rhs, 0, sizeof(double) * LA);

    gpuEqsDefKernel<<<6, 128>>>(x0, x1, y0, y1, Nm1, dx, dy, LA, A, Rhs, Pts, TSlopeLength);
    
    return;
}

__global__
void gpuEqsDefKernel(double x0, double x1, double y0, double y1, int Nm1, double dx, double dy, int LA, double* A, double* Rhs, double* Pts, int TSlopeLength)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int gridSize = blockDim.x * gridDim.x;

    int np, pos;
    double x, y, Eps;

    for (np = id; np < LA; np += gridSize)
    {
        x = Pts[index2D(np, 0, TSlopeLength)];
        y = Pts[index2D(np, 1, TSlopeLength)];

        A[index2D(np, np, LA)] = -4.0;

        Rhs[np] = (x + y) * dx * dy;

        // define Eps function of grid dimensions 
        Eps = (dx + dy) / 20.0;

        // where is P(x-dx,y) ? 
        if (fabs((x - dx) - x0) < Eps) Rhs[np] = Rhs[np] - gpuSolution(x0, y);
        else
        {
            // Find pos = position of P(x-dx,y)
            pos = np - Nm1;
            A[index2D(np, pos, LA)] = 1.0;
        }

        // where is P(x+dx,y) ? 
        if (fabs((x + dx) - x1) < Eps) Rhs[np] = Rhs[np] - gpuSolution(x1, y);
        else
        {
            // Find pos = position of P(x+dx,y)
            pos = np + Nm1;
            A[index2D(np, pos, LA)] = 1.0;
        }

        // where is P(x,y-dy) ? 
        if (fabs((y - dy) - y0) < Eps) Rhs[np] = Rhs[np] - gpuSolution(x, y0);
        else
        {
            // Find pos = position of P(x,y-dy)
            pos = np - 1;
            A[index2D(np, pos, LA)] = 1.0;
        }

        // where is P(x,y+dy) ? 
        if (fabs((y + dy) - y1) < Eps) Rhs[np] = Rhs[np] - gpuSolution(x, y1);
        else
        {
            // Find pos = position of P(x,y-dy)
            pos = np + 1;
            A[index2D(np, pos, LA)] = 1.0;
        }
    }
}

double Solution(double x, double y)
{
    return ((x * x * x) + (y * y * y)) / (double)6.0;
}

__device__
double gpuSolution(double x, double y)
{
    return ((x * x * x) + (y * y * y)) / (double)6.0;
}

int LinEquSolve(double* a, int n, double* b)
{
    /* Gauss-Jordan elimination algorithm */
    int i, j, k, l, icol, irow;
    int *indcol, *indrow, *ipiv;
    double bigger, temp;

    /* Allocate indcol */
    indcol = (int*)malloc(sizeof(int) * n);
    if (indcol == NULL) 
    {
        fprintf(stderr, "(Error@LinEquSolve) >> Cannot allocate indcol[%d]\n", n);
        return(-1);
    }

    /* Allocate indrow */
    indrow = (int*)malloc(sizeof((int)1) * n);
    if (indrow == NULL) 
    {
        fprintf(stderr, "(Error@LinEquSolve) >> Cannot allocate indrow[%d]\n", n);
        return(-1);
    }

    /* Allocate ipiv */
    ipiv = (int*)malloc(sizeof((int)1) * n);
    if (ipiv == NULL) 
    {
        fprintf(stderr, "(Error@LinEquSolve) >> Cannot allocate ipiv[%d]\n", n);
        return(-1);
    }
    SetIntValue(ipiv, n, 0); // OPP: you can use calloc?

    /* Actual algorithm */

    for (i = 0; i < n; i++) 
    {
        bigger = 0.0;

        for (j = 0; j < n; j++) 
        {
            if (ipiv[j] != 1) 
            {
                for (k = 0; k < n; k++) 
                {
                    if (ipiv[k] == 0 && bigger <= fabs(a[index2D(j, k, n)])) 
                    {
                        bigger = fabs(a[index2D(j, k, n)]);
                        irow = j;
                        icol = k;
                    }
                }
            }
        }

        ipiv[icol] = ipiv[icol] + 1;

        if (irow != icol) 
        {
            for (l = 0; l < n; l++) 
            {
                temp = a[index2D(irow, l, n)];
                a[index2D(irow, l, n)] = a[index2D(icol, l, n)];
                a[index2D(icol, l, n)] = temp;
            }
            temp = b[irow];
            b[irow] = b[icol];
            b[icol] = temp;
        }

        indrow[i] = irow;
        indcol[i] = icol;

        if (a[index2D(icol, icol, n)] == 0.0) 
        {
            fprintf(stderr, "(Error@LinEquSolve) >> a(%d,%d): singular matrix!", icol, icol);
            return -2;
        }

        temp = (double)1.0 / a[index2D(icol, icol, n)];
        a[index2D(icol, icol, n)] = 1.0;

        for (l = 0; l < n; l++) a[index2D(icol, l, n)] = a[index2D(icol, l, n)] * temp;

        b[icol] = b[icol] * temp;

        for (l = 0; l < n; l++) 
        {
            if (l != icol) 
            {
                temp = a[index2D(l, icol, n)];
                a[index2D(l, icol, n)] = 0.0;
                for (k = 0; k < n; k++) 
                {
                    a[index2D(l, k, n)] = a[index2D(l, k, n)] - a[index2D(icol, k, n)] * temp;
                }
                b[l] = b[l] - b[icol] * temp;
            }
        }
    }

    for (l = n - 1; l >= 0; l--) 
    {
        if (indrow[l] != indcol[l]) 
        {
            for (k = 0; k < n; k++) 
            {
                temp = a[index2D(k, indrow[l], n)];
                a[index2D(k, indrow[l], n)] = a[index2D(k, indcol[l], n)];
                a[index2D(k, indcol[l], n)] = temp;
            }
        }
    }

    free(indcol);
    free(indrow);
    free(ipiv);

    return 0;
}

int gpuLinEquSolve(double* a, int n, double* b)
{
    /* Gauss-Jordan elimination algorithm */
    int* indcol, * indrow, * ipiv;

    cudaError_t err;

    /* Allocate indcol */
    err = cudaMalloc(&indcol, sizeof(int) * n);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "(Error@LinEquSolve) >> Cannot allocate indcol[%d] on GPU\n", n);
        return(-1);
    }

    /* Allocate indrow */
    err = cudaMalloc(&indrow, sizeof(int) * n);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "(Error@LinEquSolve) >> Cannot allocate indrow[%d] on GPU\n", n);
        return(-1);
    }

    /* Allocate ipiv */
    err = cudaMalloc(&ipiv, sizeof(int) * n);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "(Error@LinEquSolve) >> Cannot allocate ipiv[%d] on GPU\n", n);
        return(-1);
    }
    cudaMemset(ipiv, 0, sizeof(int) * n);

    /* Actual algorithm */

    int* maxIndex;
    double* maxima;

    err = cudaMalloc(&maxIndex, sizeof(int) * n);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "(Error@LinEquSolve) >> Cannot allocate maxIndex on GPU\n", n);
        return(-1);
    }

    err = cudaMalloc(&maxima, sizeof(double) * n);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "(Error@LinEquSolve) >> Cannot allocate maxima on GPU\n", n);
        return(-1);
    }

    gpuLinEquSolveKernel<<<32, 256>>>(maxima, maxIndex, a, b, indrow, indcol, ipiv, n);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) printf("(CUDA Error) >> %s\n", cudaGetErrorString(err));

    cudaFree(indcol);
    cudaFree(indrow);
    cudaFree(ipiv);
    cudaFree(maxIndex);
    cudaFree(maxima);

    return 0;
}

__global__
void gpuLinEquSolveKernel(double* maxima, int* maxIndex, double* a, double* b, int* indrow, int* indcol, int* ipiv, int n)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int gridSize = blockDim.x * gridDim.x;

    int iter, i, j, k, icol;
    double max, tmp;

    for (iter = 0; iter < n; iter++)
    {
        for (i = id; i < n; i += gridSize)
        {
            max = 0;
            icol = 0;

            if (ipiv[i] != 1)
            {
                for (j = 0; j < n; j++)
                {
                    if (ipiv[j] == 0 && max <= fabs(a[index2D(i, j, n)]))
                    {
                        max = fabs(a[index2D(i, j, n)]);
                        icol = j;
                    }
                }
            }

            maxima[i] = max;
            maxIndex[i] = icol;
        }

        __syncthreads();

        if (id == 0)
        {
            j = 0;

            for (i = 1; i < n; i++)
                if (maxima[i] > maxima[j]) j = i;

            maxRow = j;
            maxCol = maxIndex[j];

            ipiv[maxCol] = ipiv[maxCol] + 1;
        }

        __syncthreads();

        if (maxRow != maxCol)
        {
            for (i = id; i < n; i += gridSize)
            {
                tmp = a[index2D(maxRow, i, n)];
                a[index2D(maxRow, i, n)] = a[index2D(maxCol, i, n)];
                a[index2D(maxCol, i, n)] = tmp;
            }

            if (id == 0)
            {
                tmp = b[maxRow];
                b[maxRow] = b[maxCol];
                b[maxCol] = tmp;
            }
        }

        if (id == 0)
        {
            indrow[iter] = maxRow;
            indcol[iter] = maxCol;
        }

        __syncthreads();

        // TODO: Missing check on singularity

        if (id == 0)
        {
            temp = a[index2D(maxCol, maxCol, n)];
            a[index2D(maxCol, maxCol, n)] = 1.0;
            b[maxCol] /= temp;
        }

        for (i = id; i < n; i += gridSize)
        {
            a[index2D(maxCol, i, n)] /= temp;
        }

        __syncthreads();

        for (i = id; i < n; i += gridSize)
        {
            if (i != maxCol)
            {
                tmp = a[index2D(i, maxCol, n)];
                a[index2D(i, maxCol, n)] = 0.0;
                for (k = 0; k < n; k++)
                {
                    a[index2D(i, k, n)] = a[index2D(i, k, n)] - a[index2D(maxCol, k, n)] * tmp;
                }
                b[i] = b[i] - b[maxCol] * tmp;
            }
        }

        __syncthreads();
    }
}

double NearestValue(double xc, double yc, int ld, double* Values)
{
    // look for the best values near xc, yc coordinates
    double v;

    double d, md; // minimum distance
    int np; // number of nearest points
    int i;

    md = ((xc - Values[index2D(0, 0, ld)]) * (xc - Values[index2D(0, 0, ld)])) +
         ((yc - Values[index2D(0, 1, ld)]) * (yc - Values[index2D(0, 1, ld)]));

    // Compute lowest distance
    for (i = 0; i < ld; i++) 
    {
        d = ((xc - Values[index2D(i, 0, ld)]) * (xc - Values[index2D(i, 0, ld)])) +
            ((yc - Values[index2D(i, 1, ld)]) * (yc - Values[index2D(i, 1, ld)]));
        if (md > d) md = d;
    }

    np = 0;
    v = 0.0;

    // Compute nearest value
    for (i = 0; i < ld; i++) 
    {
        d = ((xc - Values[index2D(i, 0, ld)]) * (xc - Values[index2D(i, 0, ld)])) +
            ((yc - Values[index2D(i, 1, ld)]) * (yc - Values[index2D(i, 1, ld)]));
        if (md == d) 
        {
            // add contributed value
            np = np + 1;
            v = v + Values[index2D(i, 2, ld)];
        }
    }

    // mean value
    v = v / (double)np;

    return v;
}

double gpuNearestValue(double xc, double yc, int ld, double* Values)
{
    double v;

    double* dist;
    int* mask;
    double* partialResult;

    cudaError_t err;

    /* Allocate dist */
    err = cudaMalloc(&dist, sizeof(double) * ld);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "(CUDA Error) >> Cannot allocate dist in GPU\n");
        exit(-1);
    }

    /* Allocate mask */
    err = cudaMalloc(&mask, sizeof(int) * ld);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "(CUDA Error) >> Cannot allocate mask in GPU\n");
        exit(-1);
    }

    /* Allocate partialResult */
    err = cudaMalloc(&partialResult, sizeof(double) * ld);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "(CUDA Error) >> Cannot allocate partialResult in GPU\n");
        exit(-1);
    }

    gpuNearestValueKernel<<<1, 768 >>>(xc, yc, Values, dist, mask, partialResult, ld);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "(CUDA Error) >> %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    cudaMemcpyFromSymbol(&v, globalV, sizeof(double), 0, cudaMemcpyDeviceToHost);

    cudaFree(dist);
    cudaFree(mask);
    cudaFree(partialResult);

    return v;
}

__global__
void gpuNearestValueKernel(double xc, double yc, double* Values, double* dist, int* mask, double* partialResult, int n)
{
    int i;
    double a, b, v;
    __shared__ int np;
    __shared__ double md;

    // Compute distances

    for (i = threadIdx.x; i < n; i += blockDim.x)
    {
        a = xc - Values[index2D(i, 0, n)];
        b = yc - Values[index2D(i, 1, n)];
        dist[i] = a * a + b * b;
    }
    __syncthreads();

    // Compute lowest distance

    if (threadIdx.x == 0)
    {
        md = dist[0];

        for (i = 1; i < n; i++)
            if (dist[i] < md) md = dist[i];
    }
    __syncthreads();

    // Compute nearest value

    for (i = threadIdx.x; i < n; i += blockDim.x)
    {
        mask[i] = (dist[i] == md);
        partialResult[i] = (dist[i] == md) * Values[index2D(i, 2, n)];
    }
    __syncthreads();

    // Reduce

    if (threadIdx.x == 0)
    {
        np = 0;
        v = 0.0;

        for (i = 0; i < n; i++)
        {
            np = np + mask[i];
            v = v + partialResult[i];
        }

        globalV = v / (double)np;
    }
}

void FieldPoints(double Diff)
{
    int ix, iy;
    double xc, yc, sv;
    double rmin, rmax;

    rmax = MaxIntVal(Xdots * Ydots, FieldWeight);
    rmin = MinIntVal(Xdots * Ydots, FieldWeight);

    for (iy = 0; iy < Ydots; iy++) 
    {
        for (ix = 0; ix < Xdots; ix++) 
        {
            xc = FieldCoord[index3D(ix, iy, 0, Xdots, Ydots)];
            yc = FieldCoord[index3D(ix, iy, 1, Xdots, Ydots)];

            // Compute effects of field in every point
            sv = NearestValue(xc, yc, TSlopeLength, TheorSlope);
            FieldValues[index3D(ix, iy, 0, Xdots, Ydots)] = 293.16 + 80 * (Diff + sv) * (FieldWeight[index2D(ix, iy, Xdots)] - rmin) / (rmax - rmin);
        }
    }

    // Copy initial status 
    // OPP: use memcpy?
    for (iy = 0; iy < Ydots; iy++) 
    {
        for (ix = 0; ix < Xdots; ix++) 
        {
            FieldValues[index3D(ix, iy, 1, Xdots, Ydots)] = FieldValues[index3D(ix, iy, 0, Xdots, Ydots)];
        }
    }

    return;
}

void gpuFieldPoints(double Diff)
{
    cudaError_t err;

    MinMaxIntVal(FieldWeight, Xdots * Ydots);

    gpuFieldPointsKernel<<<dim3(16, 16, 1), dim3(32, 32, 1)>>>(FieldCoord, FieldWeight, FieldValues, Diff, TSlopeLength, TheorSlope);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "(CUDA Error) >> %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(&FieldValues[Xdots * Ydots], FieldValues, sizeof(double) * Xdots * Ydots, cudaMemcpyDeviceToDevice);

    return;
}

void MinMaxIntVal(int* Values, int len)
{
    int* tmpMax;
    int* tmpMin;

    cudaError_t err;

    /* Allocate Temporary Results */
    err = cudaMalloc(&tmpMax, sizeof(int) * (len + 1));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "(cudaError) >>> %s\n", cudaGetErrorString(err));
        return;
    }

    err = cudaMalloc(&tmpMin, sizeof(int) * (len + 1));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "(cudaError) >>> %s\n", cudaGetErrorString(err));
        return;
    }

    MinMaxIntValKernel<<<1, 768>>>(Values, len, tmpMin, tmpMax);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "(cudaError) >>> %s\n", cudaGetErrorString(err));
        return;
    }

    cudaFree(tmpMax);
    cudaFree(tmpMin);
}

__global__
void MinMaxIntValKernel(int *Values, int len, int *tmpMin, int *tmpMax)
{
    int i;
    double a, b;
    int layerLength = (len + 1) / 2;
    int last;

    for (i = threadIdx.x; i < layerLength; i += blockDim.x)
    {
        if ((2 * i + 1) < len)
        {
            a = Values[2 * i];
            b = Values[2 * i + 1];

            tmpMin[i] = (a < b) ? a : b;
            tmpMax[i] = (a > b) ? a : b;
        }
        else 
        {
            tmpMin[i] = Values[2 * i];
            tmpMax[i] = Values[2 * i];
        }
    }

    last = layerLength % 2;
    layerLength = (layerLength + 1) / 2;

    __syncthreads();

    while (layerLength > 1)
    {
        for (i = threadIdx.x; i < layerLength; i += blockDim.x)
        {
            if (i < layerLength - last)
            {
                a = tmpMin[2 * i];
                b = tmpMin[2 * i + 1];
                tmpMin[2 * i] = (a < b) ? a : b;

                a = tmpMax[2 * i];
                b = tmpMax[2 * i + 1];
                tmpMax[2 * i] = (a > b) ? a : b;
            }
        }

        __syncthreads();

        for (i = threadIdx.x; i < layerLength; i += blockDim.x)
        {
            tmpMin[i] = tmpMin[2 * i];
            tmpMax[i] = tmpMax[2 * i];
        }

        last = layerLength % 2;
        layerLength = (layerLength + 1) / 2;

        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        rMin = tmpMin[0];
        rMax = tmpMax[0];
    }
}

__global__
void gpuFieldPointsKernel(double* FieldCoord, int* FieldWeight, double* FieldValues, double Diff, int TSlopeLength, double* TheorSlope)
{
    int iy, ix;
    double xc, yc, sv;

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int sizeX = blockDim.x * gridDim.x;
    int sizeY = blockDim.y * gridDim.y;

    for (iy = idy; iy < Ydots; iy += sizeY)
    {
        for (ix = idx; ix < Xdots; ix += sizeX)
        {
            xc = FieldCoord[index3D(ix, iy, 0, Xdots, Ydots)];
            yc = FieldCoord[index3D(ix, iy, 1, Xdots, Ydots)];

            // Compute effects of field in every point
            sv = deviceNearestValue(xc, yc, TSlopeLength, TheorSlope);
            FieldValues[index3D(ix, iy, 0, Xdots, Ydots)] = 293.16 + 80 * (Diff + sv) * (FieldWeight[index2D(ix, iy, Xdots)] - rMin) / (rMax - rMin);
        }
    }
}

__device__
double deviceNearestValue(double xc, double yc, int ld, double* Values)
{
    // look for the best values near xc, yc coordinates
    double v;

    double d, md; // minimum distance
    int np;       // number of nearest points
    int i;

    md = ((xc - Values[index2D(0, 0, ld)]) * (xc - Values[index2D(0, 0, ld)])) +
         ((yc - Values[index2D(0, 1, ld)]) * (yc - Values[index2D(0, 1, ld)]));

    // Compute lowest distance
    for (i = 0; i < ld; i++)
    {
        d = ((xc - Values[index2D(i, 0, ld)]) * (xc - Values[index2D(i, 0, ld)])) +
            ((yc - Values[index2D(i, 1, ld)]) * (yc - Values[index2D(i, 1, ld)]));
        if (md > d) md = d;
    }

    np = 0;
    v = 0.0;

    // Compute nearest value
    for (i = 0; i < ld; i++)
    {
        d = ((xc - Values[index2D(i, 0, ld)]) * (xc - Values[index2D(i, 0, ld)])) +
            ((yc - Values[index2D(i, 1, ld)]) * (yc - Values[index2D(i, 1, ld)]));
        if (md == d)
        {
            // add contributed value
            np = np + 1;
            v = v + Values[index2D(i, 2, ld)];
        }
    }

    // mean value
    v = v / (double)np;

    return v;
}

void RealData2ppm(int s1, int s2, double* rdata, double* vmin, double* vmax, char* name)
{
    /* Simple subroutine to dump integer data in a PPM format */

    int cm[3][256];  /* R,G,B, Colour Map */
    FILE* ouni, * ColMap;
    int i, j, rc, vp, vs;
    double  rmin, rmax;
    char  fname[80], jname[80], command[80];

    /* Load color map: 256 colours */
    ColMap = fopen("ColorMap.txt", "r");
    if (ColMap == NULL) 
    {
        fprintf(stderr, "(Error@RealData2ppm) >> Cannot open ColorMap.txt\n");
        exit(-1);
    }
    for (i = 0; i < 256; i++) 
    {
        if (fscanf(ColMap, " %3d %3d %3d", &cm[0][i], &cm[1][i], &cm[2][i]) < 3) 
        {
            fprintf(stderr, "(Error@RealData2ppm) >> reading colour map at line %d: r, g, b =", (i + 1));
            fprintf(stderr, " %3.3d %3.3d %3.3d\n", cm[0][i], cm[1][i], cm[2][i]);
            exit(1);
        }
    }
    fclose(ColMap);

    /* Write on unit 700 with PPM format */
    strcpy(fname, name);
    strcat(fname, ".ppm\0");

    ouni = fopen(fname, "w");
    if (!ouni) fprintf(stderr, "(Error@RealData2ppm) >> write access to file %s\n", fname);

    /*  Magic code */
    fprintf(ouni, "P3\n");

    /*  Dimensions */
    fprintf(ouni, "%d %d\n", s1, s2);

    /*  Maximum value */
    fprintf(ouni, "255\n");

    /*  Values from 0 to 255 */
    rmin = MinDoubleVal(s1 * s2, rdata); 
    rmax = MaxDoubleVal(s1 * s2, rdata);

    if ((*vmin == *vmax) && (*vmin == (double)0.0)) 
    {
        *vmin = rmin; 
        *vmax = rmax;
    }
    else 
    {
        rmin = *vmin; 
        rmax = *vmax;
    }

    vs = 0;
    for (i = 0; i < s1; i++) 
    {
        for (j = 0; j < s2; j++) 
        {
            vp = (int)((rdata[i + (j * s1)] - rmin) * 255.0 / (rmax - rmin));

            if (vp < 0) vp = 0;
            if (vp > 255) vp = 255;

            vs++;

            fprintf(ouni, " %3.3d %3.3d %3.3d", cm[0][vp], cm[1][vp], cm[2][vp]);

            if (vs >= 10) 
            {
                fprintf(ouni, " \n");
                vs = 0;
            }
        }
        fprintf(ouni, " ");
        vs = 0;
    }
    fclose(ouni);

    return;
}

void Statistics(int s1, int s2, double* rdata, int step)
{
    double mnv, mv, mxv, sd;
    int i, j;

    // OPP: Can mean value and standard deviation be computed together?

    // Compute MEAN VALUE 
    mv = 0.0;
    mnv = mxv = rdata[0];
    for (i = 0; i < s1; i++) 
    {
        for (j = 0; j < s2; j++) 
        {
            mv = mv + rdata[i + (j * s1)];
            if (mnv > rdata[i + (j * s1)]) mnv = rdata[i + (j * s1)];
            if (mxv < rdata[i + (j * s1)]) mxv = rdata[i + (j * s1)];
        }
    }
    mv = mv / (double)(s1 * s2);

    // Compute STANDARD DEVIATION
    sd = 0.0;
    for (i = 0; i < s1; i++) 
    {
        for (j = 0; j < s2; j++) 
        {
            sd = sd + (rdata[i + (j * s1)] - mv) * (rdata[i + (j * s1)] - mv);
        }
    }
    sd = sqrt(sd / (double)(s1 * s2));

    fprintf(stdout, ">> Step %4d: min, mean, max, std = %12.3e, %12.3e, %12.3e, %12.3e\n", step, mnv, mv, mxv, sd);

    return;
}

void gpuStatistics(int s1, int s2, double* rdata, double *tmp, int step)
{
    double mnv, mv, mxv, sd;

    double* tmpMin;
    double* tmpMax;
    double* tmpMean;
    double* tmpStd;

    cudaError_t err;

    int reduceLayer = (s1 * s2 + 1) / 2;

    tmpMin = tmp;
    tmpMax = &tmpMin[reduceLayer];
    tmpMean = &tmpMax[reduceLayer];
    tmpStd = &tmpMean[reduceLayer];

    gpuStatisticsKernel<<<256, 128>>>(tmpMin, tmpMax, tmpMean, tmpStd, rdata, s1 * s2, reduceLayer);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "(CUDA Error) >> %s\n", cudaGetErrorString(err));
        return;
    }

    cudaMemcpyFromSymbol(&mnv, rMin, sizeof(double), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&mv, rMean, sizeof(double), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&mxv, rMax, sizeof(double), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&sd, rStd, sizeof(double), 0, cudaMemcpyDeviceToHost);

    fprintf(stdout, ">> Step %4d: min, mean, max, std = %12.3e, %12.3e, %12.3e, %12.3e\n", step, mnv, mv, mxv, sd);

    return;
}

__global__
void gpuStatisticsKernel(double *tmpMin, double *tmpMax, double *tmpMean, double *tmpStd, double *Values, int len, int reduceLayer)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int i;
    double a, b;
    int layerLength = reduceLayer;

    int last;

    for (i = id; i < layerLength; i += stride)
    {
        if ((2 * i + 1) < len)
        {
            a = Values[2 * i];
            b = Values[2 * i + 1];

            tmpMin[i] = (a < b) ? a : b;
            tmpMax[i] = (a > b) ? a : b;
            tmpMean[i] = a + b;
        }
        else
        {
            tmpMin[i] = Values[2 * i];
            tmpMax[i] = Values[2 * i];
            tmpMean[i] = Values[2 * i];
        }
    }

    last = layerLength % 2;
    layerLength = (layerLength + 1) / 2;

    __syncthreads();

    while (layerLength > 1)
    {
        for (i = id; i < layerLength; i += stride)
        {
            if (i < layerLength - last)
            {
                a = tmpMin[2 * i];
                b = tmpMin[2 * i + 1];
                tmpMin[2 * i] = (a < b) ? a : b;

                a = tmpMax[2 * i];
                b = tmpMax[2 * i + 1];
                tmpMax[2 * i] = (a > b) ? a : b;

                a = tmpMean[2 * i];
                b = tmpMean[2 * i + 1];
                tmpMean[2 * i] = a + b;
            }
        }

        __syncthreads();

        for (i = id; i < layerLength; i += stride)
        {
            tmpMin[i] = tmpMin[2 * i];
            tmpMax[i] = tmpMax[2 * i];
            tmpMean[i] = tmpMean[2 * i];
        }

        last = layerLength % 2;
        layerLength = (layerLength + 1) / 2;

        __syncthreads();
    }

    if (id == 0)
    {   
        a = tmpMin[0];
        b = tmpMin[1];
        rMin = (a < b) ? a : b;

        a = tmpMax[0];
        b = tmpMax[1];
        rMax = (a > b) ? a : b;

        a = tmpMean[0];
        b = tmpMean[1];
        rMean = (a + b) / (double)len;
    }

    __syncthreads();

    // Compute STD

    double mv = rMean;

    layerLength = reduceLayer;

    for (i = id; i < layerLength; i += stride)
    {
        if ((2 * i + 1) < len)
        {
            a = Values[2 * i];
            b = Values[2 * i + 1];

            a = (a - mv) * (a - mv);
            b = (b - mv) * (b - mv);

            tmpStd[i] = a + b;
        }
        else
        {
            a = Values[2 * i];
            a = (a - mv) * (a - mv);

            tmpStd[i] = a;
        }
    }

    last = layerLength % 2;
    layerLength = (layerLength + 1) / 2;

    __syncthreads();

    while (layerLength > 1)
    {
        for (i = id; i < layerLength; i += stride)
        {
            if (i < layerLength - last)
            {
                a = tmpStd[2 * i];
                b = tmpStd[2 * i + 1];
                tmpStd[2 * i] = a + b;
            }
        }

        __syncthreads();

        for (i = id; i < layerLength; i += stride) tmpStd[i] = tmpStd[2 * i];

        last = layerLength % 2;
        layerLength = (layerLength + 1) / 2;

        __syncthreads();
    }

    if (id == 0)
    {
        rStd = sqrt((tmpStd[0] + tmpStd[1]) / (double)len);
    }
}

void Update(int xdots, int ydots, double* u1, double* u2)
{
    /* Compute next step using matrices g1, g2 of dimension (nr,nc) */

    int i, j;
    double CX, CY;
    double hx, dgx, hy, dgy, dd;

    dd = 0.0000001;
    hx = 1.0 / (double)xdots;
    hy = 1.0 / (double)ydots;
    dgx = -2.0 + hx * hx / (2 * dd);
    dgy = -2.0 + hy * hy / (2 * dd);
    CX = dd / (hx * hx);
    CY = dd / (hy * hy);

    for (j = 0; j < ydots - 1; j++) 
    {
        for (i = 0; i < xdots - 1; i++) 
        {
            if (i <= 0 || i >= xdots - 1) 
            {
                u2[index2D(i, j, xdots)] = u1[index2D(i, j, xdots)];
                continue;
            }

            if (j <= 0 || j >= ydots - 1) 
            {
                u2[index2D(i, j, xdots)] = u1[index2D(i, j, xdots)];
                continue;
            }

            u2[index2D(i, j, xdots)] = CX * (u1[index2D((i - 1), j, xdots)]
                                       + u1[index2D((i + 1), j, xdots)] + dgx * u1[index2D((i + 1), j, xdots)])
                                       + CY * (u1[index2D(i, (j - 1), xdots)]
                                       + u1[index2D(i, (j + 1), xdots)] + dgy * u1[index2D(i, j, xdots)]);
        }
    }

    for (j = 0; j < ydots - 1; j++) 
    {
        u2[index2D(0, j, xdots)] = u2[index2D(1, j, xdots)];
        u2[index2D(Xdots - 1, j, xdots)] = u2[index2D(Xdots - 2, j, xdots)];
    }

    for (i = 0; i < xdots - 1; i++) 
    {
        u2[index2D(i, 0, xdots)] = u2[index2D(i, 1, xdots)];
        u2[index2D(i, Ydots - 1, xdots)] = u2[index2D(i, Ydots - 2, xdots)];
    }

    return;
}

void gpuUpdate(int xdots, int ydots, double* u1, double* u2)
{
    /* Compute next step using matrices g1, g2 of dimension (nr,nc) */

    int i, j;
    double CX, CY;
    double hx, dgx, hy, dgy, dd;

    dd = 0.0000001;
    hx = 1.0 / (double)xdots;
    hy = 1.0 / (double)ydots;
    dgx = -2.0 + hx * hx / (2 * dd);
    dgy = -2.0 + hy * hy / (2 * dd);
    CX = dd / (hx * hx);
    CY = dd / (hy * hy);

    gpuUpdateKernel<<<dim3(16, 16, 1), 128>>>(xdots, ydots, u1, u2, CX, CY, dgx, dgy);

    return;
}

__global__
void gpuUpdateKernel(int xdots, int ydots, double* u1, double* u2, double CX, double CY, double dgx, double dgy)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int sizeX = blockDim.x * gridDim.x;
    int sizeY = blockDim.y * gridDim.y;

    int i, j;

    for (j = idy; j < ydots - 1; j += sizeY)
    {
        for (i = idx; i < xdots - 1; i += sizeX)
        {
            if (i <= 0 || i >= xdots - 1)
            {
                u2[index2D(i, j, xdots)] = u1[index2D(i, j, xdots)];
                continue;
            }

            if (j <= 0 || j >= ydots - 1)
            {
                u2[index2D(i, j, xdots)] = u1[index2D(i, j, xdots)];
                continue;
            }

            u2[index2D(i, j, xdots)] = CX * (u1[index2D((i - 1), j, xdots)]
                                     + u1[index2D((i + 1), j, xdots)] 
                                     + dgx * u1[index2D((i + 1), j, xdots)])
                                     + CY * (u1[index2D(i, (j - 1), xdots)]
                                     + u1[index2D(i, (j + 1), xdots)] 
                                     + dgy * u1[index2D(i, j, xdots)]);
        }
    }

    __syncthreads();

    for (j = idy; j < ydots - 1; j += sizeY)
    {
        u2[index2D(0, j, xdots)] = u2[index2D(1, j, xdots)];
        u2[index2D(Xdots - 1, j, xdots)] = u2[index2D(Xdots - 2, j, xdots)];
    }

    for (i = idx; i < xdots - 1; i += sizeX)
    {
        u2[index2D(i, 0, xdots)] = u2[index2D(i, 1, xdots)];
        u2[index2D(i, Ydots - 1, xdots)] = u2[index2D(i, Ydots - 2, xdots)];
    }
}