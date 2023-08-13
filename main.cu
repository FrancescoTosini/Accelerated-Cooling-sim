#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"

#include <cublas_v2.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils/constants.h"
#include "utils/parallel.cuh"
#include "utils/serial.h"

int main(int argc, char *argv[]) {
    clock_t t0, t1, p0, p1;

    t0 = clock();
    printf(">> Starting\n");
    int devices = 0;

    cudaError_t err = cudaGetDeviceCount(&devices);

    if (devices > 0 && err == cudaSuccess) {
        printf("running on GPU\n");
    }

    // Read input file
    p0 = clock();
#ifdef ACCEL
    gpuInitGrid(PATH_INPUT "Cooling.inp");
#else
    InitGrid(PATH_INPUT "Cooling.inp");
#endif
    p1 = clock();
    fprintf(stdout, ">> InitGrid ended in %lf seconds\n", (double)(p1 - p0) / CLOCKS_PER_SEC);

    // TheorSlope(TSlopeLength,3)
    p0 = clock();
#ifdef ACCEL
    gpuFieldDistribution();
#else
    FieldDistribution();
#endif
    p1 = clock();
    fprintf(stdout, ">> FieldDistribution ended in %lf seconds\n",
            (double)(p1 - p0) / CLOCKS_PER_SEC);

    // FieldCoord(Xdots,Ydots,2), FieldWeight(Xdots,Ydots)
    p0 = clock();
#ifdef ACCEL
    gpuSensiblePoints(Sreal, Simag, Rreal, Rimag, MaxIters);
#else
    SensiblePoints(Sreal, Simag, Rreal, Rimag, MaxIters);
#endif
    p1 = clock();
    fprintf(stdout, ">> SensiblePoints ended in %lf seconds\n",
            (double)(p1 - p0) / CLOCKS_PER_SEC);

    // MeasuredValues(:,3), FieldWeight(Xdots,Ydots) ->
    // FieldValues(Xdots,Ydots,2)
    p0 = clock();
#ifdef ACCEL
    gpuFieldInit();
#else
    FieldInit();
#endif
    p1 = clock();
    fprintf(stdout, ">> FieldInit ended in %lf seconds\n",
            (double)(p1 - p0) / CLOCKS_PER_SEC);

    // FieldValues(Xdots,Ydots,2)
    p0 = clock();
#ifdef ACCEL
    gpuCooling(TimeSteps);
#else
    Cooling(TimeSteps);
#endif
    p1 = clock();
    fprintf(stdout, ">> Cooling ended in %lf seconds\n",
            (double)(p1 - p0) / CLOCKS_PER_SEC);

    t1 = clock();
    fprintf(stdout, ">> Computations ended in %lf seconds\n",
            (double)(t1 - t0) / CLOCKS_PER_SEC);
    // End Program

#ifdef ACCEL
    cudaFree(FieldWeight);
    cudaFree(FieldCoord);
    cudaFree(TheorSlope);
    cudaFree(FieldValues);
#else
    free(FieldWeight);
    free(FieldCoord);
    free(TheorSlope);
    free(FieldValues);
#endif
    free(MeasuredValues);

    return 0;
}
