#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"

#include <cusolverDn.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "utils/constants.h"
#include "utils/parallel.cuh"
#include "utils/serial.h"

//undef ACCELERATED

// For Testing Purposes

void getGpuPointers(int** gpuFieldWeight, double** gpuFieldCoord, double** gpuTheorSlope, double** gpuFieldValues) {
    *gpuFieldWeight = FieldWeight;
    *gpuFieldCoord = FieldCoord;
    *gpuTheorSlope = TheorSlope;
    *gpuFieldValues = FieldValues;
}

void getCpuPointers(int** cpuFieldWeight, double** cpuFieldCoord, double** cpuTheorSlope, double** cpuFieldValues) {
    *cpuFieldWeight = FieldWeight;
    *cpuFieldCoord = FieldCoord;
    *cpuTheorSlope = TheorSlope;
    *cpuFieldValues = FieldValues;
}

void setGpuPointers(int* gpuFieldWeight, double* gpuFieldCoord, double* gpuTheorSlope, double* gpuFieldValues) {
    FieldWeight = gpuFieldWeight;
    FieldCoord = gpuFieldCoord;
    TheorSlope = gpuTheorSlope;
    FieldValues = gpuFieldValues;
}

void setCpuPointers(int* cpuFieldWeight, double* cpuFieldCoord, double* cpuTheorSlope, double* cpuFieldValues) {
    FieldWeight = cpuFieldWeight;
    FieldCoord = cpuFieldCoord;
    TheorSlope = cpuTheorSlope;
    FieldValues = cpuFieldValues;
}

int test() {

    int i, correct;

    int* cpuFieldWeight;
    double* cpuFieldCoord;
    double* cpuTheorSlope;
    double* cpuFieldValues;

    int* gpuFieldWeight;
    double* gpuFieldCoord;
    double* gpuTheorSlope;
    double* gpuFieldValues;

    int* tmpFieldWeight;
    double* tmpFieldCoord;
    double* tmpTheorSlope;
    double* tmpFieldValues;

    // Check correctness against CPU
    printf("\tTEST SESSION\n");
    printf(">> Checking correctness against CPU...\n");

    gpuInitGrid("Cooling.inp");
    getGpuPointers(&gpuFieldWeight, &gpuFieldCoord, &gpuTheorSlope, &gpuFieldValues);

    InitGrid("Cooling.inp");
    getCpuPointers(&cpuFieldWeight, &cpuFieldCoord, &cpuTheorSlope, &cpuFieldValues);

    printf(">> Checking correctness against CPU... initialization completed.\n");

    setGpuPointers(gpuFieldWeight, gpuFieldCoord, gpuTheorSlope, gpuFieldValues);
    gpuFieldDistribution();
    getGpuPointers(&gpuFieldWeight, &gpuFieldCoord, &gpuTheorSlope, &gpuFieldValues);

    setCpuPointers(cpuFieldWeight, cpuFieldCoord, cpuTheorSlope, cpuFieldValues);
    FieldDistribution();
    getCpuPointers(&cpuFieldWeight, &cpuFieldCoord, &cpuTheorSlope, &cpuFieldValues);

    printf(">> Checking correctness against CPU... do FieldDistribution match?\n");

    tmpTheorSlope = (double*)malloc(sizeof(double) * TSlopeLength);
    cudaMemcpy(tmpTheorSlope, &gpuTheorSlope[index2D(0, 2, TSlopeLength)], sizeof(double) * TSlopeLength, cudaMemcpyDeviceToHost);

    correct = TSlopeLength;

    for (i = 0; i < TSlopeLength; i++) {
        if (cpuTheorSlope[index2D(i, 2, TSlopeLength)] != tmpTheorSlope[i]) correct--;
    }

    free(tmpTheorSlope);

    printf("\t%.2f %% of values do actually coincide...\n\n", (float)correct * 100 / TSlopeLength);

    setGpuPointers(gpuFieldWeight, gpuFieldCoord, gpuTheorSlope, gpuFieldValues);
    cudaMemcpy(gpuTheorSlope, cpuTheorSlope, sizeof(double) * TSlopeLength * 3, cudaMemcpyHostToDevice);


    printf(">> Checking correctness against CPU... do SensiblePoints match?\n");

    //setGpuPointers(gpuFieldWeight, gpuFieldCoord, gpuTheorSlope, gpuFieldValues);
    gpuSensiblePoints(Sreal, Simag, Rreal, Rimag, MaxIters);
    getGpuPointers(&gpuFieldWeight, &gpuFieldCoord, &gpuTheorSlope, &gpuFieldValues);

    setCpuPointers(cpuFieldWeight, cpuFieldCoord, cpuTheorSlope, cpuFieldValues);
    SensiblePoints(Sreal, Simag, Rreal, Rimag, MaxIters);
    getCpuPointers(&cpuFieldWeight, &cpuFieldCoord, &cpuTheorSlope, &cpuFieldValues);

    tmpFieldWeight = (int*)malloc(sizeof(int) * Xdots * Ydots);
    cudaMemcpy(tmpFieldWeight, gpuFieldWeight, sizeof(int) * Xdots * Ydots, cudaMemcpyDeviceToHost);

    tmpFieldCoord = (double*)malloc(sizeof(double) * Xdots * Ydots * 2);
    cudaMemcpy(tmpFieldCoord, gpuFieldCoord, sizeof(double) * Xdots * Ydots * 2, cudaMemcpyDeviceToHost);

    correct = Xdots * Ydots;

    for (i = 0; i < Xdots * Ydots; i++) {
        if (fabs(cpuFieldWeight[i] - tmpFieldWeight[i]) > 0.001) correct--;
    }

    printf("\t%.2f %% of values in FieldWeight do actually coincide...\n", (float)correct * 100 / (Xdots * Ydots));

    correct = Xdots * Ydots * 2;

    for (i = 0; i < Xdots * Ydots * 2; i++) {
        if (cpuFieldCoord[i] != tmpFieldCoord[i]) correct--;
    }

    printf("\t%.2f %% of values in FieldCoord do actually coincide...\n\n", (float)correct * 100 / (Xdots * Ydots * 2));

    free(tmpFieldWeight);
    free(tmpFieldCoord);


    printf(">> Checking correctness against CPU... do FieldInit match?\n");

    setGpuPointers(gpuFieldWeight, gpuFieldCoord, gpuTheorSlope, gpuFieldValues);
    gpuFieldInit();
    getGpuPointers(&gpuFieldWeight, &gpuFieldCoord, &gpuTheorSlope, &gpuFieldValues);

    setCpuPointers(cpuFieldWeight, cpuFieldCoord, cpuTheorSlope, cpuFieldValues);
    FieldInit();
    getCpuPointers(&cpuFieldWeight, &cpuFieldCoord, &cpuTheorSlope, &cpuFieldValues);

    tmpFieldValues = (double*)malloc(sizeof(double) * Xdots * Ydots * 2);
    cudaMemcpy(tmpFieldValues, gpuFieldValues, sizeof(double) * Xdots * Ydots * 2, cudaMemcpyDeviceToHost);

    correct = Xdots * Ydots * 2;

    for (i = 0; i < Xdots * Ydots * 2; i++) {
        if (fabs(cpuFieldValues[i] - tmpFieldValues[i]) > 0.001) correct--;
    }

    printf("\n\t%.2f %% of values in FieldValues do actually coincide...\n\n", (float)correct * 100 / (Xdots * Ydots * 2));



    printf(">> Checking correctness against CPU... do Cooling match?\n");

    setGpuPointers(gpuFieldWeight, gpuFieldCoord, gpuTheorSlope, gpuFieldValues);
    gpuCooling(TimeSteps);
    getGpuPointers(&gpuFieldWeight, &gpuFieldCoord, &gpuTheorSlope, &gpuFieldValues);

    setCpuPointers(cpuFieldWeight, cpuFieldCoord, cpuTheorSlope, cpuFieldValues);
    Cooling(TimeSteps);
    getCpuPointers(&cpuFieldWeight, &cpuFieldCoord, &cpuTheorSlope, &cpuFieldValues);

    cudaMemcpy(tmpFieldValues, gpuFieldValues, sizeof(double) * Xdots * Ydots * 2, cudaMemcpyDeviceToHost);

    correct = Xdots * Ydots * 2;

    for (i = 0; i < Xdots * Ydots * 2; i++) {
        if (fabs(cpuFieldValues[i] - tmpFieldValues[i]) > 0.001) correct--;
    }

    printf("\n\t%.2f %% of values in FieldValues do actually coincide...\n\n", (float)correct * 100 / (Xdots * Ydots * 2));

    free(tmpFieldValues);

    // End Program

    cudaFree(gpuFieldWeight);
    cudaFree(gpuFieldCoord);
    cudaFree(gpuTheorSlope);
    cudaFree(gpuFieldValues);

    free(cpuFieldWeight);
    free(cpuFieldCoord);
    free(cpuTheorSlope);
    free(cpuFieldValues);

    free(MeasuredValues);

    return 0;
}




int main(int argc, char *argv[]) {

    /*test();
    return 0;*/


    clock_t t0, t1, p0, p1;

    t0 = clock();
    printf(">> Starting\n");
    int devices = 0;

    cudaError_t err = cudaGetDeviceCount(&devices);

#ifdef ACCELERATED
    if (devices > 0 && err == cudaSuccess) {
        printf(">> running on GPU\n");
    }
#endif

    // Read input file
    p0 = clock();
#ifdef ACCELERATED
    gpuInitGrid("Cooling.inp");
#else
    InitGrid("Cooling.inp");
#endif
    p1 = clock();
    fprintf(stdout, ">> InitGrid ended in %lf seconds\n", (double)(p1 - p0) / CLOCKS_PER_SEC);

    // TheorSlope(TSlopeLength,3)
    p0 = clock();
#ifdef ACCELERATED
    gpuFieldDistribution();
#else
    FieldDistribution();
#endif
    p1 = clock();
    fprintf(stdout, ">> FieldDistribution ended in %lf seconds\n",
            (double)(p1 - p0) / CLOCKS_PER_SEC);

    // FieldCoord(Xdots,Ydots,2), FieldWeight(Xdots,Ydots)
    p0 = clock();
#ifdef ACCELERATED
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
#ifdef ACCELERATED
    gpuFieldInit();
#else
    FieldInit();
#endif
    p1 = clock();
    fprintf(stdout, ">> FieldInit ended in %lf seconds\n",
            (double)(p1 - p0) / CLOCKS_PER_SEC);

    // FieldValues(Xdots,Ydots,2)
    p0 = clock();
#ifdef ACCELERATED
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

#ifdef ACCELERATED
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
