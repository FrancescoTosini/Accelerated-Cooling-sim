#ifndef UTIL
#define UTIL

// #include "parallel.cuh"
#include <stdio.h>
#include <string.h>

#define index2D(i, j, LD1) i + ((j)*LD1) // element position in 2-D arrays
#define index3D(i, j, k, LD1, LD2) \
    i + ((j)*LD1) + ((k)*LD1 * LD2) // element position in 3-D arrays

/* CPU-only UTIL */

int rowlen(char *riga);
int readrow(char *rg, int nc, FILE *daleg);
double MinIntVal(int s, int *a);
double MaxIntVal(int s, int *a);
double MinDoubleVal(int s, double *a);
double MaxDoubleVal(int s, double *a);
void SetIntValue(int *a, int l, int v);
void SetDoubleValue(double *a, int l, double v);
void RealData2ppm(int s1, int s2, double *rdata, double *vmin, double *vmax, char *name);
void Statistics(int s1, int s2, double *rdata, int step);
#endif