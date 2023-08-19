#ifndef SERIAL_H
#define SERIAL_H

#include <stdio.h>
#include <stdlib.h>

#include "constants.h"
#include "util.cuh"

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

void InitGrid(char *InputFile);
int LinEquSolve(double *h_A, int n, double *h_b);
int LinEquSolve_ACC(double *h_A, int n, double *h_b);
void EqsDef(double x0, double x1, double y0, double y1, int N, int LA,
            double *A, double *Rhs, double *Pts);
double Solution(double x, double y);
double *FieldDistribution();
double *FieldDistribution_mixed();
void GridDef(double x0, double x1, double y0, double y1, int N, double *Pts);
void SensiblePoints(double Ir, double Ii, double Sr, double Si, int MaxIt);
void FieldInit();
double NearestValue(double xc, double yc, int ld, double *Values);
void FieldPoints(double Diff);
void Cooling(int steps);
void RealData2ppm(int s1, int s2, double *rdata, double *vmin, double *vmax, char *name);
void Statistics(int s1, int s2, double *rdata, int s);
void Update(int xdots, int ydots, double *u1, double *u2);
#endif