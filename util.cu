#include <stdio.h>
#include <string.h>

#ifndef UTIL
#define UTIL

/* CPU-only UTIL */

int rowlen(char* riga);
int readrow(char* rg, int nc, FILE* daleg);
double MinIntVal(int s, int* a);
double MaxIntVal(int s, int* a);
double MinDoubleVal(int s, double* a);
double MaxDoubleVal(int s, double* a);
void SetIntValue(int* a, int l, int v);
void SetDoubleValue(double* a, int l, double v);

int rowlen(char* riga)
{
    int lungh;
    char c;

    lungh = strlen(riga);
    while (lungh > 0) {
        lungh--;
        c = *(riga + lungh);
        if (c == '\0') continue;
        if (c == '\40') continue;     /*  carattere spazio  */
        if (c == '\b') continue;
        if (c == '\f') continue;
        if (c == '\r') continue;
        if (c == '\v') continue;
        if (c == '\n') continue;
        if (c == '\t') continue;
        return(lungh + 1);
    }
    return(0);
}

int readrow(char* rg, int nc, FILE* daleg)
{
    int lrg;

    if (fgets(rg, nc, daleg) == NULL) return(0);
    lrg = rowlen(rg);
    if (lrg < nc) 
    {
        rg[lrg] = '\0';
        lrg++;
    }
    return(lrg);
}

double MinIntVal(int s, int* a) // OPP: can be accelerated?
{
    int v;
    int e;

    v = a[0];

    for (e = 0; e < s; e++)
    {
        if (v > a[e]) v = a[e];
    }

    return v;
}

double MaxIntVal(int s, int* a) // OPP: can be accelerated?
{
    int v;
    int e;

    v = a[0];

    for (e = 0; e < s; e++)
    {
        if (v < a[e]) v = a[e];
    }

    return v;
}

double MinDoubleVal(int s, double* a) // OPP: can be accelerated?
{
    double v;
    int e;

    v = a[0];

    for (e = 0; e < s; e++)
    {
        if (v > a[e]) v = a[e];
    }

    return v;
}

double MaxDoubleVal(int s, double* a) // OPP: can be accelerated?
{
    double v;
    int e;

    v = a[0];

    for (e = 0; e < s; e++)
    {
        if (v < a[e]) v = a[e];
    }

    return v;
}

void SetIntValue(int* a, int l, int v)
{
    int i;
    for (i = 0; i < l; i++) a[i] = v;
    return;
}

void SetDoubleValue(double* a, int l, double v)
{
    int i;
    for (i = 0; i < l; i++) a[i] = v;
    return;
}

/* ACCELERATED UTIL */

/* TODO */

#endif