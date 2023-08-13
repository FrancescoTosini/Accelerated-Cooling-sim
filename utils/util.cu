#include "util.cuh"

int rowlen(char *riga) {
    int lungh;
    char c;

    lungh = strlen(riga);
    while (lungh > 0) {
        lungh--;
        c = *(riga + lungh);
        if (c == '\0')
            continue;
        if (c == '\40')
            continue; /*  carattere spazio  */
        if (c == '\b')
            continue;
        if (c == '\f')
            continue;
        if (c == '\r')
            continue;
        if (c == '\v')
            continue;
        if (c == '\n')
            continue;
        if (c == '\t')
            continue;
        return (lungh + 1);
    }
    return (0);
}

int readrow(char *rg, int nc, FILE *daleg) {
    int lrg;

    if (fgets(rg, nc, daleg) == NULL)
        return (0);
    lrg = rowlen(rg);
    if (lrg < nc) {
        rg[lrg] = '\0';
        lrg++;
    }
    return (lrg);
}

double MinIntVal(int s, int *a) // OPP: can be accelerated?
{
    int v;
    int e;

    v = a[0];

    for (e = 0; e < s; e++) {
        if (v > a[e])
            v = a[e];
    }

    return v;
}

double MaxIntVal(int s, int *a) // OPP: can be accelerated?
{
    int v;
    int e;

    v = a[0];

    for (e = 0; e < s; e++) {
        if (v < a[e])
            v = a[e];
    }

    return v;
}

double MinDoubleVal(int s, double *a) // OPP: can be accelerated?
{
    double v;
    int e;

    v = a[0];

    for (e = 0; e < s; e++) {
        if (v > a[e])
            v = a[e];
    }

    return v;
}

double MaxDoubleVal(int s, double *a) // OPP: can be accelerated?
{
    double v;
    int e;

    v = a[0];

    for (e = 0; e < s; e++) {
        if (v < a[e])
            v = a[e];
    }

    return v;
}

void SetIntValue(int *a, int l, int v) {
    int i;
    for (i = 0; i < l; i++)
        a[i] = v;
    return;
}

void SetDoubleValue(double *a, int l, double v) {
    int i;
    for (i = 0; i < l; i++)
        a[i] = v;
    return;
}

void RealData2ppm(int s1, int s2, double *rdata, double *vmin, double *vmax, char *name) {
    /* Simple subroutine to dump integer data in a PPM format */

    int cm[3][256]; /* R,G,B, Colour Map */
    FILE *ouni, *ColMap;
    int i, j, rc, vp, vs;
    double rmin, rmax;
    char fname[80], jname[80], command[80];

    /* Load color map: 256 colours */
    ColMap = fopen("ColorMap.txt", "r");
    if (ColMap == NULL) {
        fprintf(stderr, "(Error@RealData2ppm) >> Cannot open ColorMap.txt\n");
        exit(-1);
    }
    for (i = 0; i < 256; i++) {
        if (fscanf(ColMap, " %3d %3d %3d", &cm[0][i], &cm[1][i], &cm[2][i]) < 3) {
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
    if (!ouni)
        fprintf(stderr, "(Error@RealData2ppm) >> write access to file %s\n", fname);

    /*  Magic code */
    fprintf(ouni, "P3\n");

    /*  Dimensions */
    fprintf(ouni, "%d %d\n", s1, s2);

    /*  Maximum value */
    fprintf(ouni, "255\n");

    /*  Values from 0 to 255 */
    rmin = MinDoubleVal(s1 * s2, rdata);
    rmax = MaxDoubleVal(s1 * s2, rdata);

    if ((*vmin == *vmax) && (*vmin == (double)0.0)) {
        *vmin = rmin;
        *vmax = rmax;
    } else {
        rmin = *vmin;
        rmax = *vmax;
    }

    vs = 0;
    for (i = 0; i < s1; i++) {
        for (j = 0; j < s2; j++) {
            vp = (int)((rdata[i + (j * s1)] - rmin) * 255.0 / (rmax - rmin));

            if (vp < 0)
                vp = 0;
            if (vp > 255)
                vp = 255;

            vs++;

            fprintf(ouni, " %3.3d %3.3d %3.3d", cm[0][vp], cm[1][vp],
                    cm[2][vp]);

            if (vs >= 10) {
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

void Statistics(int s1, int s2, double *rdata, int step) {
    double mnv, mv, mxv, sd;
    int i, j;

    // OPP: Can mean value and standard deviation be computed together?

    // Compute MEAN VALUE
    mv = 0.0;
    mnv = mxv = rdata[0];
    for (i = 0; i < s1; i++) {
        for (j = 0; j < s2; j++) {
            mv = mv + rdata[i + (j * s1)];
            if (mnv > rdata[i + (j * s1)])
                mnv = rdata[i + (j * s1)];
            if (mxv < rdata[i + (j * s1)])
                mxv = rdata[i + (j * s1)];
        }
    }
    mv = mv / (double)(s1 * s2);

    // Compute STANDARD DEVIATION
    sd = 0.0;
    for (i = 0; i < s1; i++) {
        for (j = 0; j < s2; j++) {
            sd = sd + (rdata[i + (j * s1)] - mv) * (rdata[i + (j * s1)] - mv);
        }
    }
    sd = sqrt(sd / (double)(s1 * s2));

    fprintf(
        stdout,
        ">> Step %4d: min, mean, max, std = %12.3e, %12.3e, %12.3e, %12.3e\n",
        step, mnv, mv, mxv, sd);

    return;
}

/* ACCELERATED UTIL */
void gpuStatistics(int s1, int s2, double *rdata, double *tmp, int step) {
    double mnv, mv, mxv, sd;

    double *tmpMin;
    double *tmpMax;
    double *tmpMean;
    double *tmpStd;

    cudaError_t err;

    int reduceLayer = (s1 * s2 + 1) / 2;

    tmpMin = tmp;
    tmpMax = &tmpMin[reduceLayer];
    tmpMean = &tmpMax[reduceLayer];
    tmpStd = &tmpMean[reduceLayer];

    gpuStatisticsKernel<<<256, 128>>>(tmpMin, tmpMax, tmpMean, tmpStd, rdata, s1 * s2, reduceLayer);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
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

__global__ void gpuStatisticsKernel(double *tmpMin, double *tmpMax, double *tmpMean, double *tmpStd, double *Values, int len, int reduceLayer) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int i;
    double a, b;
    int layerLength = reduceLayer;

    int last;

    for (i = id; i < layerLength; i += stride) {
        if ((2 * i + 1) < len) {
            a = Values[2 * i];
            b = Values[2 * i + 1];

            tmpMin[i] = (a < b) ? a : b;
            tmpMax[i] = (a > b) ? a : b;
            tmpMean[i] = a + b;
        } else {
            tmpMin[i] = Values[2 * i];
            tmpMax[i] = Values[2 * i];
            tmpMean[i] = Values[2 * i];
        }
    }

    last = layerLength % 2;
    layerLength = (layerLength + 1) / 2;

    __syncthreads();

    while (layerLength > 1) {
        for (i = id; i < layerLength; i += stride) {
            if (i < layerLength - last) {
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

        for (i = id; i < layerLength; i += stride) {
            tmpMin[i] = tmpMin[2 * i];
            tmpMax[i] = tmpMax[2 * i];
            tmpMean[i] = tmpMean[2 * i];
        }

        last = layerLength % 2;
        layerLength = (layerLength + 1) / 2;

        __syncthreads();
    }

    if (id == 0) {
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

    for (i = id; i < layerLength; i += stride) {
        if ((2 * i + 1) < len) {
            a = Values[2 * i];
            b = Values[2 * i + 1];

            a = (a - mv) * (a - mv);
            b = (b - mv) * (b - mv);

            tmpStd[i] = a + b;
        } else {
            a = Values[2 * i];
            a = (a - mv) * (a - mv);

            tmpStd[i] = a;
        }
    }

    last = layerLength % 2;
    layerLength = (layerLength + 1) / 2;

    __syncthreads();

    while (layerLength > 1) {
        for (i = id; i < layerLength; i += stride) {
            if (i < layerLength - last) {
                a = tmpStd[2 * i];
                b = tmpStd[2 * i + 1];
                tmpStd[2 * i] = a + b;
            }
        }

        __syncthreads();

        for (i = id; i < layerLength; i += stride)
            tmpStd[i] = tmpStd[2 * i];

        last = layerLength % 2;
        layerLength = (layerLength + 1) / 2;

        __syncthreads();
    }

    if (id == 0) {
        rStd = sqrt((tmpStd[0] + tmpStd[1]) / (double)len);
    }
}

/* TODO */
