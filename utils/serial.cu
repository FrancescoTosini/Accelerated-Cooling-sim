#include "serial.h"

void InitGrid(char *InputFile) {
    /* Output:
    !  MeasuredValues(:,3) - values read from input file
    !  Initialization of FieldWeight(Xdots,Ydots) and FieldCoord(Xdots,Ydots,2)
    */

    int valrows, st;
    char filerow[80];
    FILE *inpunit;

    fprintf(stdout, ">> Initializing grid ...\n");

    inpunit = fopen(InputFile, "r");
    if (!inpunit) {
        fprintf(stderr, "(Error) >>> Cannot access file %s\n", InputFile);
        exit(-1);
    }

    // Read measured values
    NumInputValues = 0;
    valrows = 0;
    while (1) {
        st = readrow(filerow, 80, inpunit);
        if (filerow[0] == '#')
            continue;
        if (NumInputValues <= 0) {
            if (sscanf(filerow, "  %d", &NumInputValues) < 1) {
                if (NumInputValues <= 0) {
                    fprintf(stderr, "(Error) >> there seems to be %d input values...\n", NumInputValues);
                    exit(-1);
                }
            } else {
                MeasuredValues = (double *)malloc(sizeof(double) * NumInputValues * 3);
                if (MeasuredValues == NULL) {
                    fprintf(stderr, "(Error) >> Cannot allocate MeasuredValues[%d,3] :(\n", NumInputValues);
                    exit(-1);
                }
            }
        } else {
            if (sscanf(filerow, "%lf %lf %lf",
                       &MeasuredValues[index2D(valrows, 0, NumInputValues)], // X coord
                       &MeasuredValues[index2D(valrows, 1, NumInputValues)], // Y coord
                       &MeasuredValues[index2D(valrows, 2, NumInputValues)]) // Measured value
                < 3) {
                fprintf(stderr, "(Error) >>> something went wrong while reading MeasuredValues(%d,*)", valrows);
                exit(-1);
            }
            valrows++;
            if (valrows >= NumInputValues)
                break;
        }
    }

    /* Create and initialize FieldWeight */
    FieldWeight = (int *)malloc(sizeof(int) * Xdots * Ydots);
    if (FieldWeight == NULL) {
        fprintf(stderr, "(Error) >> Cannot allocate FieldWeight[%d,%d]\n", Xdots, Ydots);
        exit(-1);
    }
    SetIntValue(FieldWeight, Xdots * Ydots, 0); // OPP: you can use calloc?

    /* Create and initialize FieldCoord */
    FieldCoord = (double *)malloc(sizeof(double) * Xdots * Ydots * 2);
    if (FieldCoord == NULL) {
        fprintf(stderr, "(Error) >> Cannot allocate FieldCoord[%d,%d,2]\n", Xdots, Ydots);
        exit(-1);
    }
    SetDoubleValue(FieldCoord, Xdots * Ydots * 2,
                   (double)0); // OPP: you can use calloc?

    /* Now read Sreal, Simag, Rreal, Rimag */
    Sreal = Simag = Rreal = Rimag = 0.0;
    while (1) {
        if (readrow(filerow, 80, inpunit) < 1) {
            fprintf(stderr, "(Error) >> Cannot read Sreal from input file.\n");
            exit(-1);
        }
        if (filerow[0] == '#')
            continue;
        if (sscanf(filerow, "%lf", &Sreal) < 1) {
            fprintf(stderr, "(Error) >> Cannot read Sreal from string.\n");
            exit(-1);
        }
        if (fscanf(inpunit, "%lf", &Simag) < 1) {
            fprintf(stderr, "(Error) >> Cannot read Simag from input file.\n");
            exit(-1);
        }
        if (fscanf(inpunit, "%lf", &Rreal) < 1) {
            fprintf(stderr, "(Error) >> Cannot read Rreal from input file.\n");
            exit(-1);
        }
        if (fscanf(inpunit, "%lf", &Rimag) < 1) {
            fprintf(stderr, "(Error) >> Cannot read Rimag from input file.\n");
            exit(-1);
        }
        break;
    }

    /* Now read MaxIters */
    MaxIters = 0;
    while (1) {
        if (readrow(filerow, 80, inpunit) < 1) {
            fprintf(stderr, "(Error) >> Cannot read MaxIters from input file.\n");
            exit(-1);
        }
        if (filerow[0] == '#' || rowlen(filerow) < 1)
            continue;
        if (sscanf(filerow, "%d", &MaxIters) < 1) {
            fprintf(stderr, "(Error) >> Cannot read MaxIters from string.\n");
            exit(-1);
        }
        break;
    }

    /* Now read TimeSteps */
    TimeSteps = 0;
    while (1) {
        if (readrow(filerow, 80, inpunit) < 1) {
            fprintf(stderr, "(Error) >> Cannot read MaxIters from input file.\n");
            exit(-1);
        }
        if (filerow[0] == '#' || rowlen(filerow) < 1)
            continue;
        if (sscanf(filerow, "%d", &TimeSteps) < 1) {
            fprintf(stderr, "(Error) >> Cannot read TimeSteps from string.\n");
            exit(-1);
        }
        break;
    }

    fclose(inpunit);
    return;
}

double *FieldDistribution() {
    /*
    !  Compute theoretical value distribution of the perturbing field
    !  Output: TheorSlope(TSlopeLength,3) - theoretical field distribution
    function
    */
    double *CoeffMatrix, *B;
    double x0, y0, x1, y1;
    double t0, t1;

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

    Nm1 = N - 1; // Grid points minus boundary
    Mm1 = M - 1; // Grid points minus boundary

    LA = Mm1 * Nm1; // unknown points
    TSlopeLength = LA;

    CoeffMatrix = (double *)malloc(sizeof(double) * LA * LA);
    TheorSlope = (double *)malloc(sizeof(double) * TSlopeLength * 3);
    B = (double *)malloc(sizeof(double) * LA);

    if (CoeffMatrix == NULL || TheorSlope == NULL || B == NULL) {
        fprintf(
            stderr,
            "(Error) >> Cannot allocate memory. \nCoeffMatrix: %p; TheorSlope: "
            "%p, B: %p\n",
            CoeffMatrix, TheorSlope, B);
        exit(-1);
    }

    GridDef(x0, x1, y0, y1, N, TheorSlope);
    EqsDef(x0, x1, y0, y1, N, LA, CoeffMatrix, B, TheorSlope);
    // GridDef(x0, x1, y0, y1, N, TheorSlope);

    rc = LinEquSolve(CoeffMatrix, LA, B);
    if (rc != 0)
        exit(-1);

    for (i = 0; i < LA; i++)
        TheorSlope[index2D(i, 2, TSlopeLength)] = B[i]; // OPP: why not use memcpy?

    free(CoeffMatrix);
    // free(B);

    return B;
}

double *FieldDistribution_mixed() {
    /*
    !  Compute theoretical value distribution of the perturbing field
    !  Output: TheorSlope(TSlopeLength,3) - theoretical field distribution
    function
    */
    double *CoeffMatrix, *B;
    double x0, y0, x1, y1;
    double t0, t1;

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

    Nm1 = N - 1; // Grid points minus boundary
    Mm1 = M - 1; // Grid points minus boundary

    LA = Mm1 * Nm1; // unknown points
    TSlopeLength = LA;

    CoeffMatrix = (double *)malloc(sizeof(double) * LA * LA);
    TheorSlope = (double *)malloc(sizeof(double) * TSlopeLength * 3);
    B = (double *)malloc(sizeof(double) * LA);

    if (CoeffMatrix == NULL || TheorSlope == NULL || B == NULL) {
        fprintf(
            stderr,
            "(Error) >> Cannot allocate memory. \nCoeffMatrix: %p; TheorSlope: "
            "%p, B: %p\n",
            CoeffMatrix, TheorSlope, B);
        exit(-1);
    }

    GridDef(x0, x1, y0, y1, N, TheorSlope);
    EqsDef(x0, x1, y0, y1, N, LA, CoeffMatrix, B, TheorSlope);

    double *result_seq = B;
    double *result_acc = NULL;
    result_acc = (double *)malloc(sizeof(double) * LA);
    memcpy(result_acc, B, sizeof(double) * LA);

    t0 = clock();
    double *d_A, *d_b;
    cudaMalloc(&d_A, sizeof(double) * LA * LA);
    cudaMemcpy(d_A, CoeffMatrix, sizeof(double) * LA * LA, cudaMemcpyHostToDevice);
    cudaMalloc(&d_b, sizeof(double) * LA);
    cudaMemcpy(d_b, B, sizeof(double) * LA, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    rc = LinEquSolve_ACC(d_A, LA, d_b);
    t1 = clock();
    cudaMemcpy(result_acc, d_b, sizeof(double) * LA, cudaMemcpyDeviceToHost);
    fprintf(stdout, "\t>> LinEquSolve_CUDA took %lf seconds\n", (double)(t1 - t0) / CLOCKS_PER_SEC);
    t0 = clock();
    rc = LinEquSolve(CoeffMatrix, LA, result_seq);
    t1 = clock();
    fprintf(stdout, "\t>> LinEquSolve_seq took %lf seconds\n", (double)(t1 - t0) / CLOCKS_PER_SEC);

    for (i = 0; i < LA; i++) {
        result_seq[i] -= result_acc[i];
    }
    double ninf = -1;
    for (i = 0; i < LA; i++)
        ninf = max(ninf, abs(result_seq[i]));

    printf("---------maximum difference between solutions is %f. Good enough?\n", ninf);
    if (rc != 0)
        exit(-1);

    for (i = 0; i < LA; i++)
        TheorSlope[index2D(i, 2, TSlopeLength)] = B[i]; // OPP: why not use memcpy?

    free(CoeffMatrix);
    // free(B);

    return B;
}

void SensiblePoints(double Ir, double Ii, double Sr, double Si, int MaxIt) {
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

    for (iy = 0; iy < Ydots; iy++) {
        for (ix = 0; ix < Xdots; ix++) {
            ca = Xinc * ix + Ir;
            cb = Yinc * iy + Ii;
            FieldCoord[index3D(ix, iy, 0, Xdots, Ydots)] = ca;
            FieldCoord[index3D(ix, iy, 1, Xdots, Ydots)] = cb;
            rad = ca * ca * ((double)1.0 + (cb / ca) * (cb / ca));
            zan = 0.0;
            zbn = 0.0;
            for (iz = 1; iz <= MaxIt; iz++) {
                if (rad > (double)4.0)
                    break;
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

void FieldInit() {
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
    FieldValues = (double *)malloc(sizeof(double) * Xdots * Ydots * 2);
    if (FieldValues == NULL) {
        fprintf(stderr, "(Error@FieldInit) >> Cannot allocate FieldValues[%d,%d,2]\n", Xdots, Ydots);
        exit(-1);
    }
    SetDoubleValue(FieldValues, Xdots * Ydots * 2,
                   (double)0); // OPP: you can use calloc?

    /* Allocate DiffValues */
    DiffValues = (double *)malloc(sizeof(double) * NumInputValues);
    if (DiffValues == NULL) {
        fprintf(stderr, "(Error@FieldInit) >> Cannot allocate DiffValues[%d]\n", NumInputValues);
        exit(-1);
    }
    SetDoubleValue(DiffValues, NumInputValues,
                   (double)0.0); // OPP: you can use calloc?

    /* Compute discrepancy between Measured and Theoretical value */
    DiscrValue = 0.0;
    for (rv = 0; rv < NumInputValues; rv++) {
        xc = MeasuredValues[index2D(rv, 0, NumInputValues)];
        yc = MeasuredValues[index2D(rv, 1, NumInputValues)];

        // TheorSlope is computed on the basis of a coarser grid, so look for the
        // best values near xc, yc coordinates
        sv = NearestValue(xc, yc, TSlopeLength, TheorSlope);
        ev = MeasuredValues[index2D(rv, 2, NumInputValues)];

        DiffValues[rv] = ev - sv;
        DiscrValue += ev - sv;
    }
    DiscrValue = DiscrValue / (double)NumInputValues;

    // Compute standard deviation
    sd = 0.0;
    for (rv = 0; rv < NumInputValues; rv++)
        sd = sd + (DiffValues[rv] - DiscrValue) * (DiffValues[rv] - DiscrValue);
    sd = sqrt(sd / (double)NumInputValues);

    // Print statistics
    fprintf(
        stdout,
        "\t...Number of Points, Mean value, Standard deviation = %d, %12.3e, "
        "%12.3e\n",
        NumInputValues, DiscrValue, sd);

    // Compute FieldValues stage 1
    FieldPoints(DiscrValue);

    free(DiffValues);

    return;
}

void Cooling(int steps) {
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
    // RealData2ppm(Xdots, Ydots, &FieldValues[index3D(0, 0, 0, Xdots, Ydots)], &vmin, &vmax, fname);
    Statistics(Xdots, Ydots, &FieldValues[index3D(0, 0, 0, Xdots, Ydots)], 0);

    iz = 1;
    for (it = 1; it <= steps; it++) {
        // Update the value of grid points
        Update(Xdots, Ydots, &FieldValues[index3D(0, 0, iz - 1, Xdots, Ydots)],
               &FieldValues[index3D(0, 0, 2 - iz, Xdots, Ydots)]);
        iz = 3 - iz;

        // Print and show results
        sprintf(fname, "FieldValues%4.4d", it);
        // if (it % 4 == 0) RealData2ppm(Xdots, Ydots, &FieldValues[index3D(0, 0, iz - 1, Xdots, Ydots)], &vmin, &vmax, fname);
        Statistics(Xdots, Ydots, &FieldValues[index3D(0, 0, iz - 1, Xdots, Ydots)], it);
    }

    return;
}

void GridDef(double x0, double x1, double y0, double y1, int N, double *Pts) {
    double x, y, dx, dy;
    int i, j, np, Mm1, Nm1;

    Mm1 = sqrt((double)Xdots) - 1;
    Nm1 = sqrt((double)Ydots) - 1;
    dx = (x1 - x0) / (double)N;
    dy = (y1 - y0) / (double)N;

    np = -1;
    for (i = 0; i < Mm1; i++) {
        for (j = 0; j < Nm1; j++) {
            np++;
            if (np > Mm1 * Nm1) {
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

void EqsDef(double x0, double x1, double y0, double y1, int N, int LA, double *A, double *Rhs, double *Pts) {
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
    SetDoubleValue(Rhs, LA, (double)0);    // OPP: you can use calloc?

    for (np = 0; np < LA; np++) {
        x = Pts[index2D(np, 0, TSlopeLength)];
        y = Pts[index2D(np, 1, TSlopeLength)];

        A[index2D(np, np, LA)] = -4.0;

        Rhs[np] = (x + y) * dx * dy;

        // define Eps function of grid dimensions
        Eps = (dx + dy) / 20.0;

        // where is P(x-dx,y) ?
        if (fabs((x - dx) - x0) < Eps)
            Rhs[np] = Rhs[np] - Solution(x0, y);
        else {
            // Find pos = position of P(x-dx,y)
            pos = np - Nm1;
            if (fabs(Pts[index2D(pos, 0, TSlopeLength)] - (x - dx)) > Eps) {
                fprintf(stderr,
                        "(Error@EqsDef) >> x-dx: pos, np, d = %d %d %lf\n", pos,
                        np,
                        fabs(Pts[index2D(pos, 0, TSlopeLength)] - (x - dx)));
                exit(-1);
            }
            A[index2D(np, pos, LA)] = 1.0;
        }

        // where is P(x+dx,y) ?
        if (fabs((x + dx) - x1) < Eps)
            Rhs[np] = Rhs[np] - Solution(x1, y);
        else {
            // Find pos = position of P(x+dx,y)
            pos = np + Nm1;
            if (fabs(Pts[index2D(pos, 0, TSlopeLength)] - (x + dx)) > Eps) {
                fprintf(stderr, "(Error@EqsDef) >> x+dx: %lf\n",
                        fabs(Pts[index2D(pos, 0, TSlopeLength)] - (x + dx)));
                exit(-1);
            }
            A[index2D(np, pos, LA)] = 1.0;
        }

        // where is P(x,y-dy) ?
        if (fabs((y - dy) - y0) < Eps)
            Rhs[np] = Rhs[np] - Solution(x, y0);
        else {
            // Find pos = position of P(x,y-dy)
            pos = np - 1;
            if (fabs(Pts[index2D(pos, 1, TSlopeLength)] - (y - dy)) > Eps) {
                fprintf(stderr, "(Error@EqsDef) >> y-dy: %lf\n",
                        fabs(Pts[index2D(pos, 1, TSlopeLength)] - (y - dy)));
                exit(-1);
            }
            A[index2D(np, pos, LA)] = 1.0;
        }

        // where is P(x,y+dy) ?
        if (fabs((y + dy) - y1) < Eps)
            Rhs[np] = Rhs[np] - Solution(x, y1);
        else {
            // Find pos = position of P(x,y-dy)
            pos = np + 1;
            if (fabs(Pts[index2D(pos, 1, TSlopeLength)] - (y + dy)) > Eps) {
                fprintf(stderr, "(Error@EqsDef) >> y+dy: %lf\n",
                        fabs(Pts[index2D(pos, 1, TSlopeLength)] - (y + dy)));
                exit(-1);
            }
            A[index2D(np, pos, LA)] = 1.0;
        }
    }
    return;
}

double Solution(double x, double y) {
    return ((x * x * x) + (y * y * y)) / (double)6.0;
}

int LinEquSolve(double *a, int n, double *b) {
    /* Gauss-Jordan elimination algorithm */
    int i, j, k, l, icol, irow;
    int *indcol, *indrow, *ipiv;
    double bigger, temp;

    /* Allocate indcol */
    indcol = (int *)malloc(sizeof(int) * n);
    if (indcol == NULL) {
        fprintf(stderr, "(Error@LinEquSolve) >> Cannot allocate indcol[%d]\n", n);
        return (-1);
    }

    /* Allocate indrow */
    indrow = (int *)malloc(sizeof((int)1) * n);
    if (indrow == NULL) {
        fprintf(stderr, "(Error@LinEquSolve) >> Cannot allocate indrow[%d]\n", n);
        return (-1);
    }

    /* Allocate ipiv */
    ipiv = (int *)malloc(sizeof((int)1) * n);
    if (ipiv == NULL) {
        fprintf(stderr, "(Error@LinEquSolve) >> Cannot allocate ipiv[%d]\n", n);
        return (-1);
    }
    SetIntValue(ipiv, n, 0); // OPP: you can use calloc?

    /* Actual algorithm */

    for (i = 0; i < n; i++) {
        bigger = 0.0;

        for (j = 0; j < n; j++) {
            if (ipiv[j] != 1) {
                for (k = 0; k < n; k++) {
                    if (ipiv[k] == 0 && bigger <= fabs(a[index2D(j, k, n)])) {
                        bigger = fabs(a[index2D(j, k, n)]);
                        irow = j;
                        icol = k;
                    }
                }
            }
        }

        ipiv[icol] = ipiv[icol] + 1;

        if (irow != icol) {
            for (l = 0; l < n; l++) {
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

        if (a[index2D(icol, icol, n)] == 0.0) {
            fprintf(stderr, "(Error@LinEquSolve) >> a(%d,%d): singular matrix!", icol, icol);
            return -2;
        }

        temp = (double)1.0 / a[index2D(icol, icol, n)];
        a[index2D(icol, icol, n)] = 1.0;

        for (l = 0; l < n; l++)
            a[index2D(icol, l, n)] = a[index2D(icol, l, n)] * temp;

        b[icol] = b[icol] * temp;

        for (l = 0; l < n; l++) {
            if (l != icol) {
                temp = a[index2D(l, icol, n)];
                a[index2D(l, icol, n)] = 0.0;
                for (k = 0; k < n; k++) {
                    a[index2D(l, k, n)] = a[index2D(l, k, n)] - a[index2D(icol, k, n)] * temp;
                }
                b[l] = b[l] - b[icol] * temp;
            }
        }
    }

    for (l = n - 1; l >= 0; l--) {
        if (indrow[l] != indcol[l]) {
            for (k = 0; k < n; k++) {
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

double NearestValue(double xc, double yc, int ld, double *Values) {

    // look for the best values near xc, yc coordinates
    double v;

    double d, md; // minimum distance
    int np;       // number of nearest points
    int i;

    md = ((xc - Values[index2D(0, 0, ld)]) * (xc - Values[index2D(0, 0, ld)])) + ((yc - Values[index2D(0, 1, ld)]) * (yc - Values[index2D(0, 1, ld)]));

    // Compute lowest distance
    for (i = 0; i < ld; i++) {
        d = ((xc - Values[index2D(i, 0, ld)]) * (xc - Values[index2D(i, 0, ld)])) +
            ((yc - Values[index2D(i, 1, ld)]) * (yc - Values[index2D(i, 1, ld)]));
        if (md > d)
            md = d;
    }

    np = 0;
    v = 0.0;

    // Compute nearest value
    for (i = 0; i < ld; i++) {
        d = ((xc - Values[index2D(i, 0, ld)]) * (xc - Values[index2D(i, 0, ld)])) +
            ((yc - Values[index2D(i, 1, ld)]) * (yc - Values[index2D(i, 1, ld)]));
        if (md == d) {
            // add contributed value
            np = np + 1;
            v = v + Values[index2D(i, 2, ld)];
        }
    }

    // mean value
    v = v / (double)np;

    return v;
}

void FieldPoints(double Diff) {
    int ix, iy;
    double xc, yc, sv;
    int rmin, rmax;

    rmax = MaxIntVal(Xdots * Ydots, FieldWeight);
    rmin = MinIntVal(Xdots * Ydots, FieldWeight);

    printf("-----> rmin = %d, rmax = %d on CPU\n", rmin, rmax);

    for (iy = 0; iy < Ydots; iy++) {
        for (ix = 0; ix < Xdots; ix++) {
            xc = FieldCoord[index3D(ix, iy, 0, Xdots, Ydots)];
            yc = FieldCoord[index3D(ix, iy, 1, Xdots, Ydots)];

            // Compute effects of field in every point
            sv = NearestValue(xc, yc, TSlopeLength, TheorSlope);
            FieldValues[index3D(ix, iy, 0, Xdots, Ydots)] = 293.16 + 80 * (Diff + sv) * (FieldWeight[index2D(ix, iy, Xdots)] - rmin) / (rmax - rmin);
        }
    }

    // Copy initial status
    // OPP: use memcpy?
    for (iy = 0; iy < Ydots; iy++) {
        for (ix = 0; ix < Xdots; ix++) {
            FieldValues[index3D(ix, iy, 1, Xdots, Ydots)] = FieldValues[index3D(ix, iy, 0, Xdots, Ydots)];
        }
    }

    return;
}

void Update(int xdots, int ydots, double *u1, double *u2) {
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

    for (j = 0; j < ydots - 1; j++) {
        for (i = 0; i < xdots - 1; i++) {
            if (i <= 0 || i >= xdots - 1) {
                u2[index2D(i, j, xdots)] = u1[index2D(i, j, xdots)];
                continue;
            }

            if (j <= 0 || j >= ydots - 1) {
                u2[index2D(i, j, xdots)] = u1[index2D(i, j, xdots)];
                continue;
            }

            u2[index2D(i, j, xdots)] = CX * (u1[index2D((i - 1), j, xdots)] + u1[index2D((i + 1), j, xdots)] + dgx * u1[index2D((i + 1), j, xdots)]) + CY * (u1[index2D(i, (j - 1), xdots)] + u1[index2D(i, (j + 1), xdots)] + dgy * u1[index2D(i, j, xdots)]);
        }
    }

    for (j = 0; j < ydots - 1; j++) {
        u2[index2D(0, j, xdots)] = u2[index2D(1, j, xdots)];
        u2[index2D(Xdots - 1, j, xdots)] = u2[index2D(Xdots - 2, j, xdots)];
    }

    for (i = 0; i < xdots - 1; i++) {
        u2[index2D(i, 0, xdots)] = u2[index2D(i, 1, xdots)];
        u2[index2D(i, Ydots - 1, xdots)] = u2[index2D(i, Ydots - 2, xdots)];
    }

    return;
}
