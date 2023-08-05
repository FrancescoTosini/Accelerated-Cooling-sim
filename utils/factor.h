#include "cuda_runtime.h"
#include "cusolverDn.h"
#include "helper_cuda.h"
#include "helper_cusolver.h"

/*
 *  solve A*x = b by LU with partial pivoting
 *  result is left in d_b
 */
int linearSolverLU(
    cusolverDnHandle_t handle,
    int n,
    double *A,
    int lda,
    double *d_b)
{
    int bufferSize = 0;// size of workspace
    double *d_buffer = NULL;// device workspace for getrf
    int *d_info = NULL;// error info
    double *d_A = NULL;// device copy of A
    int *d_ipiv = NULL; // pivoting sequence
    int h_info = 0; // error info copy on host
    double start, stop;
    double time_solve;

    // cusolver does not allocate anything on its own, so you need to allocate
    // the working buffer for it. this is how you know how big it has to be
    checkCudaErrors(cusolverDnDgetrf_bufferSize(handle, n, n, (double*)A, lda, &bufferSize));

    checkCudaErrors(cudaMalloc(&d_info, sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_buffer, sizeof(double)*bufferSize));
    checkCudaErrors(cudaMalloc(&d_A, sizeof(double)*lda*n));
    checkCudaErrors(cudaMalloc(&d_ipiv, sizeof(int)*n));

    checkCudaErrors(cudaMemset(d_info, 0, sizeof(int)));

    start = second();
    
    printf("\t\t... factorizing ...\n");
    checkCudaErrors(cusolverDnDgetrf(handle, n, n, A, lda, d_buffer, d_ipiv, d_info));
    checkCudaErrors(cudaDeviceSynchronize()); // no point syncing..

    printf("\t\t\t done! fetching results...\n");
    checkCudaErrors(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));

    if (0 > h_info) {
        printf("%d-th parameter is wrong \n", -h_info);
        exit(EXIT_FAILURE);
    }

    printf("\t\t... solving ...\n");
    checkCudaErrors(cusolverDnDgetrs(handle, CUBLAS_OP_N, n, 1, A, lda, d_ipiv, d_b, n, d_info));
    checkCudaErrors(cudaDeviceSynchronize());
    stop = second();
    printf("\t\t\t done! fetching results...\n");

    time_solve = stop - start;
    fprintf (stdout, "\t\t factorizing and solving took: LU = %10.6f sec\n", time_solve);

    if (d_info  ) { checkCudaErrors(cudaFree(d_info)); }
    if (d_buffer) { checkCudaErrors(cudaFree(d_buffer)); }
    if (d_A     ) { checkCudaErrors(cudaFree(d_A)); }
    if (d_ipiv  ) { checkCudaErrors(cudaFree(d_ipiv));}

    return EXIT_SUCCESS;
}
