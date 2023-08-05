#include "cuda_runtime.h"
#include "cusolverDn.h"
#include "helper_cuda.h"
#include "helper_cusolver.h"

/*
 *  solve A*x = b by LU with partial pivoting
 *  result is left in x
 */
int linearSolverLU(
    cusolverDnHandle_t handle,
    int n,
    const double *Acopy,
    int lda,
    double *d_b,
    double *d_x)
{
    int bufferSize = 0;// size of workspace
    double *d_buffer = NULL;// device workspace for getrf
    int *d_info = NULL;// error info
    double *d_A = NULL;// device copy of A
    int *d_ipiv = NULL; // pivoting sequence
    int h_info = 0; // error info copy on host
    double start, stop;
    double time_solve;

    checkCudaErrors(cusolverDnDgetrf_bufferSize(handle, n, n, (double*)Acopy, lda, &bufferSize));

    cudaDeviceSynchronize();
    printf("buffer size set as: %d", bufferSize);

    checkCudaErrors(cudaMalloc(&d_info, sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_buffer, sizeof(double)*bufferSize));
    checkCudaErrors(cudaMalloc(&d_A, sizeof(double)*lda*n));
    checkCudaErrors(cudaMalloc(&d_ipiv, sizeof(int)*n));


    // prepare a copy of A because getrf will overwrite A with L
    checkCudaErrors(cudaMemcpy(d_A, Acopy, sizeof(double)*lda*n, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemset(d_info, 0, sizeof(int)));

    start = second();
    
    printf("... factorization ...");
    checkCudaErrors(cusolverDnDgetrf(handle, n, n, d_A, lda, d_buffer, d_ipiv, d_info));
    cudaDeviceSynchronize();

    printf("\t\t done! fetching results...\n");
    checkCudaErrors(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));

    if (0 > h_info) {
        printf("%d-th parameter is wrong \n", -h_info);
        exit(1);
    }

    checkCudaErrors(cusolverDnDgetrs(handle, CUBLAS_OP_N, n, 1, d_A, lda, d_ipiv, d_b, n, d_info));
    checkCudaErrors(cudaMemcpy(d_x,d_b, sizeof(double)*n, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    stop = second();

    time_solve = stop - start;
    fprintf (stdout, "timing: LU = %10.6f sec\n", time_solve);

    if (d_info  ) { checkCudaErrors(cudaFree(d_info)); }
    if (d_buffer) { checkCudaErrors(cudaFree(d_buffer)); }
    if (d_A     ) { checkCudaErrors(cudaFree(d_A)); }
    if (d_ipiv  ) { checkCudaErrors(cudaFree(d_ipiv));}

    return EXIT_SUCCESS;
}
