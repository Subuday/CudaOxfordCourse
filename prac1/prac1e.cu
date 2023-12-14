//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_cuda.h>

__global__ void init(float *x) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    x[tid] = (float)threadIdx.x;
}

__global__ void add(float *x, float *y, float *z) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  z[tid] = x[tid] + y[tid];
}

int main(int argc, const char **argv) {
    float *h_x, *d_x1, *d_x2, *d_x3;
    int nblocks, nthreads, nsize, n;

    findCudaDevice(argc, argv);

    nblocks = 2;
    nthreads = 4;
    nsize = nblocks * nthreads;

    h_x = (float *)malloc(nsize  * sizeof(float));
    checkCudaErrors(cudaMalloc((void **)&d_x1, nsize * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_x2, nsize * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_x3, nsize * sizeof(float)));

    init<<<nblocks, nthreads>>>(d_x1);
    getLastCudaError("init execution failed\n");
    init<<<nblocks, nthreads>>>(d_x2);
    getLastCudaError("init execution failed\n");
    add<<<nblocks, nthreads>>>(d_x1, d_x2, d_x3);
    getLastCudaError("add execution failed\n");

    checkCudaErrors(cudaMemcpy(h_x, d_x3, nsize * sizeof(float), cudaMemcpyDeviceToHost));
    for (n = 0; n < nsize; n++) printf(" n,  x  =  %d  %f \n", n, h_x[n]);

    // free memory
    checkCudaErrors(cudaFree(d_x1));
    checkCudaErrors(cudaFree(d_x2));
    checkCudaErrors(cudaFree(d_x3));
    free(h_x);

    cudaDeviceReset();

    return 0;
}