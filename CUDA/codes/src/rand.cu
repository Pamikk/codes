#include <curand.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include "utils.cuh"
template __global__ void generate_kernel(unsigned *,int);
int main()
{
  size_t n = 1000;
    size_t i;
    const unsigned int threadsPerBlock = 64;
    const unsigned int blockCount = 64;
    unsigned int *devData;
    const int mem_size = sizeof(unsigned)*n;
    unsigned int * hostData = (unsigned*) malloc(mem_size);

    /* Allocate n floats on device */
    cudaMalloc((unsigned int**)&devData, mem_size);
    generate_kernel<<<10,10>>>(devData,100);
    cudaDeviceSynchronize();     
    cudaMemcpy(hostData, devData, mem_size, cudaMemcpyDeviceToHost);
    for(i = 0; i < 10; i++) {
        printf("%u ", hostData[i]);
    }
    printf("\n");
    /* Cleanup */
    cudaFree(devData);   
  
}