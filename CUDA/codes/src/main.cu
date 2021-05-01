#include <curand.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include "utils.cuh"
template __global__ void generate_kernel(unsigned short *,int,int);
int main(){
    int m = 20;
    int N = 60001;
    int n = m*N;
    
    size_t i;
    const unsigned int threadsPerBlock = 64;
    const unsigned int blockCount = (n-1)/threadsPerBlock+1;
    unsigned short *devData;
    const int mem_size = sizeof(short)*n;
    unsigned short * hostData = new unsigned short[n];
    unsigned int * hostidx = new unsigned int[n];
    unsigned int *devidx;
    
    /* Allocate n floats on device */
    int start,end;
    start= clock();
    cudaMalloc(&devData, mem_size);
    cudaMalloc(&devidx, sizeof(unsigned int)*n);
    printf("%d\n",blockCount);
    generate_kernel<<<blockCount,threadsPerBlock>>>(devData,10000,n);
    init_n_gpu_batch<<<m,1>>>(devidx,N);
    cudaMemcpy(hostData, devData, mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostidx, devidx, sizeof(unsigned int)*n, cudaMemcpyDeviceToHost);
    start= clock();
    sort_data_gpu_batch<<<m,1>>>(devData,devidx,N);   
    end = clock();
    cudaDeviceSynchronize();
    //cudaMemcpy(devData, hostData, mem_size, cudaMemcpyHostToDevice);
    //cudaMemcpy(devidx, hostidx,  sizeof(int)*n, cudaMemcpyHostToDevice);
    /*start= clock();
    sort_data_gpu_batch_pair<<<10,1>>>(devData,devidx,N);   
    end = clock();
    printf("time: %u\n",(end-start));*/
    start= clock();
    for (int i=0;i<m;i++){
        sort_data_cpu(hostData+i*N,hostidx+i*N,N);
    }   
    //sort_data_cpu(hostData,hostidx,N);
    end = clock();
    printf("time: %u\n",(end-start));
    
    for (int j=0;j<min(m,2);j++){
        for (i=0;i<100;i++){
            printf("%u ",hostData[j*10*N+hostidx[j*10*N+i]]);
        }
        printf("\n");
    }
    unsigned int * hostidx_ = new unsigned int[n];
    cudaMemcpy(hostidx_, devidx, sizeof(unsigned int)*n, cudaMemcpyDeviceToHost);
    for (int j=0;j<min(m,2);j++){
        for (i=0;i<100;i++){
            printf("%u ",hostData[j*10*N+hostidx_[j*10*N+i]]);
        }
        printf("\n");
    }
    for (int j=0;j<m;j++){
        for (int i=0;i<N;i++){
            if (hostData[j*N+hostidx[j*N+i]]!=hostData[j*N+hostidx_[j*N+i]]){
                printf("%u %u %u\n",j*N+i,hostidx[j*N+i],hostidx_[j*N+i]);
                printf("%u %u %u\n",j*N+i,hostData[j*N+hostidx[j*N+i]],hostData[j*N+hostidx_[j*N+i]]);
                exit(1);
            }
        }
    }
    
    /* Cleanup */
    cudaFree(devData);   
  
}
