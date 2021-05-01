#include <curand.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include "utils.cuh"
template __global__ void generate_kernel(unsigned short *,int,int);
class Im_a_genius{
    public:
        unsigned short** tmp_data;
        unsigned short * dev_data;

};
__global__ void generate_kernel_2d(unsigned short** data,int ub,int m,int n){
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id<m){
        const unsigned int threadsPerBlock = 1024;
        const unsigned int blockCount = (n-1)/threadsPerBlock+1;
        generate_kernel<<<blockCount,threadsPerBlock>>>(data[id],ub,n);
    }
}
__global__ void generate_kernel_batch(unsigned short* data,int ub,int m,int n){
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id<m){
        const unsigned int threadsPerBlock = 1024;
        const unsigned int blockCount = (n-1)/threadsPerBlock+1;
        generate_kernel<<<blockCount,threadsPerBlock>>>(data+id*n,ub,n);
    }
}
void test1(unsigned short* data,int ub,int m,int n){
    const unsigned int threadsPerBlock = 32;
    const unsigned int blockCount = (m-1)/threadsPerBlock+1;
    generate_kernel_batch<<<blockCount,threadsPerBlock>>>(data,ub,m,n);
}
__global__ void test2(unsigned short* data,int ub,int m,int n){
    const unsigned int threadsPerBlock = 32;
    const unsigned int blockCount = (m-1)/threadsPerBlock+1;
    generate_kernel_batch<<<blockCount,threadsPerBlock>>>(data,ub,m,n);
    
}
__global__ void test(Im_a_genius wtf,int m,int n){
    //only call once
    if ((threadIdx.x>0)||(blockIdx.x>0)){return;}
    wtf.tmp_data =new unsigned short*[m];
    int i=0;
    for (i=0;i<m;i++){
        wtf.tmp_data[i] = new unsigned short[n];
    }
    generate_kernel_2d<<<m,1>>>(wtf.tmp_data,10000,m,n);
    for (i=0;i<m;i++){
        memcpy(wtf.dev_data+i*n,wtf.tmp_data[i],sizeof(unsigned short)*n);
    }
}
template <typename T> 
void my_free(T** a){
    if ((*a)!=NULL){
        delete(*a);
        *a=NULL;
        printf("free once\n");
    }
}
void add_one(int *a){
    int * tmp = a+1;
    printf("%d\n",tmp[0]);
}
int main(int argc, char *argv[]){
    char * pend;
    int n = (int)strtol(argv[2],&pend,10);
    int m = (int)strtol(argv[2],&pend,10);
    int total = m*n;
    size_t i;
    const unsigned int threadsPerBlock = 1024;
    const unsigned int blockCount = total/threadsPerBlock+1;
    unsigned short *devData;
    const int mem_size = sizeof(short)*total;
    unsigned short * hostData = new unsigned short[total];
    /*unsigned int *devidx;
    unsigned int * hostidx = new unsigned int[total];*/
    /* Allocate n floats on device */
    int start,end;
    /*Im_a_genius wtf;
    cudaMalloc(&wtf.dev_data, mem_size);  
    start= clock();
    test<<<1,1>>>(wtf,m,n);
    end = clock();
    printf("time: %u\n",(end-start));*/
    cudaMalloc(&devData, mem_size);
    //cudaMalloc(&devidx, sizeof(int)*total);
    test1(devData,10000,1,1);
    start= clock();
    for (i=0;i<100;i++){
        test1(devData,10000,m,n);
    }    
    end = clock();
    printf("time: %u\n",(end-start));
    test2<<<1,1>>>(devData,10000,1,1);
    start= clock();
    for (i=0;i<100;i++){
        test2<<<1,1>>>(devData,10000,m,n);
    }    
    end = clock();
    printf("time: %u\n",(end-start));
    /*cudaDeviceSynchronize(); 
    cudaMemcpy(hostData, devData, mem_size, cudaMemcpyDeviceToHost);
    init_n_gpu_batch<<<m,1>>>(devidx,n);
    cudaDeviceSynchronize();     
    cudaMemcpy(hostidx, devidx, sizeof(int)*total, cudaMemcpyDeviceToHost);
    sort_data_gpu_batch<<<m,1>>>(devData,devidx,n); */   
    //cudaMemcpy(hostidx, devidx, sizeof(int)*total, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostData, devData, mem_size, cudaMemcpyDeviceToHost); 
    for (i=0;i<100;i++){
        printf("%u ",hostData[i]);
    }
    printf("\n");
    //cudaDeviceSynchronize(); 
    /* Cleanup */
    cudaFree(devData);  
  
}
