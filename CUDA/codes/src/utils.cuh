#include<stdlib.h>
#include<stdio.h>
#include<time.h>
#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>
//gpu random
template<typename T1>
__global__ void generate_kernel(T1 *result,int ub,int n){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    //Copy state to local memory for efficiency 
    curandStateXORWOW localState;
    curand_init((int)clock64(), id, 0, &localState);
    // Store results
    if (id<n){
        result[id] = static_cast<T1>(curand(&localState)%ub);
        //if (id<100){printf("%u ",result[id]);}
    }    
}
__global__ void init_n_gpu_batch(unsigned int* result,int n);
//ub:upper bound, generate n unsigned less than ub
__global__ void sample_wo_rep_kernel(unsigned int* result,const int n,int m);
__global__ void sort_data_gpu(unsigned short*data,unsigned int *idx,int num);
__global__ void sort_data_gpu_batch(unsigned short*data,unsigned int *idx,int num);
__global__ void sort_data_gpu_batch_pair(unsigned short*data,unsigned int *idx,int num);
//generate m unsigned(no repeat) less than n
//cpu random
void generate(unsigned int* result,int ub);
//ub:upper bound, generate n unsigned less than ub
void sample_wo_rep(unsigned int* result,const int n,int m);
//generate m unsigned(no repeat) less than n
void sort_data_cpu(unsigned short*data,unsigned int *idx,int num);
//bubble sort data_idx based on feature-id
//data data points x FeatNum