#include "class2.cuh"
__global__ void fill_dataset(dataset ds){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < ds.size) {
        generate_kernel<<<10,10>>>(ds.dev_data,100,100);        
    } 
}
void fill_dataset(dataset ds,int n){
    for (int i=0;i<n;i++){
        ds.host_data[i] = 20;
    }
}