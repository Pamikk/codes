#include "class2.cuh"
void dataset::test_gpu(int n){
    cudaMalloc(&dev_data,sizeof(int)*n);
    fill_dataset<<<10,10>>>(*this);
    cudaDeviceSynchronize();
}
void dataset::test_cpu(int n){
    host_data = new int[n];
    printf("????\n");
    fill_dataset(*this,20);
}