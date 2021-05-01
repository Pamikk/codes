#include "utils.cuh"
#include "class1.cuh"
template __global__ void generate_kernel(int *,int,int);
__global__ void fill_dataset(dataset ds);
void fill_dataset(dataset ds,int n);