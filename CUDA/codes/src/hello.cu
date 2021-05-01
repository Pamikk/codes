#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand.h>
#include <cuda.h>
# define CUDA_CALL ( x) do { if (( x) != cudaSuccess ) { \
  printf (" Error at %s :% d \n" , __FILE__ , __LINE__ ) ;\
  return EXIT_FAILURE ;}} while (0)
  # define CURAND_CALL (x) do { if (( x ) != CURAND_STATUS_SUCCESS ) { \
  printf (" Error at %s :% d \n" , __FILE__ , __LINE__ ) ;\
  return EXIT_FAILURE ;}} while (0)

void helloCPU()
{
  printf("Hello from the CPU.\n");
}
/*
 * The addition of `__global__` signifies that this function
 * should be launced on the GPU.
 */
 __global__ void helloid(){
   printf("Hello from the GPU.\n");
 }
 __host__ __device__ void addone(int &a){
  srand (time(NULL));
}
__global__ void helloGPU()
{ 
  int i=0;
  addone(i);
  printf("Hello from the GPU.\n");
  helloid<<<2,2>>>();
}

int main()
{
  helloCPU();


  /*
   * Add an execution configuration with the <<<...>>> syntax
   * will launch this function as a kernel on the GPU.
   */
  helloGPU<<<2, 2>>>();
  /*
   * `cudaDeviceSynchronize` will block the CPU stream until
   * all GPU kernels have completed.
   */

  cudaDeviceSynchronize();
}
