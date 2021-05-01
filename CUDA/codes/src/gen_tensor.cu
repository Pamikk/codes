#include <stdio.h>
__global__ void gen_tensor(float** tensor){
  float x = blockIdx.x+1;
  float y = threadIdx.x+1;
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  tensor[blockIdx][threadIdx] = sqrt(x*y);
}
 int main()
 { 
   const int n =2;
   const int dim = 3;
   float tensor[n][dim];
   //cudaMalloc(&tensor,sizeof(float)*(n*dim));
   gen_tensor<<<dim,n>>>(tensor);
   cudaDeviceSynchronize();
   float htensor = new float[n][dim];
   cudaMemcpy(htensor,tensor, sizeof(float)*(n*dim),cudaMemcpyDeviceToHost);
   for (int i=0;i<dim*n;i++){
      printf("%1.4f\n",htensor[i]);
   } 
 }
 