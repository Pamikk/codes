# Notes for GPU Accelerating Computing
+ Ref:
+ Arrays of Parallel Threads
  + CUDA kernel is executed by a grid(array) of threads
    + all threads in a grid run the same kernel program
  + Thread Blocks
    + scalable cooperation
    + divede thread array into multiple blocks
      + within a block cooperate via shared mem, atomic operations ,barrier synchronization
      + Diff blocks do not interact.
      + 
      ```
        i=blockIdx.x*blockDim.x+threadIdx.x;//blockDim is size of block
        C[i] = A[i] + B[i];
      ```
      + kernel vs block
        + kernel is more like a function 
        + block is division of threads
      + blockIdx and thread Idx
        + each thread uses indices to decide what data to work on
        + simplifie mem addressing when processing multidimensional data
      + Exmaple: Vec Addition Kernel
        ```
        //Device Code
        __global__
        void vecAddKernel(float* A, float* B, float* C,int n){
            //
            int i = threadIdx.x + blockDim.x*blockIdx.x;
            if (i<n) c[i] = A[i] + B[i];
        }
        ```
        host allocate mem on the dev
        ```
        //Host Code
        __host__
        void vecAdd(float* H_A,float* h_B, float* h_C, int n){
            // d_A,d_B,d_C allocations and initialize

            //run ceil(n/256.0) blocks of 256 threads each
            dim3 DimGrid((n-1)/256+1,1,1);
            dim3 DimBlock(256,1,1);
            vecAddKernel<<<DimGrid,DimBLock>>>(d_A,d_B,d_C,n);

            //code copy results to h_c
        }
        ```
        <img src="imgs/screenshot1.png">

        + __global__ defines a kernle function
          + a kernel func must return void
        + __device__ and __host__ can be used together
        + __host__ is optional if used alone
    + Compiling A CUDA Program
      <img src="imgs/screenshot2.png">
    + 
