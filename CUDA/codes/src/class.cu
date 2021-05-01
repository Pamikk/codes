#include <stdio.h>
#include <stdlib.h>
#include "class1.cuh"
int main(){
    size_t n = 10;
    size_t i;
    const unsigned int threadsPerBlock = 64;
    const unsigned int blockCount = 64;
    dataset ds;
    ds.size = (int) n;
    ds.test_cpu(n);
    ds.test_gpu(n);
    //cudaMemcpy(ds.host_data,ds.dev_data,sizeof(int)*n,cudaMemcpyDeviceToHost);
    for (i=0;i<n;i++){
        printf("%d\n",ds.host_data[i]);
    }
    cudaMemcpy(ds.host_data,ds.dev_data,sizeof(int)*n,cudaMemcpyDeviceToHost);
    for (i=0;i<n;i++){
        printf("%d\n",ds.host_data[i]);
    }
    /* Cleanup */
    //cudaFree (ds.dev_data);
    return 0;  
}