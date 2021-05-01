#include "utils.cuh"
struct pair{
    unsigned int idx;
    unsigned int val;
};
bool cmpfunc(pair i,pair j){
    return (i.val<j.val);
}
__global__ void init_n(unsigned int* result,int n){
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    //if (id ==n-1){printf("%u\n",start);}
    if (id<n)result[id] = id;    
}
__global__ void init_n_gpu_batch(unsigned int* result,int n){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int threadsPerBlock = 64;
    const unsigned int blockCount = (n-1)/threadsPerBlock+1;
    if (id<28){
        init_n<<<blockCount,threadsPerBlock>>>(result+id*n,n);
    }
    
}
__global__ void swap(unsigned int * result,int n){
    unsigned id = (blockIdx.x * blockDim.x + threadIdx.x)%n;   
    curandStateXORWOW localState;
    curand_init((int)clock64(), id, 0, &localState);
    // Store result
    unsigned new_id = curand(&localState)%n;
    unsigned old = atomicExch(&result[new_id],result[id]);
    result[id]=old;
}
__global__ void sample_wo_rep_kernel(unsigned int* result,const int n,int m){
    unsigned *tmp;
    cudaMalloc(&tmp,sizeof(unsigned)*n);
    int block_num = 64;
    int threadsPerBlock = (n-1)/block_num+1;
    init_n<<< block_num,threadsPerBlock>>>(tmp,n);
    swap<<<block_num,threadsPerBlock>>>(tmp,n);
    cudaDeviceSynchronize();
    memcpy(result,tmp,m*sizeof(unsigned int));
}
__global__ void swap_even(unsigned short*data,unsigned int *idx,int num){
    int id = threadIdx.x + blockDim.x*blockIdx.x;
    int a;
    if ((2*id+1)<num){
        if (data[idx[2*id]]>data[idx[2*id+1]]){
            a = idx[2*id];
            idx[2*id]=idx[2*id+1];
            idx[2*id+1] = a;
        }
    }
}
__global__ void swap_odd(unsigned short*data,unsigned int *idx,int num){
    int id = threadIdx.x + blockDim.x*blockIdx.x;
    int a;
    if ((2*id+2)<num){
        if (data[idx[2*id+1]]>data[idx[2*id+2]]){
            a = idx[2*id+1];
            idx[2*id+1]=idx[2*id+2];
            idx[2*id+2] = a;
        }
    }
}
__global__ void sort_data_gpu(unsigned short*data,unsigned int *idx,int num){
    int i=0;
    const unsigned int threadsPerBlock = 64;
    const unsigned int blockCount = (num/2)/threadsPerBlock+1;
    for (i=0;i<num/2;i++){
        swap_even<<<blockCount,threadsPerBlock>>>(data,idx,num);
        swap_odd<<<blockCount,threadsPerBlock>>>(data,idx,num);    
    }
};
__global__ void sort_data_gpu_batch(unsigned short*data,unsigned int *idx,int num){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int i=0;
    const unsigned int threadsPerBlock = 64;
    const unsigned int blockCount = (num/2)/threadsPerBlock+1;
    for (i=0;i<num/2;i++){
        swap_even<<<blockCount,threadsPerBlock>>>(data+id*num,idx+id*num,num);
        swap_odd<<<blockCount,threadsPerBlock>>>(data+id*num,idx+id*num,num);    
    };
}
__global__ void sort_data_gpu_pair(pair* data,unsigned int *idx,int num){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int i=0;
    pair a;
    for (i=0;i<num/2;i++){
        if (2*id+1<num){
            if (data[2*id].val>data[2*id+1].val){
                a = data[2*id];
                data[2*id]=data[2*id+1];
                data[2*id+1] = a;
            }
        }
        __syncthreads();
        if (2*id+2<num){
            if (data[2*id+1].val>data[2*id+2].val){
                a = data[2*id+1];
                data[2*id+1]=data[2*id+2];
                data[2*id+2] = a;
            }
        }
        __syncthreads();
    }
    __syncthreads();
    if (2*id<num){
        idx[2*id] = data[2*id].idx;
    }
    if (2*id+1<num){
        idx[2*id+1] = data[2*id+1].idx;
    }     
};
__global__ void sort_data_gpu_batch_pair(unsigned short*data,unsigned int *idx,int num){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int threadsPerBlock = 64;
    const unsigned int blockCount = (num-1)/threadsPerBlock+1;
    pair *tmp = new pair[num];
    for (int i=0;i<num;i++){
        tmp[i].idx = idx[id*num+i];
        tmp[i].val = data[tmp[i].idx];
    }
    sort_data_gpu<<<blockCount,threadsPerBlock>>>(data+id*num,idx+id*num,num);
    delete tmp;
    cudaDeviceSynchronize();
}
void init_n_cpu(unsigned int* result,int n){
    for (unsigned i=0;i<n;i++){
        result[i] = i;
    }
}
void generate(unsigned int* result,int ub,int n){
    for (unsigned i=0;i<n;i++){
        result[i] = rand()%ub;
    }
}
void swap_cpu(unsigned int * result,int n){
    for (unsigned i=0;i<n;i++){
        // Store results
        unsigned new_id = rand()%n;
        unsigned old = result[i%n];
        result[i%n] = result[new_id];
        result[new_id] = old;
    }    
}
void sample_wo_rep(unsigned int* result,const int n,int m){
    unsigned *tmp;
    tmp = (unsigned*) malloc(sizeof(unsigned)*n);
    init_n_cpu(tmp,n);
    swap_cpu(tmp,n);
    for (int i=0;i<m;i++){
        result[i] = tmp[i];
    }
    free(tmp);
}

void sort_data_cpu(unsigned short*data,unsigned int *data_idx,int num){

    /*for (int i=0;i<num-1;i++){
        for (int j=0;j<num-i-1;j++){
            if (data[data_idx[j]]>data[data_idx[j+1]]){
                int tmp = data_idx[j];
                data_idx[j] = data_idx[j+1];
                data_idx[j+1] = tmp;
            }
        }
    }*/
    pair * tmp = new pair[num]; 
    for (int i=0;i<num;i++){
         tmp[i].idx = data_idx[i];
         tmp[i].val = data[data_idx[i]];
    }
    std::sort(tmp,tmp+num,cmpfunc);
    for (int i=0;i<num;i++){
        data_idx[i]= tmp[i].idx;
   }
}