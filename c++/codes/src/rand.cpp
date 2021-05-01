#include<stdio.h>
#include<stdlib.h>
#include<time.h>
void init_n(unsigned int* result,int n){
    for (unsigned i=0;i<n;i++){
        result[i] = i;

    }
}
void sample(unsigned int* result,int n){
    for (unsigned i=0;i<n;i++){
        result[i] = rand()%n;
    }
}
void swap(unsigned int * result,int n){
    for (unsigned i=0;i<n*2;i++){
        // Store results
        unsigned new_id = rand()%n;
        unsigned old = result[i%n];
        result[i%n] = result[new_id];
        result[new_id] = old;
    }    
}
void sample_wo_rep_kernel(unsigned int* result,const int n,int m){
    unsigned *tmp = new unsigned[n];
    init_n(tmp,n);
    swap(tmp,n);
    for (int i=0;i<m;i++){
        result[i] = tmp[i];
    }
    free(tmp);
}
int main(){
    const int n = 784;
    unsigned result[100*1024*20];
    printf("%d\n",100*1024*20);
    srand(time(NULL));
    sample(result,1000);    
    for (int i=0;i<28;i++){
        printf("%u ",result[i]);
    }
    printf("\n");
    return 0;
}