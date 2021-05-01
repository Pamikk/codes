#include <iostream>
#include <random>
#include <algorithm>
void my_shuffle(int *start, int *end){
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(start,end,g);
}
void my_sample(int *out,int n,int ub){
    std::random_device rd;
    std::mt19937 g(rd());
    generate_n(out,n,g);
    for (int i=0;i<n;i++){
        out[i] =(unsigned) out[i] %ub;
    }
}
void init_range_n(long long *start,int n){
    for (int i=0;i<n;i++){
        start[i] = i;
    }
}
int main(){
    int n = 784;
    long long arr[784];
    init_range_n(arr+10,n-10);
    std::cout <<arr[11]<<std::endl;
    int out[12];
    std::cout<<"++++++++++++++++++++++++++++++++\n";
    my_sample(out,10,60000);
    for (int i=0;i<10;i++){
        std::cout<<out[i]<<std::endl;
    }
    std::cout<<10%10<<std::endl;
    return 0;
}