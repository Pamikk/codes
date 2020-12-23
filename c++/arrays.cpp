#include <iostream>
using namespace std;
void assign(int *a[]){
    int b[2]={3,4};
    a[1]=b;
}
int sum(int a[]){
    return a[0]+a[1];
}

int main(){
    int a[3]={0,1,2};
    unsigned short b=0;
    for (int i=4;i>=0;i--){
        cout<<i<<endl;
    }
    return 0;

}