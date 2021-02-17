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
    for (int i=4;i>=0;i--){
        cout<<i<<endl;
    }
    int* b[3];
    int * c = new int(4);
    b[0] = c;
    cout<<*(b[0])<<endl;
    cout<< max(10,2)<<endl;
    return 0;

}