#include <iostream>
using namespace std;
class Node{
    public:
       static int val1=0;
       void addvals(int val){
           val1+=val;
           val2+=val;
           val3+=val;
       }
       void print(){
           cout<<val1<<endl;
       }
    private:
        int val2 = 0;
    protected:
        int val3 = 0;
};
class childNode1: public Node{
    int val;
};
int main(){
    Node node1,node2;
    childNode1 node11;
    node1.addvals(1);
    node2.addvals(2);
    node1.print();
    node2.print();
    node11.addvals(11);
    node11.print();
    node1.print();
    return 0;
}