using namespace std;
struct node{
   int val;
   node * p;
   node * l;
   node * r;
};

class BSTree{
    node *root=0;
    public:
    bool insert(int);
    bool del(int);
    bool print(int,int);
    node* lookup(int);
};