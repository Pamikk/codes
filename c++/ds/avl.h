using namespace std;
struct node{
    int val;
    node *l;
    node *r;
};
struct Tree_node{
   int val;
   Tree_node * p;
   Tree_node * l;
   Tree_node * r;
};

class BSTree{
    node *root=0;
    public:
    bool insert(int);
    bool del(int);
    bool print(int,int);
    node* lookup(int);
};