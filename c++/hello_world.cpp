#include <iostream>
#include <limits>
using namespace std;
int add(int a,int b);
int main(){
    cout<< "Hello Word!" << endl;
    int a(4.5);
    cout<< a <<endl;
    cout<< add(a,a)<<endl;
    /* cin.clear(); // reset any error flags
    cin.ignore(numeric_limits<streamsize>::max(), '\n'); // ignore any characters in the input buffer until we find an enter character
    cin.get(); // get one more char from the user */
    return 0;
}