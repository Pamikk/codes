#include <iostream>
using namespace std;
typedef struct {
	float XYZW[4];
	float RGBA[4];
} Vertex;
void assign_pos(Vertex &v, float pos[4]) {
	for (int i = 0; i < 4; i++) {
		v.XYZW[i] = pos[i];
	}
}
int main(void){
    Vertex a;
    float pos[4] ={1,2,3,4};
    assign_pos(a,pos);
    cout<<a.XYZW[0]<<endl;
	int x,y;
	x=1;
    y=2;
	cout<<float(x)/float(y)<<endl;


}