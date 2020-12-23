package first.hello;
import java.util.HashMap;
import first.test.Test;
public class Hello{
    public String cont = "123";
    public static HashMap<Integer, int[]> map=new HashMap<Integer,int[]>();
    public static void init() {
        new Test(10);
    }
    public static byte boolToByte(boolean[] owned,int start,int end){
		byte res = 0;
		int i=0;
		while(i<8){
			boolean x = start+i<end ? owned[start+i]:false;
			res = (byte) (res << 1);
			res = (byte) (res | (x ? 1 : 0));
			i+=1;
		}
		return res;
	}
    public static void main(String[] args){
         init();
         boolean[] a={true,true,false,true};
         byte res = boolToByte(a, 0, 4);
    }
    public static void change(){
        int[] tmp = map.get(1);
        tmp[1]=0;
    }
    public static void check(){
        int[] aa = map.get(1);
        System.out.println(aa[1]);
    }
}
