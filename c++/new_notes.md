# C++ Advanced(?) Notes

 Note based on <http://www.cplusplus.com/doc/tutorial/>, a quick review compared to notes.

## Datatype

- String
  - `include <string>`
  
  - initialization

    ```c++
    string mystring = "This is a string";
    string mystring ("This is a string");
    string mystring {"This is a string"};
    ```

- Compound Datatype
  - Array
    - series of elements of the same type placed in contiguous memory locations that can be individually referenced by adding an index to a unique identifier
    - Initialize 
      - type name [num of elements] 
      - type name = {elements}
    - Access
      - name[index]
    - Multidimensional arrays
      - type name [][]...
    - As parameters
      - void procedure (int arg[])
    - Library arrays
      - array<type,num> name {}
      - arr.size()
    
## Basic I/O

- cin
  - getline(cin,var)
- cout
- cerr
- clog
- `#include <sstream>`

    treat string as stream

## Control Flow

- if: see notes
  - switch

    ```c++
    switch(x){
        case 1:
            cout << "x is 1";
            break;
        case 2:
            cout << "x is 2";
            break;
        default:
            cout << "value of x unknown";
    }
    ```

- loops
  - while
  - do-while

  ```c++
  string str;
  do{
      cout << "Enter text: ";
      getline (cin,str);
      cout << "You entered: " << str << '\n';
  } while (str != "goodbye");
  ```

  - for
    - range-based loop

    ```c++
    string str {"Hello!"};
    for (char c : str){
        cout << "[" << c << "]";
    }
    ```

    or

    ```c++
    string str {"Hello!"};
    for (auto c : str){
        cout << "[" << c << "]";
    }
    ```

  - jump
    - break
    - continue
    - goto

      ```c++
      int main (){
          int n=10;
          mylabel:
              cout << n << ", ";
              n--;
              if (n>0) goto mylabel;
              cout << "liftoff!\n";
      }
      ```

## Function

- return value of main func
  - 0: return successfully
  - EXIT_SUCCESS: defined in `<cstdlib>`
  - EXIT_FAILURE: defined in `<cstdlib>`
- pass by value:
  - initial parameters by var value
  - not modify the value of original var
- pass by reference:
  - declare parameters as reference(&)
  - pass var itself
  - save memory by avoid copying
    - const reference forbid func to modify the var
- inline function
  - my understanding: copy function instead of calling funciton,ref: <https://www.cnblogs.com/litifeng/p/10300902.html>
  - same define in different files or define in header
- declare function
- overload function
  - same name but different input type or output type or function body
  - cannot be overloaded only by its return type. At least one of its parameters must have a different type.
  - function template

    ```c++
    template <class T>
    T sum (T a, T b){
        T result;
        result = a + b;
        return result;
    }
    template <class T, class U>
    bool are_equal (T a, U b){
        return (a==b);
    }
    ```

  - non-type template arguments
  
  ```c++
  #include <iostream>
  using namespace std;

  template <class T, int N>
  T fixed_multiply (T val){
      return val * N;
  }

  int main() {
      std::cout << fixed_multiply<int,2>(10) << '\n';
      std::cout << fixed_multiply<int,3>(10) << '\n';
  }
  ```
   
   
   



## Tricks

- auto and delctype
  - auto

    ```c++
    int foo = 0;
    auto bar = foo;  // the same as: int bar = foo;
    ```

  - delctype
  
    ```c++
    int foo = 0;
    delctype(foo) bar; //the same as int bar;
    ```

  - namespace and using

    ```c++
    #include <iostream>
    using namespace std;

    namespace first{
        int x = 5;
        int y = 10;
    }

    namespace second{
        double x = 3.1416;
        double y = 2.7183;
    }
    int main () {
        using first::x;
        using second::y;
        using namespace first;
        cout << x << '\n';
        cout << y << '\n';
        cout << first::y << '\n';
        cout << second::x << '\n';
        return 0;
    ```

## Concepts(?)

- Storage classes
  - static storage: allocated for the entire duration of the program
    - global variable
    - namespace(namespace can only defined in global scope)
  - automatic storage
    - local variable
- memory orgnization
  - structure:static heap->free memory<-stack
  - stack: local variable
  - heap: new/delete pointers




