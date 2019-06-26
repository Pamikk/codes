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


