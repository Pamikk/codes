# C++ Notes
Notes based on https://www.learncpp.com/cpp-tutorial/

Reviewing C++ to prepare for learning Caffe and Opencv \_(:3\|—\|\_
### GCC/G++ Commands

+ disable language extension: -pedantic-errors flag

+ increasing warning level: -Wall -Weffc++ -Wextra -Wsign-conversion 

+ treat warnings as errors: -Werror

+ gcc need –l stdc++ flags
    + -g turn on debugging    
    + -c output an object file
    + -I specify an includedirectory
    + -L specify a libdirectory
    + -l link with libraylib.a
+ Multi-file Project
    + gcc/g++ main.cpp add.cpp -o main

### C++ Properties

+ direct memory access is not allowed: object(region of memory), store and retrive values; named one variable and its name is identifier.
    
    + instantiation: create vairable and assign memory address

    + instantiated object also called instance

+ Data Type:     
    + binary digits, bits
        + smallest unit of memory
        + 0,1
    + memory addresses, address
        +  orgnized memory sequential units
        + hold 1 byte of data
        + byte: comprised of 8 sequential bits.
    + must known at complie time to tell compiler how to interpret the corresponding memory.
    + fundamental data type
        + https://www.learncpp.com/cpp-tutorial/introduction-to-fundamental-data-types/
        + void, means no type
            + can not used to define var
            + usually to indicate that a function does not return a value,deprecated in c++
        + Sizes
            + https://www.learncpp.com/cpp-tutorial/object-sizes-and-the-sizeof-operator/
            + sizeof_op.cpp
            + signed integers
                + integers are signed by default
                + short, 1 byte signed -128 to 127, $[-2^7-2^7-1]$
                + int, 2 byte signed -32,768 to 32,767,$[-2^{15}-2^{15}-1]$
                + long, 4 byte signed -2,147,483,648 to 2,147,483,647, $[-2^{31}-2^{31}-1]$
                + long long, 8 byte signed -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807,  $[-2^{63}-2^{63}-1]$
                + integer overflow number out of range
                + integer division drop the fractional part
            + unsigned integers
                + e.g. unsigned short
                + short range [$0,2^8-1$] and so on.
                + unsigned overflow
                    
                    divided by the maximum of the corresponding datatype, and store the reamain.
                    
                    e.g. for short, 280 will be stored as
                    280 % 256 = 24.
            + fixed width integers and size_t

                unsigned integral type, typically used to represent the size or length of objects
            
            + float point number
                + Assuming IEEE 754 representation
                + float 4bytes
                + double 8 bytes
                + long double min:8bytes, typical 8, 12, or 16 bytes
            + boolean,bool
                + std::cin only accepts two inputs for Boolean variables: 0 and 1
            + chars
                + ASCII stands for American Standard Code for Information Interchange.
                0-127,https://www.learncpp.com/cpp-tutorial/chars/
                + 1 byte in size
                + usually signed(can be unsigned)
                + escape sequences. 
                    
                    An escape sequence starts with a ‘\’ (backslash)
                    + \n,\t,https://www.learncpp.com/cpp-tutorial/chars/
                + '' for char and "" for string
            + const not changed
                + c++11 constexpr
                    compile time constant
                + symbolic constant
                    + object-like macros
                    + constexpr

+ Variables

    + copy initialization:
        + int width =5;
    
    + direct intializaiton:
        + int width(5);\\\recomended befor c++11 for performance boost

        can not used for all types;
    
    + brace initialization (uniform initialization)
        + int wideth{5};
        + int width{};zero  initialzation to val 0
        + int width{4.5};\\\error

          int width(4.5);\\\drop fractional part

+ C++ Keywords
    + 84, https://www.learncpp.com/cpp-tutorial/keywords-and-naming-identifiers/

+ Literals and operators(operands)

+ expression: combination of literals, operators and variables,etc.
+ namespace
    + std::cout
    + using namespace std;

### Function

+ user-defined function
    + nested function is not allowed
+ return values or void
+ ignore codes after returning
+ function parameters and arguments
    + parameters: variable used in functions
    + argument:a value that is passed from the caller to the function when a function call is made
    + pass by value: function called, parameters created, values of arguments copied to parameters.
+ local variable
    + destroy in the opposite order of creation at the end of the set of curly braces in which they are defined (or for function parameters, at the end of the function).
    + lifetime: time between its creation and destruction. Note that variable creation and destruction happen when the program is running (called runtime), not at compile time
    + scope determines where the identifier can be accessed within the source code, compile-time property
    + forward declaration tell the compiler about the existence of an identifier before actually defining the identifier.
        + function prototype return type, name, parameters, but no function body, terminated with a semicolon.
        + forgetting function body will lead linker failed
        + delared in header file
        + header guard to avoid repeat include
            
            + #ifndef SOME_UNIQUE_NAME_HERE

                #define SOME_UNIQUE_NAME_HERE
            + some compliers support #pragma once




### Fromatting

+ Whitespace is a term that refers to characters that are used for formatting purposes.
    + refers primarily to spaces, tabs, and newlines
    + whitespace-independent language
+ Newlines are not allowed in quoted text:
    + std::cout << "Hello
      world!"; // Not allowed!
+ Quoted text separated by nothing but whitespace (spaces, tabs, or newlines) will be concatenated:
    + std::cout << "Hello "
     "world!"; // prints "Hello world!"
+ "\\\\" for single-line comment and "\\* *\\" for multi-line comments.
### Preprocessor
+ instructions that start with a # symbol and end with a newline (NOT a semicolon)
+ tell the preprocessor to perform specific particular text manipulation tasks
+ not the same but resemble c++ syntax
+ include
    + include hearder files
+ Macro defines
    + object-like macro define
        + with substitution text

            identifier replaced by substitution text
        + w/o substitution text

            identifier will be removed ad replaced by nothing
    + function-like macro define
    + Conditional compilation
        + #ifdef,#ifndef,#endif

            tell pre-processor to check if an identifier has been previously #defined
        + #if 0 

            exclude a block of code from being compiled
    + scope of #define

        resolved before compilation, from top to bottom on a file-by-file basis.

### Operators
+ precedence and associativity 

    https://www.learncpp.com/cpp-tutorial/31-precedence-and-associativity/
+ x++ and ++x
    + x++ postfix increment
        
        evaluate x and then increase x
        
        y = x++;\\y=x,x+=1
    + ++x prefix increment 
        
        increase x and then evaluate x

        y = ++x;\\ x+=1,y=x
+ conditional operator ?:
    + c?x:y
    + if c then x else y
+ 
### Control Flow
+ if
    + if (condition) statement1;
    + if (condition) statement1;else statement2;
    + if (condition1) statement1;
    
      else if (condtion2) statement2;

      else statement3;





