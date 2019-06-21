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
            
     + user-defined types

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
        
### Syntax and Error


