# C++ Notes

### GCC/G++ Commands

+ disable language extension: -pedantic-errors flag

+ increasing warning level: -Wall -Weffc++ -Wextra -Wsign-conversion 

+ treat warnings as errors: -Werror

### C++ Properties

+ direct memory access is not allowed: object(region of memory), store and retrive values; named one variable and its name is identifier.
    
    + instantiation: create vairable and assign memory address

    + instantiated object also called instance

+ Data Type:
        
    + integer,double etc.

    + must known at complie time

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
+  


