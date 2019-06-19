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
    + 


