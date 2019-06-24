#include <iostream>
#include <iomanip> // for std::setprecision()
 
int main()
{
    std::cout << "bool:\t\t" << sizeof(bool) << " bytes\n";
    std::cout << "char:\t\t" << sizeof(char) << " bytes\n";
    std::cout << "wchar_t:\t" << sizeof(wchar_t) << " bytes\n";
    std::cout << "char16_t:\t" << sizeof(char16_t) << " bytes\n"; // C++11 only
    std::cout << "char32_t:\t" << sizeof(char32_t) << " bytes\n"; // C++11 only
    std::cout << "short:\t\t" << sizeof(short) << " bytes\n";
    std::cout << "int:\t\t" << sizeof(int) << " bytes\n";
    std::cout << "long:\t\t" << sizeof(long) << " bytes\n";
    std::cout << "long long:\t" << sizeof(long long) << " bytes\n"; // C++11/C99
    std::cout << "float:\t\t" << sizeof(float) << " bytes\n";
    std::cout << "double:\t\t" << sizeof(double) << " bytes\n";
    std::cout << "long double:\t" << sizeof(long double) << " bytes\n";
    std::cout << std::setprecision(16); // show 16 digits of precision
    std::cout << 3.33333333333333333333333333333333333333f <<'\n'; // f suffix means float
    std::cout << 3.33333333333333333333333333333333333333 << '\n'; // no suffix means double
    std::cout << 3.3f << '\n';// redundant part are the conversion from binary to decimal.
    std::cout << true << std::endl;
    std::cout << false << std::endl;
 
    std::cout << std::boolalpha; // print bools as true or false
 
    std::cout << true << std::endl;
    std::cout << false << std::endl;
    std::cout << std::noboolalpha;
    std::cout << true << std::endl;
    std::cout << false << std::endl;

    char ch{ 'a' };
    int i{ch};
    std::cout << ch << '\n';
    std::cout << static_cast<int>(ch) << std::endl;
    std::cout << ch << std::endl;
    std::cout << i << '\n';
    return 0;
}