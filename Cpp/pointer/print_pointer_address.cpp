#include <iostream>
using namespace std;

int main()
{
    const char *pstr = "hello world";
    cout << pstr << endl;
    cout << static_cast<const void*>(pstr) << endl;
    return 0;
}