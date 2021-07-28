#include <iostream>
using namespace std;

int main()
{
    int array[3] = {1, 2, 3};
    // int &b = array; // WRONG!!
    int &b = *array; // left-hand side is `Int`, then right-hand side should also be int
    int c = *array;
    cout << array[0] << endl;
    cout << &array[0] << endl;
    cout << &b << endl;
    cout << &c << endl;
}