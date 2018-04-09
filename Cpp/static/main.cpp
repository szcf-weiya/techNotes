#include <iostream>
using namespace std;

class A
{
public:
  static void f() {cout << "f()" << endl;} ;
};

int main()
{
  A a;
  //  a::f();
  a.f();
  A::f();
}
