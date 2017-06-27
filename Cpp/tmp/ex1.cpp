#include<iostream>
using namespace std;

class A{
public:
  A& operator=(const A& r)
  {
    cout << "A";
  }
};


class B{
public:
  B& operator=(const B& r)
  {
    cout << "B";
  }
};

class C{
private:
    B b;
  A a;

  int c;
};

int main()
{
  C m, n;
  m = n;
  return 0;
}
