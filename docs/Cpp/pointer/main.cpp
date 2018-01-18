#include <iostream>
using namespace std;

class Ptr
{
  int* m_p;
public:
  Ptr(int *p)
  {
    m_p = p;
  }
  int *operator -> ()
  {
    return m_p;    
  }
  int &operator * ()
  {
    return *m_p;
  }
  
    
};

int main()
{
  int a = 3;
  Ptr ptr = &a;

  cout << *ptr << endl;

  int* p = &a;
  cout << *p << endl;
  
}
