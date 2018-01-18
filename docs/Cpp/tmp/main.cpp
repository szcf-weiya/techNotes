#include<iostream>
using namespace std;

class Base{
public:
  virtual int f1(char x) {return (int)x;}
  
};

class Derived: public Base{
public:
  virtual int f1(char x) {return (int)(-x);}
};

class base  
{  
public:  
    base(){cout<<"base::base()!"<<endl;}  
    void printBase(){cout<<"base::printBase()!"<<endl;}  
};  
  
class derived:public base  
{  
public:  
    derived(){cout<<"derived::derived()!"<<endl;}  
    void printDerived(){cout<<"derived::printDerived()!"<<endl;}  
};  

void print(Base& b)
{
  cout << b.f1('a');
}

int main()
{
    derived oo;  
    base oo1(static_cast<base>(oo));  
    oo1.printBase();  
    derived oo2 = static_cast<derived&>(oo1);  
    oo2.printDerived();  
  Base b;
  Derived d;
  print(b);
  print(d);
  char a = 'a';
  cout << (int)(-a) << endl;
  return 0;
}
