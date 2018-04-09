#include<iostream>
#include<string>
#include<sstream>
using namespace std;

int main()
{
  string s = "3";
  stringstream ss(s);
  istringstream iss(s);
  int a;
  // for iss
  iss >> a;
  cout << a << endl;
  ss >> a;
  cout << a << endl;
  
  return 0;
}
