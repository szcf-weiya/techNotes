#include<iostream>
#include<string>

using namespace std;

int main()
{
    string s = "snp_12";
    size_t pos;
    pos = s.find("_", 0);
    cout << pos << endl 
        << s << endl
        << s.npos << endl
        << s.substr(0, s.npos) << endl;
    return 0;
}