#include <iostream>
#include <string>
#include <vector>
#include <sstream>
using namespace std;

void splitString(const string& s, vector<int> & v, const string &split)
{
    string::size_type pos1, pos2;
    pos2 = s.find(split);
    pos1 = 0;
    int tmp_i;
    string tmp_s;
    stringstream ss;
    while(string::npos != pos2)
    {
      ss.str("");
        tmp_s = s.substr(pos1, pos2-pos1);
        ss << tmp_s;
	//	cout << "tmp_s" << tmp_s << endl;
        ss >> tmp_i;
	//	cout << tmp_i << endl;
        v.push_back(tmp_i);
	
        pos1 = pos2 + split.size();
        pos2 = s.find(split, pos1);
    }
    if (pos1 != s.length())
      {
	ss.str("");
	tmp_s = s.substr(pos1);
	ss << tmp_s;
	//cout << "tmp_s" << tmp_s << endl;
	ss >> tmp_i;
	//cout << "tmp_i" << tmp_i << endl;
        v.push_back(tmp_i);
      }
}
void splitString2(const string& s, vector<string> & v, const string &split)
{
    string::size_type pos1, pos2;
    pos2 = s.find(split);
    pos1 = 0;
    string tmp_s;
    while(string::npos != pos2)
    {
        tmp_s = s.substr(pos1, pos2-pos1);
        v.push_back(tmp_s);
        pos1 = pos2 + split.size();
        pos2 = s.find(split, pos1);
    }
    if (pos1 != s.length())
        v.push_back(s.substr(pos1));
}


int main()
{
  string s = "1 2 3 4 5";
  vector<int> v;
  vector<string> v1;
  string split = " ";
  splitString(s, v, split);
  string s1 = "12";
  stringstream ss;
  istringstream iss(s);
  int a;
  ss << s1;
  ss >> a;
  cout << "a = " << a << endl;
  ss.str("");
  ss << s;
  for (int i = 0; i < 5; i++)
    {
      iss >> a;
      cout << a << endl;
    }
  for (size_t i = 0; i < v.size(); i++)
    cout << v[i] << endl;
  return 0;
}
