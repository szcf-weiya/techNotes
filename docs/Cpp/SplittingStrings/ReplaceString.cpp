#include <iostream>
#include <string>

using namespace std;

bool replaceString(string &, const string& oldSubStr, const string& newSubStr);

int main()
{
  string s, oldSubStr, newSubStr;
  cin >> s >> oldSubStr >> newSubStr;

  bool isReplaced = replaceString(s, oldSubStr, newSubStr);

  if(isReplaced)
    cout << s << endl;
  else
    cout << "NO MATCHES" << endl;
  return 0;
}

bool replaceString(string& s, const string& oldSubStr, const string& newSubStr)
{
  bool isReplaced = false;

  int currentPosition = 0;
  while (currentPosition < s.length())
    {
      int position = s.find(oldSubStr, currentPosition);
      if (position == string::npos)
	return isReplaced;
      else
	{
	  s.replace(position, oldSubStr.length(), newSubStr);
	  currentPosition = position + newSubStr.length();
	  isReplaced = true;
	}
    }
  return isReplaced;
}
