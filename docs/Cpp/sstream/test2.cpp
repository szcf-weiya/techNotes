#include <iostream>
#include <string>
#include <sstream>

using namespace std;

int main(int argc, char const *argv[]) {
  string line = "1 0 0 1 0 0 1 1\r";
  stringstream ss(line);
  string word, word1, word2;
  //ss >> word1 >> word2;
  cout << "pair results " << endl;
  while (!ss.eof()) {
    ss >> word1 >> word2;
    word = word1+word2;
    cout << word << endl;
  }
  cout << "single results " << endl;
  stringstream ss1(line);
  //ss.clear();
  //ss.str(line);
  while (!ss1.eof()) {
    ss1 >> word;
    //word = word1+word2;
    cout << word << endl;
  }
  return 0;
}
