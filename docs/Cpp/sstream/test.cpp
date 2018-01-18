#include <sstream>
#include <iostream>
using namespace std;

int main()
{
    string line = "And the mome raths outgrabe.";
    string word;

    istringstream ls1(line);
    while (ls1 >> word, !ls1.eof()) {
        cout << "loop 1: " << word << endl;
    }
    cout << endl;
    istringstream ls2(line);
    while (ls2 >> word) {
        cout << "loop 2: " << word << endl;
    }
    cout << endl;
    istringstream ls3(line);
    while (!ls3.eof()) {
        ls3 >> word;
        cout << "loop 3: " << word << endl;
    }
}
