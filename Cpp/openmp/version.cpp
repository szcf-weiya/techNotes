// https://stackoverflow.com/questions/1304363/how-to-check-the-version-of-openmp-on-linux
#include <unordered_map>
#include <iostream>
#include <omp.h>
using namespace std;

int main()
{
    unordered_map<unsigned, string> map{{200505,"2.5"},{200805,"3.0"},{201107,"3.1"},{201307,"4.0"},{201511,"4.5"},{201811,"5.0"},{202011,"5.1"}};
    cout << "We have OpenMP " << map.at(_OPENMP) << endl;
}