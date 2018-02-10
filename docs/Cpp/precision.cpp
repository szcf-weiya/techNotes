#include<cstdio>
#include<cstdlib>
#include<gsl/gsl_cdf.h>
#include <gsl/gsl_randist.h>

using namespace std;

int main()
{
    double a = -20.000;
    double b = 300;
    printf("%e\n", 1.0 - 2* gsl_cdf_tdist_P(a,b));

    return 0;
}