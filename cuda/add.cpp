#include <iostream>
#include <cmath>

void add (int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<20; // 1M elements
  float *x = new float[N];
  float *y = new float[N];

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++)
  {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // run kernel on 1M elements on the CPU
  add(N, x, y);

  // check for errors

  float maxError = 0.0f;

  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max Error = " << maxError << '\n';

  // free memory
  delete [] x;
  delete [] y;
}
