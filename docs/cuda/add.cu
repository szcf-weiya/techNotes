#include <iostream>
#include <cmath>
__global__
void add (int n, float *x, float *y)
{
  //int index = threadIdx.x;
  //int stride = blockDim.x;
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i+= stride)
    y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<20; // 1M elements
  /*
  float *x = new float[N];
  float *y = new float[N];
  */
  // allocate unified memory
  float *x, *y;
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++)
  {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // run kernel on 1M elements on the CPU
  //add(N, x, y);
  int blockSize = 256;
  int numBlocks = (N+blockSize-1)/blockSize;
  add<<<numBlocks, blockSize>>>(N, x, y);
  //add<<<1, 256>>>(N, x, y);
  // wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  // check for errors

  float maxError = 0.0f;

  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max Error = " << maxError << '\n';

  // free memory
  /*
  delete [] x;
  delete [] y;
  */
  cudaFree(x);
  cudaFree(y);
}
