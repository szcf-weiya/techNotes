#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define n 6

int main(int argc, char const *argv[]) {
  cudaError_t cudaStat;
  cublasStatus_t Stat;
  cublasHandle_t handle;
  int j;
  float *x;
  x = (float*)malloc(n*sizeof(float));
  for (j = 0; j < n; j++)
    x[j] = (float)j;
  printf("x: ");
  for (j = 0; j < n; j++)
    printf("%4.0f, ", x[j]);
  printf("\n");

  float *d_x;
  cudaStat = cudaMalloc(&d_x, n*sizeof(float));
  Stat = cublasCreate(&handle);
  Stat = cublasSetVector(n, sizeof(float), x, 1, d_x, 1);
  int result;

  Stat = cublasIsamax(handle, n, d_x, 1, &result);
  printf("max |x[i]|:%4.0f\n", fabs(x[result-1]));
  cudaFree(d_x);
  cublasDestroy(handle);
  free(x);
  return EXIT_SUCCESS;
}
