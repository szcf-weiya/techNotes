/*
 * https://stackoverflow.com/questions/27094612/cublas-matrix-inversion-from-device
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define PERR(call) \
  if (call) {\
   fprintf(stderr, "%s:%d Error [%s] on "#call"\n", __FILE__, __LINE__,\
      cudaGetErrorString(cudaGetLastError()));\
   exit(1);\
  }

#define ERRCHECK \
  if (cudaPeekAtLastError()) { \
    fprintf(stderr, "%s:%d Error [%s]\n", __FILE__, __LINE__,\
       cudaGetErrorString(cudaGetLastError()));\
    exit(1);\
  }

__global__ void
inv_kernel(float *a_i, float *c_o, int n)
{
  int *p = (int *)malloc(3*sizeof(int));
  int *info = (int *)malloc(sizeof(int));
  int batch;
  cublasHandle_t hdl;
  cublasStatus_t status = cublasCreate_v2(&hdl);
  printf("handle %d n = %d\n", status, n);
  info[0] = 0;
  batch = 1;
  float **a = (float **)malloc(sizeof(float *));
  *a = a_i;
  const float **aconst = (const float **)a;
  float **c = (float **)malloc(sizeof(float *));
  *c = c_o;
  status = cublasSgetrfBatched(hdl, n, a, n, p, info, batch);
  __syncthreads();
  printf("rf %d info %d\n", status, info[0]);
  status = cublasSgetriBatched(hdl, n, aconst, n, p,
      c, n, info, batch);
  __syncthreads();
  printf("ri %d info %d\n", status, info[0]);
  cublasDestroy_v2(hdl);
  printf("done\n");
}

static void
run_inv(float *in, float *out, int n)
{
  float *a_d, *c_d;

  PERR(cudaMalloc(&a_d, n*n*sizeof(float)));
  PERR(cudaMalloc(&c_d, n*n*sizeof(float)));
  PERR(cudaMemcpy(a_d, in, n*n*sizeof(float), cudaMemcpyHostToDevice));

  inv_kernel<<<1, 1>>>(a_d, c_d, n);

  cudaDeviceSynchronize();
  ERRCHECK;

  PERR(cudaMemcpy(out, c_d, n*n*sizeof(float), cudaMemcpyDeviceToHost));
  PERR(cudaFree(a_d));
  PERR(cudaFree(c_d));
}

int
main(int argc, char **argv)
{
  float c[9];
  float a[] = {
    1,   2,   3,
    0,   4,   5,
    1,   0,   6 };

  run_inv(a, c, 3);
  for (int i = 0; i < 3; i++){
    for (int j = 0; j < 3; j++) printf("%f, ",c[(3*i)+j]);
    printf("\n");}

  return 0;
}
