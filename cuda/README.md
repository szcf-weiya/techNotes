# CUDA notes

## Tutorials

1. https://devblogs.nvidia.com/parallelforall/even-easier-introduction-cuda/
2. https://devblogs.nvidia.com/parallelforall/easy-introduction-cuda-c-and-c/
3. http://docs.nvidia.com/cuda/cusolver/index.html


## Error Handle

### unified memory profiling failed

参考http://blog.csdn.net/u010837794/article/details/64443679

```
nvprof --unified-memory-profiling off ./add_cuda
```

### Strange Output

Pay attention to the size of vectors when running `cudaMemcpy()` .

and don't mess up the order of dimensions.


### cudaMemcpy fails

check whether the order of destination and source variables.

### cudaDeviceReset

think twice before adding the following code
```
cudaDeviceReset
```

### free(): invalid next size (fast/normal)

http://blog.sina.com.cn/s/blog_77f1e27f01019qq9.html

### Error [an illegal memory access was encountered]

多半数组越界，另外，注意对于double的数值型指针，

DO NOT

```
double *pone;
```

and DO NOT

```
double one;
double *pone = &one;
```
YOU MUST
```
double *pone = (double*)malloc(sizeof(double));
*pone = 1.0;
```

## help

1. [Element-by-element vector multiplication with CUDA](https://stackoverflow.com/questions/16899237/element-by-element-vector-multiplication-with-cuda)
2. [Is there a cuda function to copy a row from a Matrix in column major?](https://stackoverflow.com/questions/21002621/is-there-a-cuda-function-to-copy-a-row-from-a-matrix-in-column-major)
