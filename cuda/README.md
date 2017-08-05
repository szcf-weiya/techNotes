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
