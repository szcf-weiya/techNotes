CUDAFLAGS = -lcudart -lcublas -lcusolver
GSLFLAGS = -lgsl -lgslcblas

all: lu_ex

lu_ex.o: lu_ex.cu
	nvcc -c lu_ex.cu

lu_ex: lu_ex.o
	nvcc -o $@ $^ $(CUDAFLAGS) $(GSLFLAGS)

run:
	time ./lu_ex 100
	time ./lu_ex 1000
	time ./lu_ex 10000
