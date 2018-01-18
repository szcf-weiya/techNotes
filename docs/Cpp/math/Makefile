all: testcT testcppT 	testcF 

testcF: test.c
	gcc test.c -o test

testcT: test.c
	gcc test.c -o test -lm

testcppT: test.cpp
	g++ test.cpp -o test
