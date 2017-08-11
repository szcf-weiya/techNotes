all: say_hello say_hello_class say_hello_paras say_hello_paras_revise

say_hello.o: say_hello.cpp
	g++ -c $^

say_hello_class.o: say_hello_class.cpp
	g++ -c $^

say_hello_paras.o: say_hello_paras.cpp
	g++ -c $^

say_hello_paras_revise.o: say_hello_paras_revise.cpp
	g++ -c $^

say_hello_class: say_hello_class.o
	g++ -o $@ $^ -lpthread

say_hello_paras: say_hello_paras.o
	g++ -o $@ $^ -lpthread


say_hello_paras_revise: say_hello_paras_revise.o
	g++ -o $@ $^ -lpthread

say_hello: say_hello.o
	g++ -o $@ $^ -lpthread

run:
	./say_hello_paras_revise
	#./say_hello_class
	#./say_hello
