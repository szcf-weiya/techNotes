# include <stdio.h>
# include <stdlib.h>
# include <memory.h>
# include <iostream>
using namespace std;

struct ClassBook{
	int number;
	int age;
}; 

struct ClassBook2{
	int number;
	int age;
	ClassBook2()
	{
		memset(this, 0, sizeof(ClassBook2));
	}
}; 



int main()
{
	ClassBook bookst={1005, 10};
	// or
	/*
	ClassBook bookst;
	bookst.number = 1001;
	bookst.age = 10;
	*/
	cout << bookst.number << "\t" << bookst.age << endl;
	
	// case 2: ERROR
	/*
	ClassBook2 bookst2 = {1001, 10};
	cout << bookst2.number << "\t" << bookst2.age << endl;
	*/
	
	ClassBook2 bookst2;
	bookst2.number = 1001;
	bookst2.age = 10;
	
	cout << bookst2.number << "\t" << bookst2.age << endl;
	
	return 0;
	
}
