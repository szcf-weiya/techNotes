/*
http://blog.csdn.net/hackerain/article/details/7954006
 */

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[], char* envp[])
{
	int input;
	int n;
	int *numbers1;
	int *numbers2;
	numbers1=NULL;

	if((numbers2=(int *)malloc(5*sizeof(int)))==NULL)//为numbers2在堆中分配内存空间
	{
		printf("malloc memory unsuccessful");
		exit(1);
	}
	
	printf("numbers2 addr: %8X\n",(int)numbers2);

	for(n=0;n<5;n++) //初始化
	{
		*(numbers2+n)=n;
		printf("numbers2's data: %d\n",*(numbers2+n));
	}

	printf("Enter new size: ");
	scanf("%d",&input);

	//重新分配内存空间，如果分配成功的话，就释放numbers2指针,
	//但是并没有将numbers2指针赋为NULL,也就是说释放掉的是系统分配的堆空间，
	//和该指针没有直接的关系，现在仍然可以用numbers2来访问这部分堆空间，但是
	//现在的堆空间已经不属于该进程的了。
	numbers1=(int *)realloc(numbers2,(input+5)*sizeof(int));

	if(numbers1==NULL)
	{
		printf("Error (re)allocating memory");
		exit(1);
	}
	
	printf("numbers1 addr: %8X\n",(int)numbers1);
	for(n=0;n<5;n++) //test
	{
	  //		*(numbers2+n)=n;
		printf("numbers2's data: %d\n",*(numbers2+n));
	}
	printf("numbers2 addr: %8X\n",(int)numbers2);

	/*for(n=0;n<5;n++) //输出从numbers2拷贝来的数据
	{
		printf("the numbers1's data copy from numbers2: %d\n",*(numbers1+n));
	}*/

	for(n=0;n<input;n++)//新数据初始化
	{
		*(numbers1+5+n)=n+5;
		//printf("numbers1' new data: %d\n",*(numbers1+5+n));
	}

	printf("\n");

	free(numbers1);//释放numbers1，此处不需要释放numbers1，因为在realloc()时已经释放
	numbers1=NULL;
	//free(numbers2);//不能再次释放
	return 0;
}
