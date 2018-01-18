//此程序仅作为“GDB十分钟教程”的示例代码, by liigo
//Email: liigo@sina.com
//blog: http://blog.csdn.net/liigo
//WebSite: www.liigo.com 

#include <stdio.h>

int nGlobalVar = 0;

int tempFunction(int a, int b)
{
    printf("tempFunction is called, a = %d, b = %d /n", a, b);
    return (a + b);
}

int main()
{
    int n;
    n = 1;
    n++;
    n--;

    nGlobalVar += 100;
    nGlobalVar -= 12;

    printf("n = %d, nGlobalVar = %d /n", n, nGlobalVar);

    n = tempFunction(1, 2);
    printf("n = %d", n);

    return 0;
}
