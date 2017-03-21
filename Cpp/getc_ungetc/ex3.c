#include<stdio.h>
#include<ctype.h>
int main()
{
  int i=0;
  char ch;
  puts("Input an integer followed by a char:");
  // 读取字符直到遇到结束符或者非数字字符
  while((ch = getchar()) != EOF && isdigit(ch))
  {
    i = 10 * i + ch - 48; // 转为整数, 0的ASCII码为48
  }
  // 如果不是数字，则放回缓冲区
  if (ch != EOF)
  {
    ungetc(ch,stdin); // 把一个字符退回输入流
  }
  printf("\n\ni = %d, next char in buffer = %c\n", i, getchar());
  return 0;
}

/*
程序开始执行while循环，直到遇到非数字或者结束标识才能往下执行，紧接着判断是不是结束标识，如果不是结束标识则退回键盘缓冲区，在最后输出的时候使用getch()从缓冲区再次获取该字符输出。因为while中使用的是函数getchar()， 所以需要输入字符后按回车键。
 */
