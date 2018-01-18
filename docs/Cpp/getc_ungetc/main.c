# include <stdio.h>

/*
函数getc()用于从流中取字符，其原型如下：
int getc(FILE *stream);
参数*steam为要从中读取字符的文件流
该函数执行成功后，将返回所读取的字符。
若从一个文件中读取一个字符，读到文件尾而无数据时便返回EOF。getc()与fgetc()作用相同，但在某些库中getc()为宏定义，而非真正的函数。

 */

void main(){
  char ch;
  printf ("Input: ");
  ch = getc(stdin);
  printf("Output:'%c'\n", ch);
}
