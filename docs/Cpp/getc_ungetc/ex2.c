# include <stdio.h>
# include <stdlib.h>
# include <string.h>

int main()
{
  int ch;
  int len;
  int i = 0;
  FILE *fstream;
  char msg[100] = "Hello!I have read this file.";
  fstream=fopen("test.txt","at+");
  if(fstream==NULL)
  {
    printf("read file test.txt failed!\n");
    exit(1);
  }

  /*getc从文件流中读取字符*/
  while( (ch = getc(fstream))!=EOF)
  {
    putchar(ch);
  }
  putchar('\n');

  len = strlen(msg);
  while(len>0)/*循环写入*/
    {
    putc(msg[i],fstream);
    putchar(msg[i]);
    len--;
    i++;
    }
  fclose(fstream);
  return 0;
  
}
