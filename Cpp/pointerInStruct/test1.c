# include <stdio.h>
# include <stdlib.h>

struct test{  
    int i;  
    short c;  
    char *p;
    int *p1;
    char s[10];  
};  
   
int main(){  
    struct test *pt=NULL;
    pt = malloc(sizeof(struct test));
    //pt->p = malloc(sizeof(char));//删掉这句后segmentation fault
    //   int i;
    int *p11;
    //    p11 = pt->p1;
    /*
    for (i=0; i<10;i++)
      {
	if (pt->p1 == NULL)
	  {
	    pt->p1 = malloc(sizeof(int));
	    //	    *(pt->p1) = 12;
	    pt->p1 = pt->p1 + 1;
	  }
      }
    */
    pt->p1 = malloc(sizeof(int));
    *(pt->p1) = 1;
    *(pt->p1 +15) = 12233;
    int *ptmp = pt->p1;
    *(ptmp + 5) = 34;
    int i=0;
    //while(1)
    //{
    //if (ptmp == NULL)
    //	break;
    // i = i + 1;
    //}
    printf("%d\n",i);
    //    pt->p+1 = malloc(sizeof(char));
    //  pt->p[1] = 'q';
    //    pt->p[2] = 'p';
    //    printf("&s = %x\n",pt->s); //等价于printf("%x\n", &(pt->s) );  
    //    printf("&i = %x\n",&pt->i); //因为操作符优先级，我没有写成&(pt->i)  
    printf("&c = %x\n",&pt->c);  
    printf("&p = %x\n",&pt->p);
    //    printf("p=%c\n",*(pt->p));
    //    printf("p=%c\n",*(pt->p+2));
    printf("p=%d\n",*(pt->p1));
    printf("p=%d\n",*(pt->p1+15));
    return 0;  
}  
/*
http://blog.csdn.net/yang_yulei/article/details/23395315
http://blog.csdn.net/imred/article/details/45441457
 */
