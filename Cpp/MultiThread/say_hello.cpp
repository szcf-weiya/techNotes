#include <iostream>
#include <pthread.h>

using namespace std;

#define NUM_THREADS 5

// 函数指针
void* say_hello(void *args)
{
  cout << "hello ..." << endl;
}

int main(int argc, char const *argv[]) {
  pthread_t tids[NUM_THREADS];
  for (int i = 0; i < NUM_THREADS; i++)
  {
    // 线程的id，线程参数， 线程运行函数的起始地址，运行函数的参数
    int ret = pthread_create(&tids[i], NULL, say_hello, NULL);
    if (ret!=0)
    {
      cout << "pthread_create error: error_code = " << ret << endl;
    }
  }
  pthread_exit(NULL);
  return 0;
}
