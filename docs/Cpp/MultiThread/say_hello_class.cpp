//
// 线程调用到函数在一个类中，那必须将该函数声明为静态函数函数
// 因为静态成员函数属于静态全局区，线程可以共享这个区域，故可以各自调用。
//

#include <iostream>
#include <pthread.h>

using namespace std;

#define NUM_THREADS 5
class hello
{
public:
  static void* say_hello(void* args)
  {
    cout << "hello ..." << endl;
  }
};

int main(int argc, char const *argv[]) {
  pthread_t tids[NUM_THREADS];
  for (int i = 0; i < NUM_THREADS;i++)
  {
    int ret = pthread_create(&tids[i], NULL, hello::say_hello, NULL);
    if(ret != 0)
    {
      cout << "pthread_create error: error_code " << ret << endl;
    }
  }
  pthread_exit(NULL);
  return 0;
}
