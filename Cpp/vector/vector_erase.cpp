// ref: http://blog.sina.com.cn/s/blog_6377b8e60100ino6.html

#include <vector>
#include <iostream>
using namespace std;

int main()
{
  vector <int> v1;
  vector <int>::iterator Iter;
  v1.push_back( 10 );
  v1.push_back( 20 );
  v1.push_back( 30 );
  v1.push_back( 40 );
  v1.push_back( 50 );
  cout << "v1 =" ;
  for ( Iter = v1.begin( ) ; Iter != v1.end( ) ; Iter++ ) 
    cout << " " << *Iter;
  cout << endl;
  v1.erase( v1.begin( ) );
  cout << "v1 =";
  for ( Iter = v1.begin( ) ; Iter != v1.end( ) ; Iter++ ) 
    cout << " " << *Iter;
  cout << endl;
  v1.erase( v1.begin( ) + 1, v1.begin( ) + 3 );
  cout << "v1 =";
 for ( Iter = v1.begin( ) ; Iter != v1.end( ) ; Iter++ ) 
   cout << " " << *Iter;
 cout << endl;
 //当调用erase()后Iter迭代器就失效了，变成了一野指针。
 //所以要处理这种问题，关键是要解决调用erase()方法后，Iter迭代器变成野指针的问题，
 //这个时候呢给他赋一个新的迭代器给他。
 v1.push_back( 10 );
 v1.push_back( 30 );
 v1.push_back( 10 );
 cout << "v1 =";
 for ( Iter = v1.begin( ) ; Iter != v1.end( ) ; Iter++ ) 
   cout << " " << *Iter;
 cout << endl; 
 for(Iter = v1.begin(); Iter != v1.end(); Iter++) 
   { 
     if(*Iter == 10) 
       { 
	 v1.erase(Iter);
	 Iter = v1.begin(); //当erase后，旧的容器会被重新整理成一个新的容器
	 // or
	 // Iter = v1.erase(Iter);
       }
     if (Iter == v1.end())
       break;
   }

 cout << "v1 =";
 for ( Iter = v1.begin( ) ; Iter != v1.end( ) ; Iter++ ) 
   cout << " " << *Iter;
 cout << endl; 
 // another one
 v1.erase(v1.begin()+1);
 cout << "v1 =";
 for ( Iter = v1.begin( ) ; Iter != v1.end( ) ; Iter++ ) 
   cout << " " << *Iter;
 cout << endl; 
 
 return 0;
  
}
