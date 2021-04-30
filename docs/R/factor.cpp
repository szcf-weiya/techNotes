#include <Rcpp.h>
#include <map>
using namespace Rcpp;

// [[Rcpp::export]]
IntegerVector f(NumericVector A) {
    std::map<float, int> visited;
    int count = 0;
    int n = A.size();
    IntegerVector B(n);
    for(int i = 0; i < n; i++) {
        if (visited.find(A[i]) == visited.end()) {
            count += 1;
            B[i] = count;
            visited[A[i]] = B[i];
        } else {
            B[i] = visited[A[i]];
        }
    }
    return B;
}

/*** R
f(c(20, 20, 20, 20))
*/
