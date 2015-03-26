#include <core/CLMatrixStatic.hpp>
#include <core/CLMatrixDynamic.hpp>
#include <Eigen/Dense>
using namespace CPPLearn;
using namespace std;

int main(){
  Eigen::Matrix<double, 2, 2> m; m.fill(0);
  m(0,0) = 1;
  m(0,1) = 2;
  m(1,0) = 3;

  m(1,1) = 4;
  m=m*m.transpose();
  cout<<m<<endl;

}
