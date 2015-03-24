#include <iostream>
#include <Eigen/Dense>
using namespace std;

namespace CPPLearn{

  template <typename Type, unsigned Rows, unsigned Cols>
  class CLMatrixStatic{

  public:
    CLMatrixStatic() {
      eigenMatrix.fill(0);
    }

    size_t cols() const {
      return eigenMatrix.cols();
    }

    size_t rows() const {
      return eigenMatrix.rows();
    }

    const Type& operator()(size_t rowIndex, size_t colIndex) const {
      return eigenMatrix(rowIndex, colIndex);
    }

    Type& operator()(size_t rowIndex, size_t colIndex){
      return eigenMatrix(rowIndex, colIndex);
    }

    void display() const {
      cout<<eigenMatrix<<endl;
    }

  private:
    Eigen::Matrix<Type, Rows, Cols> eigenMatrix;
  };

}







