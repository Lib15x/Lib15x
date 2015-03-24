#include <iostream>
#include <Eigen/Dense>
using namespace std;

namespace CPPLearn{

  template <typename Type>
  class CLMatrixDynamic{

  public:
    CLMatrixDynamic(const int rows, const int cols) : eigenMatrix{rows, cols} {
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
    Eigen::Matrix<Type,Eigen::Dynamic,Eigen::Dynamic> eigenMatrix;
  };

}







